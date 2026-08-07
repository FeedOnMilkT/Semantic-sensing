#!/usr/bin/env python
"""Phase 2.5 D-arm: real-time latency/throughput benchmark for the fused MID split
(workflow §9.4). Measures the five pipeline segments at batch=1 streaming:

  ① sender forward   : backbone stem..stage3 (forward_sender_mid) + align_h/g_a + h_a
                       — GPU and CPU
  ② entropy encode   : everything in bottleneck.compress AFTER h_a (rANS z/y coding +
                       encoder-side hyper-synthesis + container packing) — CPU serial;
                       the pure-rANS subset and the full compress() are also reported
  ③ bitstream size   : bytes/frame, MEAN over the first --bytes-images real ADE20K val
                       images (stream length is content-dependent; latency is not) +
                       transmission delay at the --mbps channel assumptions
  ④ entropy decode   : container -> y_hat (rANS + hyper-synthesis) — CPU serial
  ⑤ receiver forward : g_s (_synthesize) + stage4/5 + DAPPM + neck/decoder/head
                       — GPU and CPU

End-to-end = ①(GPU) + ②(CPU) + tx(bytes/bandwidth) + ④(CPU) + ⑤(GPU), one row per
Mbps assumption, with equivalent fps. A JPEG baseline (encode Q --jpeg-q + decode +
FULL teacher forward on the receiver) runs the same protocol for comparison.

Latency uses a FIXED input (first val image) with --warmup >= 20 / --iters >= 100,
reporting median and P95; GPU timing brackets every call with torch.cuda.synchronize().
The first --ckpts entry gets the full latency treatment; extra entries (same
architecture, different beta) contribute bytes/e2e rows only.

  CUDA_VISIBLE_DEVICES=3 python tools/bench_latency.py \
    --ckpts /data/sidney/checkpoints/semantic-sensing/students/midf_b0.01_s2.pth \
            /data/sidney/checkpoints/semantic-sensing/students/midf_b0.03_s2.pth

NOTE: rANS coding is CPU-only (CompressAI range coder); update(force=True) is called
once (single-threaded — the CDF-table build segfaults under the shared host's default
OpenMP pool) before any compress/decompress.
"""
import argparse
import io
import os
import statistics
import struct
import time

from _common import pick_device
import numpy as np
import torch
from PIL import Image

from src import _compat  # noqa: F401  CompressAI import shim, caps thread pool
from src.codec.entropy_coder import set_coder, get_coder_name
from src.codec.patch_compressai import apply as apply_fast_rans
from src.models import build_model
from src.models.student import build_student
from src.datasets import build_dataset
from src.datasets.ade20k import _MEAN, _STD

_PC = time.perf_counter


# ---------------- timing primitives ----------------
def _stats(ts):
    """seconds list -> (median_ms, p95_ms)."""
    ts = sorted(ts)
    p95 = ts[min(len(ts) - 1, max(0, int(round(0.95 * len(ts))) - 1))]
    return statistics.median(ts) * 1e3, p95 * 1e3


def time_fn(fn, warmup, iters, cuda=False):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        if cuda:
            torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = _PC()
            fn()
            if cuda:
                torch.cuda.synchronize()
            ts.append(_PC() - t0)
    return _stats(ts)


def time_stages(stage_fn, warmup, iters):
    """stage_fn() -> {stage: seconds}. Returns {stage: [seconds]*iters} (CPU only)."""
    with torch.no_grad():
        for _ in range(warmup):
            stage_fn()
        acc = {}
        for _ in range(iters):
            for k, v in stage_fn().items():
                acc.setdefault(k, []).append(v)
    return acc


# ---------------- codec stage mirrors (FusedLatentBottleneck, magic RTFF) ----------------
# These replicate compress()/decompress() exactly but time each sub-call; a byte-for-
# byte equality assert against the real compress() guards against drift.
def encode_stages(bn, y, z, hw_h, hw_l):
    t = {}
    t0 = _PC(); z_strings = bn.entropy_bottleneck.compress(z);            t["z_enc"] = _PC() - t0
    t0 = _PC(); z_hat = bn.entropy_bottleneck.decompress(z_strings, z.shape[-2:]); t["z_dec"] = _PC() - t0
    t0 = _PC()
    means, scales = bn._hyper_params(z_hat, y.shape[-2:])
    indexes = bn.gaussian_conditional.build_indexes(scales)
    t["hyper"] = _PC() - t0
    t0 = _PC(); y_strings = bn.gaussian_conditional.compress(y, indexes, means=means); t["y_enc"] = _PC() - t0
    t0 = _PC()
    buf = bytearray(b"RTFF")
    buf += struct.pack(">HHHHHHHHB", *z.shape[-2:], *y.shape[-2:], *hw_h, *hw_l, len(y_strings))
    for strings in (z_strings, y_strings):
        for s in strings:
            buf += struct.pack(">I", len(s)) + s
    t["pack"] = _PC() - t0
    return bytes(buf), t


def decode_stages(bn, data):
    t = {}
    t0 = _PC()
    off = 4
    z_h, z_w, y_h, y_w, fh_h, fh_w, fl_h, fl_w, B = struct.unpack_from(">HHHHHHHHB", data, off)
    off += 17
    streams = []
    for _stream in range(2):
        ss = []
        for _b in range(B):
            (ln,) = struct.unpack_from(">I", data, off); off += 4
            ss.append(data[off:off + ln]); off += ln
        streams.append(ss)
    z_strings, y_strings = streams
    t["unpack"] = _PC() - t0
    t0 = _PC(); z_hat = bn.entropy_bottleneck.decompress(z_strings, (z_h, z_w)); t["z_dec"] = _PC() - t0
    t0 = _PC()
    means, scales = bn._hyper_params(z_hat, (y_h, y_w))
    indexes = bn.gaussian_conditional.build_indexes(scales)
    t["hyper"] = _PC() - t0
    t0 = _PC(); y_hat = bn.gaussian_conditional.decompress(y_strings, indexes, means=means); t["y_dec"] = _PC() - t0
    return y_hat, (fh_h, fh_w), (fl_h, fl_w), t


# ---------------- JPEG helpers (same chain as tools/jpeg_baseline.py) ----------------
def denorm_uint8(img):
    x = (img.cpu() * _STD + _MEAN).clamp(0, 1) * 255.0
    return x.round().byte().permute(1, 2, 0).numpy()


def jpeg_encode(uint8_hwc, q):
    buf = io.BytesIO()
    Image.fromarray(uint8_hwc).save(buf, format="JPEG", quality=int(q))
    return buf.getvalue()


def jpeg_decode(data):
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), np.uint8)


# ---------------- benchmark ----------------
def load_student(ckpt, teacher_cache):
    sd = torch.load(ckpt, map_location="cpu")
    cfg = sd["cfg"]
    if teacher_cache.get("obj") is None:
        cfg.model.backbone.args.pretrained = None
        teacher_cache["obj"] = build_model(cfg.model).eval()
    student = build_student(cfg, teacher_cache["obj"]).eval()
    student.load_state_dict(sd["student"])            # ckpt carries teacher weights too
    assert cfg.student.split == "mid" and hasattr(student.bottleneck, "_analyze"), \
        "bench_latency supports the mid split + rtformer_fused bottleneck"
    # CDF tables for the rANS coder; single-threaded (see module docstring)
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    student.update(force=True)
    torch.set_num_threads(n)
    return student, cfg, float(sd.get("beta", 0))


def measure_bytes(student, ds, n_images, jpeg_q):
    """Mean bytes/frame over the first n_images real val images (CPU sender +
    real compress) for the learned codec and for JPEG at the same resolution."""
    bn, bb = student.bottleneck, student.teacher.backbone
    fused, jpeg = [], []
    with torch.no_grad():
        for i in range(min(n_images, len(ds))):
            img = ds[i]["image"]
            feats = bb.forward_sender_mid(img.unsqueeze(0))
            fused.append(len(bn.compress(feats)))
            jpeg.append(len(jpeg_encode(denorm_uint8(img), jpeg_q)))
    return sum(fused) / len(fused), sum(jpeg) / len(jpeg)


def bench_resolution(student, ds, hw, args, device):
    """Returns {(segment, device): (median_ms, p95_ms)}."""
    bn, bb = student.bottleneck, student.teacher.backbone
    x = ds[0]["image"].unsqueeze(0)                    # fixed real input for latency
    res = {}

    # ---- CPU side ----
    student.cpu()
    with torch.no_grad():
        feats = bb.forward_sender_mid(x)
        y = bn._analyze(feats)
        z = bn.h_a(y)
        hw_h, hw_l = tuple(feats[0].shape[-2:]), tuple(feats[1].shape[-2:])
        data, _ = encode_stages(bn, y, z, hw_h, hw_l)
        assert data == bn.compress(feats), "encode mirror drifted from compress()"
        y_hat = decode_stages(bn, data)[0]

    res[("sender_fwd", "cpu")] = time_fn(
        lambda: bn.h_a(bn._analyze(bb.forward_sender_mid(x))), args.warmup, args.iters)

    enc = time_stages(lambda: encode_stages(bn, y, z, hw_h, hw_l)[1], args.warmup, args.iters)
    res[("entropy_enc", "cpu")] = _stats([sum(v) for v in zip(*enc.values())])
    res[("entropy_enc_rans", "cpu")] = _stats(
        [a + b + c for a, b, c in zip(enc["z_enc"], enc["z_dec"], enc["y_enc"])])
    res[("compress_total", "cpu")] = time_fn(lambda: bn.compress(feats), args.warmup, args.iters)

    dec = time_stages(lambda: decode_stages(bn, data)[3], args.warmup, args.iters)
    res[("entropy_dec", "cpu")] = _stats([sum(v) for v in zip(*dec.values())])
    res[("entropy_dec_rans", "cpu")] = _stats(
        [a + b for a, b in zip(dec["z_dec"], dec["y_dec"])])
    res[("decompress_total", "cpu")] = time_fn(lambda: bn.decompress(data), args.warmup, args.iters)

    res[("receiver_fwd", "cpu")] = time_fn(
        lambda: student._receiver(bn._synthesize(y_hat, hw_h, hw_l), {}), args.warmup, args.iters)

    # JPEG codec timing on the same fixed image (CPU, PIL)
    u8 = denorm_uint8(x[0])
    jdata = jpeg_encode(u8, args.jpeg_q)
    res[("jpeg_enc", "cpu")] = time_fn(lambda: jpeg_encode(u8, args.jpeg_q), args.warmup, args.iters)
    res[("jpeg_dec", "cpu")] = time_fn(lambda: jpeg_decode(jdata), args.warmup, args.iters)

    # ---- GPU side ----
    if device != "cpu":
        student.to(device)
        xg, yg = x.to(device), y_hat.to(device)
        res[("sender_fwd", "cuda")] = time_fn(
            lambda: bn.h_a(bn._analyze(bb.forward_sender_mid(xg))), args.warmup, args.iters, cuda=True)
        res[("receiver_fwd", "cuda")] = time_fn(
            lambda: student._receiver(bn._synthesize(yg, hw_h, hw_l), {}), args.warmup, args.iters, cuda=True)
        res[("teacher_fwd", "cuda")] = time_fn(
            lambda: student.teacher(xg), args.warmup, args.iters, cuda=True)
        student.cpu()
    else:
        res[("teacher_fwd", "cpu")] = time_fn(lambda: student.teacher(x), args.warmup, args.iters)
    return res


def e2e_rows(seg, nbytes, mbps_list, chain):
    """chain = [(segment, device), ...]; returns [(mbps, median_ms, p95_ms, fps)]."""
    rows = []
    for mbps in mbps_list:
        tx = nbytes * 8 / (mbps * 1e6) * 1e3
        med = sum(seg[c][0] for c in chain) + tx
        p95 = sum(seg[c][1] for c in chain) + tx
        rows.append((mbps, med, p95, 1e3 / med))
    return rows


def main():
    default_dir = "/data/sidney/checkpoints/semantic-sensing/students"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="+", default=[os.path.join(default_dir, "midf_b0.01_s2.pth")],
                    help="student ckpts; first = full latency, rest = bytes/e2e rows only")
    ap.add_argument("--resolutions", nargs="+", default=["512x512", "1024x512"], help="WxH")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--bytes-images", type=int, default=50, help="real val images for bytes/frame")
    ap.add_argument("--cpu-threads", type=int, default=8)
    ap.add_argument("--jpeg-q", type=int, default=30)
    ap.add_argument("--mbps", type=float, nargs="+", default=[1.0, 5.0, 10.0])
    ap.add_argument("--out", default=os.path.join(default_dir, "_orch", "bench_latency.csv"))
    ap.add_argument("--coder", choices=["auto", "fast", "compressai"], default="auto",
                    help="entropy coder backend for compress/decompress (default auto)")
    args = ap.parse_args()
    apply_fast_rans()
    set_coder(None if args.coder == "auto" else args.coder)
    print(f"[bench] entropy_coder={get_coder_name()} (requested={args.coder})")

    device = args.device or pick_device()
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = True             # fixed shapes -> pick fastest conv algos
    gpu = torch.cuda.get_device_name(0) if device != "cpu" else "-"
    cpu = next((ln.split(":")[1].strip() for ln in open("/proc/cpuinfo") if "model name" in ln), "?")
    print(f"[bench] device={device} ({gpu})  cpu={cpu} (threads={args.cpu_threads})")
    print(f"[bench] iters={args.iters} warmup={args.warmup} bytes_images={args.bytes_images} "
          f"jpeg_q={args.jpeg_q} batch=1")

    teacher_cache = {}
    latency = {}                                      # resolution -> segments (first ckpt)
    csv_rows = []                                     # method,beta,resolution,device,segment,...
    for ci, ckpt in enumerate(args.ckpts):
        student, cfg, beta = load_student(ckpt, teacher_cache)
        name = os.path.basename(ckpt)
        for rs in args.resolutions:
            Wr, Hr = (int(v) for v in rs.lower().split("x"))
            cfg.data.val.args.size = [Hr, Wr]
            ds = build_dataset(cfg.data.val)

            if ci == 0:
                latency[rs] = bench_resolution(student, ds, (Hr, Wr), args, device)
            seg = latency[rs]                         # same architecture -> latency reused
            fused_b, jpeg_b = measure_bytes(student, ds, args.bytes_images, args.jpeg_q)
            npx = Hr * Wr

            print(f"\n== {name} (beta={beta:g}) @ {rs} ==")
            if ci == 0:
                print(f"{'segment':<18}{'device':<8}{'median_ms':>10}{'p95_ms':>10}")
                for (s, d), (med, p95) in seg.items():
                    print(f"{s:<18}{d:<8}{med:>10.2f}{p95:>10.2f}")
                    csv_rows.append(("fused", beta, rs, d, s, f"{med:.3f}", f"{p95:.3f}", "", ""))
            print(f"bytes/frame: fused={fused_b:.0f} (bpp={fused_b*8/npx:.4f})  "
                  f"jpeg_q{args.jpeg_q}={jpeg_b:.0f} (bpp={jpeg_b*8/npx:.4f})  [n={args.bytes_images} real val]")
            csv_rows.append(("fused", beta, rs, "-", "bytes", "", "", f"{fused_b:.0f}", ""))
            if ci == 0:
                csv_rows.append(("jpeg", args.jpeg_q, rs, "-", "bytes", "", "", f"{jpeg_b:.0f}", ""))

            dv = "cuda" if device != "cpu" else "cpu"
            fused_chain = [("sender_fwd", dv), ("entropy_enc", "cpu"),
                           ("entropy_dec", "cpu"), ("receiver_fwd", dv)]
            jpeg_chain = [("jpeg_enc", "cpu"), ("jpeg_dec", "cpu"), ("teacher_fwd", dv)]
            print(f"end-to-end (sender@{dv} + enc + tx + dec + receiver@{dv}):")
            for mbps, med, p95, fps in e2e_rows(seg, fused_b, args.mbps, fused_chain):
                print(f"  fused @{mbps:g}Mbps: {med:8.2f} ms (p95 {p95:8.2f})  {fps:6.2f} fps")
                csv_rows.append(("fused", beta, rs, dv, f"e2e@{mbps:g}Mbps",
                                 f"{med:.3f}", f"{p95:.3f}", f"{fused_b:.0f}", f"{fps:.2f}"))
            if ci == 0:
                for mbps, med, p95, fps in e2e_rows(seg, jpeg_b, args.mbps, jpeg_chain):
                    print(f"  jpeg  @{mbps:g}Mbps: {med:8.2f} ms (p95 {p95:8.2f})  {fps:6.2f} fps")
                    csv_rows.append(("jpeg", args.jpeg_q, rs, dv, f"e2e@{mbps:g}Mbps",
                                     f"{med:.3f}", f"{p95:.3f}", f"{jpeg_b:.0f}", f"{fps:.2f}"))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("method,beta_or_q,resolution,device,segment,median_ms,p95_ms,bytes_per_frame,fps\n")
        for r in csv_rows:
            f.write(",".join(str(v) for v in r) + "\n")
    print(f"\n[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
