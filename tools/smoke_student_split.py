#!/usr/bin/env python
"""Smoke test for the two split-inference students (early / late) + the full
mean-scale hyperprior bottleneck, built modularly from configs.

Per split (configs/student_rtformer_{early,late}.yaml), all on synthetic data:
  #0 equivalence : (a) RTFormer.forward == forward_receiver(forward_sender)  [refactor
                   is lossless]; (b) an IDENTITY-bottleneck student reproduces the frozen
                   teacher's predictions exactly [split wiring is lossless].
  #1 forward     : student(x) returns preds + split target + reconstruction + rate info.
  #2 train       : rate-distortion loss (feature_mse + beta*bpp) drops; codec + aux
                   optimizers both step.
  #3 deployment  : real two-stream bitstream roundtrip (compress -> bytes -> decompress)
                   reproduces the forward reconstruction; estimated bpp ~= actual.

Exits non-zero on any failure.
"""
import os
import sys
import tempfile
import traceback

import _common  # noqa: F401  adds repo root to sys.path so `src` imports resolve
from omegaconf import OmegaConf
import torch

from src import _compat  # noqa: F401  applies the CompressAI import shim first
from src.utils import load_config
from src.models import build_model
from src.models.backbones.rtformer import RTFormer
from src.models.bottlenecks import build_bottleneck
from src.models.student import SplitStudent, build_student, split_in_channels
from src.losses.compression import rate_distortion_loss, bpp_from_info, feature_mse

SIZE = (64, 128)
# Run on CPU: the rANS range coder (compress/decompress) is CPU-only, and mixing the
# CUDA context with CompressAI's native coder segfaults on this NGC torch build. This is
# a fast correctness smoke; real training keeps forward/backward on GPU (verified working)
# and only moves to CPU for the bitstream eval — standard CompressAI practice.
DEVICE = "cpu"
RESULTS = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        RESULTS.append(True)
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()
        RESULTS.append(False)


def _teacher(cfg):
    cfg = cfg.copy()
    cfg.model.backbone.args.pretrained = None         # hermetic
    return build_model(cfg.model).to(DEVICE).eval()


def c0a_backbone_refactor_equivalence():
    torch.manual_seed(0)
    rt = RTFormer().to(DEVICE).eval()
    x = torch.randn(2, 3, *SIZE, device=DEVICE)
    with torch.no_grad():
        ref = rt(x)
        comp = rt.forward_receiver(*rt.forward_sender(x))
    for k in ref:
        d = (ref[k] - comp[k]).abs().max().item()
        assert d < 1e-5, f"refactor changed scale {k}: max|diff|={d:.2e}"


def _identity_student(teacher, split):
    bn = build_bottleneck(OmegaConf.create({"name": "identity", "args": {}}),
                          in_channels=split_in_channels(teacher, split))
    return SplitStudent(teacher, bn, split=split).to(DEVICE)


def make_split_checks(split, cfg):
    teacher = _teacher(cfg)
    student = build_student(cfg, teacher).to(DEVICE)

    def c0b_identity_student_equals_teacher():
        x = torch.randn(2, 3, *SIZE, device=DEVICE)
        ids = _identity_student(teacher, split)
        with torch.no_grad():
            tpred = teacher(x)["seg"] if isinstance(teacher(x), dict) and "seg" in teacher(x) else teacher(x)
            spred = ids(x)["preds"]["seg"]
        d = (tpred - spred).abs().max().item()
        assert d < 1e-4, f"[{split}] identity student != teacher: max|diff|={d:.2e}"

    def c1_forward():
        x = torch.randn(2, 3, *SIZE, device=DEVICE)
        out = student(x)
        assert "seg" in out["preds"], "no seg head in preds"
        assert out["recon"].shape == out["target"].shape, \
            f"recon {tuple(out['recon'].shape)} != target {tuple(out['target'].shape)}"
        assert "likelihoods" in out["info"] and {"y", "z"} <= set(out["info"]["likelihoods"]), \
            "hyperprior info missing y/z likelihoods"

    def c2_train():
        B = 2
        num_pixels = B * SIZE[0] * SIZE[1]
        opt = torch.optim.Adam(student.codec_parameters(), lr=1e-3)
        aux_opt = torch.optim.Adam(student.aux_parameters(), lr=1e-3)
        student.train()
        dists = []
        for step in range(60):
            x = torch.randn(B, 3, *SIZE, device=DEVICE)
            out = student(x)
            total, parts = rate_distortion_loss(out["recon"], out["target"].detach(),
                                                out["info"], beta=0.01, num_pixels=num_pixels)
            if "recon_aux" in out:                       # early + compress_s4: distill s4 too
                total = total + feature_mse(out["target_aux"].detach(), out["recon_aux"])
            opt.zero_grad(); total.backward(); opt.step()
            aux = student.aux_loss()
            aux_opt.zero_grad(); aux.backward(); aux_opt.step()
            dists.append(parts["distortion"].item())
            if step % 20 == 0:
                print(f"      [{split}] step {step:02d} dist={parts['distortion']:.4f} "
                      f"bpp={parts['bpp']:.4f} aux={aux.item():.1f}")
        assert torch.isfinite(parts["bpp"]), "bpp not finite"
        assert dists[-1] < dists[0] * 0.9, f"[{split}] distortion did not drop: {dists[0]:.4f}->{dists[-1]:.4f}"

    def c3_bitstream_roundtrip():
        # test the bottleneck class roundtrip directly (the exactness guarantee lives here;
        # the s4 bottleneck is the same class -> covered). Decoupled from student.compress payload.
        s = student.cpu().eval()
        s.update(force=True)
        bn = s.bottleneck
        B = 2
        num_pixels = B * SIZE[0] * SIZE[1]
        torch.manual_seed(1)
        feat = torch.randn(B, bn.in_channels, 16, 32)
        with torch.no_grad():
            recon_fwd, info = bn(feat)
            est_bpp = bpp_from_info(info, num_pixels).item()
            payload = bn.compress(feat)
            with tempfile.TemporaryDirectory() as d:
                total_bytes = 0
                for gi, group in enumerate(payload["strings"]):      # hyperprior: [y_strings, z_strings]
                    for i, sbytes in enumerate(group):
                        p = os.path.join(d, f"{gi}_{i}.bin")
                        with open(p, "wb") as fh:
                            fh.write(sbytes)
                        total_bytes += os.path.getsize(p)
            recon_dec = bn.decompress(payload)
        actual_bpp = total_bytes * 8 / num_pixels
        max_diff = (recon_fwd - recon_dec).abs().max().item()
        rel = abs(est_bpp - actual_bpp) / actual_bpp
        print(f"      [{split}] est_bpp={est_bpp:.5f} actual_bpp={actual_bpp:.5f} "
              f"({total_bytes}B) rel={rel*100:.1f}% roundtrip_max|diff|={max_diff:.2e}")
        assert max_diff < 1e-3, f"[{split}] roundtrip not exact: {max_diff:.2e}"
        assert rel < 0.25, f"[{split}] estimated bpp far from actual: {rel*100:.1f}%"
        student.to(DEVICE)

    return [
        (f"#0b [{split}] identity student == frozen teacher", c0b_identity_student_equals_teacher),
        (f"#1  [{split}] forward -> preds + target + recon + rate info", c1_forward),
        (f"#2  [{split}] train: RD loss drops, codec+aux step", c2_train),
        (f"#3  [{split}] hyperprior bitstream roundtrip + bpp match", c3_bitstream_roundtrip),
    ]


if __name__ == "__main__":
    print(f"[smoke_student_split] device={DEVICE} size={SIZE}")
    check("#0a RTFormer forward == forward_receiver(forward_sender)", c0a_backbone_refactor_equivalence)
    for split, path in (("neck", "configs/student_rtformer_late.yaml"),
                        ("early", "configs/student_rtformer_early.yaml")):
        cfg = load_config(path)
        for name, fn in make_split_checks(split, cfg):
            check(name, fn)
    n_ok = sum(RESULTS)
    print(f"\n[smoke_student_split] {n_ok}/{len(RESULTS)} checks passed")
    sys.exit(0 if n_ok == len(RESULTS) else 1)
