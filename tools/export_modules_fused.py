#!/usr/bin/env python
"""Per-module ONNX export for fused mid-split GDN student (B-arm §9.2 de-risk).

Exports the midf_b0.01_s2 (or any fused mid ckpt) into isolated subgraphs so each
can be parsed independently inside the Hailo suite container. Host-side only:
PyTorch -> static ONNX (opset 13) -> onnxsim -> ORT numeric check -> manifest.json.

Modules (risk order, sender first):
  sender_backbone       stem..stage3 pure conv
  bn_align_h            align_h (GDN)
  bn_g_a                joint analysis after concat (GDN in first block)
  bn_h_a                hyper analysis (conv+ReLU)
  bn_h_s                hyper synthesis (conv+ReLU, receiver-side after entropy)
  bn_g_s                synthesis trunk + dual heads (IGDN x3)
  receiver_stage4       stage4 attention; consumes reconstructed mid payload
  receiver_stage5       stage5 attention; consumes receiver_stage4 outputs
  receiver_dappm        DAPPM context fuse
  receiver_head         passthrough neck + identity decoder + seg head

  python tools/export_modules_fused.py --help
  python tools/export_modules_fused.py
  python tools/export_modules_fused.py --only bn_align_h bn_g_s
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import _compat  # noqa: F401
from src.models import build_model
from src.models.student import build_student


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

# The frozen fused checkpoint mounted into the deployment/training shared volume.
# Callers can still override this via --student-ckpt for a newly trained candidate.
_DEFAULT_CKPT = "/local/shared_with_docker/ckpt/midf_b0.01_s2.pth"
_DEFAULT_OUT = "/data/sidney/checkpoints/semantic-sensing/hailo/onnx/modules"

# ONNX op types that map to GDN/IGDN divisive norm (Hailo high-risk).
GDN_OPS = frozenset({"Pow", "Sqrt", "Div", "Mul"})
ATTN_OPS = frozenset({"MatMul", "Softmax", "Gemm", "Einsum"})
RESIZE_OPS = frozenset({"Resize"})


@dataclass
class ModuleSpec:
    name: str
    desc: str
    module: nn.Module
    inputs: tuple
    input_names: list[str]
    output_names: list[str] | None = None
    hailo_risk: str = "low"          # low | medium | high
    risk_note: str = ""


@dataclass
class ExportResult:
    name: str
    path: str
    nodes: int
    size_kb: float
    simplified: bool
    max_abs: float | None
    ops: list[str]
    risky_ops: dict[str, int] = field(default_factory=dict)
    gdn_ops: dict[str, int] = field(default_factory=dict)
    in_shapes: list = field(default_factory=list)
    out_shapes: list = field(default_factory=list)
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    sha256: str = ""
    opset: int = 13
    hailo_risk: str = "low"
    risk_note: str = ""
    ort_ok: bool | None = None
    error: str | None = None


# ---- wrapper subgraphs ----

class SenderBackbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, image):
        return self.backbone.forward_sender_mid(image)


class AlignH(nn.Module):
    def __init__(self, align_h):
        super().__init__()
        self.align_h = align_h

    def forward(self, x_h):
        return self.align_h(x_h)


class GAAnalysis(nn.Module):
    def __init__(self, g_a):
        super().__init__()
        self.g_a = g_a

    def forward(self, aligned_x_h, x_l):
        if aligned_x_h.shape[-2:] != x_l.shape[-2:]:
            aligned_x_h = F.interpolate(
                aligned_x_h, size=x_l.shape[-2:], mode="bilinear", align_corners=False)
        return self.g_a(torch.cat([aligned_x_h, x_l], dim=1))


class HAAnalysis(nn.Module):
    def __init__(self, h_a):
        super().__init__()
        self.h_a = h_a

    def forward(self, y):
        return self.h_a(y)


class HSSynthesis(nn.Module):
    def __init__(self, h_s):
        super().__init__()
        self.h_s = h_s

    def forward(self, z_hat):
        out = self.h_s(z_hat)
        means, scales = out.chunk(2, dim=1)
        return means, scales


class GSSynthesis(nn.Module):
    """g_s trunk + dual reconstruction heads (bounded IGDN on synthesis path)."""
    def __init__(self, bottleneck, hw_h, hw_l):
        super().__init__()
        self.bottleneck = bottleneck
        self.hw_h = tuple(hw_h)
        self.hw_l = tuple(hw_l)

    def forward(self, y_hat):
        return self.bottleneck._synthesize(y_hat, self.hw_h, self.hw_l)


class ReceiverStage(nn.Module):
    def __init__(self, stage):
        super().__init__()
        self.stage = stage

    def forward(self, high, low):
        return self.stage(high, low)


class ReceiverDappm(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.dappm = backbone.dappm

    def forward(self, high, low):
        ctx = F.interpolate(self.dappm(low), size=high.shape[-2:],
                            mode="bilinear", align_corners=False)
        return F.relu(high + ctx, inplace=True)


class ReceiverHead(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        self.neck = teacher.neck
        self.decoder = teacher.decoder
        self.seg_head = teacher.heads["seg"]

    def forward(self, s8):
        feats = {"s8": s8}
        return self.seg_head(self.decoder(self.neck(feats)))


def _t(*shape):
    return torch.randn(*shape)


def load_student(ckpt_path, device="cpu", gfa_mode=None):
    # This trusted project checkpoint contains its OmegaConf experiment config;
    # PyTorch >=2.6 otherwise rejects it via the weights_only=True default.
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = sd["cfg"]
    cfg.model.backbone.args.pretrained = None
    if gfa_mode is not None:
        cfg.model.backbone.args.gfa_mode = gfa_mode
    teacher = build_model(cfg.model).eval()
    student = build_student(cfg, teacher).eval()
    student.load_state_dict(sd["student"])
    assert cfg.student.split == "mid", f"expected mid split, got {cfg.student.split!r}"
    assert cfg.bottleneck.name == "rtformer_fused", \
        f"expected rtformer_fused, got {cfg.bottleneck.name!r}"
    act = cfg.bottleneck.args.get("act", "convbn")
    return student, act, cfg


def build_specs(student, H, W, cfg):
    """Spatial sizes for static 512x512 deploy square (ADE20K val eval shape)."""
    bb = student.teacher.backbone
    bn = student.bottleneck
    t = student.teacher
    bargs = cfg.bottleneck.args

    h8, w8 = H // 8, W // 8
    h16, w16 = H // 16, W // 16
    h32, w32 = H // 32, W // 32

    c2 = bb.out_channels["s8"]    # 128
    c3 = bb.out_channels["s16"]   # 256
    c4 = bb.out_channels["s32"]   # 512
    y_c = bargs.get("latent_channels", 192)
    z_c = bargs.get("hyper_channels", 128)

    # y @ 1/32, z @ 1/32 with hyper_down=1 -> z spatial is half of y
    y_h, y_w = h32, w32
    z_h, z_w = y_h // 2, y_w // 2

    return [
        ModuleSpec(
            "sender_backbone",
            "stem..stage3 纯卷积 sender (Hailo-foldable)",
            SenderBackbone(bb).eval(),
            (_t(1, 3, H, W),),
            ["image"],
            ["x_h", "x_l"],
            "low",
            "Conv/Add/Relu/Resize; demo3 conv_sanity 同类已通过 parse",
        ),
        ModuleSpec(
            "bn_align_h",
            "align_h stride-2 + GDN (encoder)",
            AlignH(bn.align_h).eval(),
            (_t(1, c2, h8, w8),),
            ["x_h"],
            ["aligned_x_h"],
            "high",
            "GDN: Pow/Sqrt/Div/Mul — Hailo 无原生 GDN,需 convbn 重训或自定义层",
        ),
        ModuleSpec(
            "bn_g_a",
            "concat(aligned_x_h,x_l) -> g_a -> y (首块 GDN)",
            GAAnalysis(bn.g_a).eval(),
            (_t(1, c2, h16, w16), _t(1, c3, h16, w16)),
            ["aligned_x_h", "x_l"],
            ["latent_y"],
            "high",
            "g_a 首块 GDN + 后续 Conv; 与 bn_align_h 同风险族",
        ),
        ModuleSpec(
            "bn_h_a",
            "hyper analysis y -> z (Conv+ReLU)",
            HAAnalysis(bn.h_a).eval(),
            (_t(1, y_c, y_h, y_w),),
            ["latent_y"],
            ["latent_z"],
            "low",
            "标准 Conv+ReLU 栈,无 GDN",
        ),
        ModuleSpec(
            "bn_h_s",
            "hyper synthesis z_hat -> means, scales (receiver, post-entropy)",
            HSSynthesis(bn.h_s).eval(),
            (_t(1, z_c, z_h, z_w),),
            ["latent_z_hat"],
            ["means", "scales"],
            "low",
            "Conv+ReLU synthesis; Gaussian dequant 在 CPU,不在 Hailo 图内",
        ),
        ModuleSpec(
            "bn_g_s",
            "y_hat -> x_h_hat, x_l_hat (IGDN x3 + Resize)",
            GSSynthesis(bn, (h8, w8), (h16, w16)).eval(),
            (_t(1, y_c, y_h, y_w),),
            ["latent_y_hat"],
            ["x_h_hat", "x_l_hat"],
            "high",
            "BoundedIGDN x3: Pow/Sqrt/Div/Mul + bilinear Resize; receiver 侧最大 GDN 风险",
        ),
        ModuleSpec(
            "receiver_stage4",
            "stage4 RTFormer attention; mid reconstruction -> stage4 pair",
            ReceiverStage(bb.stage4).eval(),
            (_t(1, c2, h8, w8), _t(1, c3, h16, w16)),
            ["x_h", "x_l"],
            ["stage4_high", "stage4_low"],
            "medium",
            "receiver cascade first half; output contract feeds receiver_stage5 exactly",
        ),
        ModuleSpec(
            "receiver_stage5",
            "stage5 RTFormer attention; stage4 pair -> pre-DAPPM pair",
            ReceiverStage(bb.stage5).eval(),
            (_t(1, c2, h8, w8), _t(1, c4, h32, w32)),
            ["stage4_high", "stage4_low"],
            ["high_feat", "low_feat"],
            "medium",
            "receiver cascade second half; outputs feed receiver_dappm exactly",
        ),
        ModuleSpec(
            "receiver_dappm",
            "DAPPM(s32) context fuse into s8",
            ReceiverDappm(bb).eval(),
            (_t(1, c2, h8, w8), _t(1, c4, h32, w32)),
            ["high_feat", "low_feat"],
            ["s8"],
            "low",
            "Conv/Add/Relu/Resize; demo3 full_backbone 同类已通过",
        ),
        ModuleSpec(
            "receiver_head",
            "passthrough(s8) + identity decoder + seg head (3x3+1x1, up x8)",
            ReceiverHead(t).eval(),
            (_t(1, c2, h8, w8),),
            ["s8"],
            ["seg_logits"],
            "low",
            "Conv/BN/ReLU/Resize; demo3 full_net_seg_depth 头尾已通过 parse",
        ),
    ]


def _risky_counts(op_types):
    c = Counter(op_types)
    gdn = {k: c[k] for k in sorted(GDN_OPS & c.keys())}
    attn = {k: c[k] for k in sorted(ATTN_OPS & c.keys())}
    resize = {k: c[k] for k in sorted(RESIZE_OPS & c.keys())}
    risky = {}
    risky.update(gdn)
    risky.update(attn)
    risky.update(resize)
    return risky, gdn


def export_one(spec: ModuleSpec, out_dir: str, opset: int, check_ort: bool) -> ExportResult:
    path = os.path.join(out_dir, f"{spec.name}.onnx")
    module = spec.module
    inputs = spec.inputs
    input_names = spec.input_names

    with torch.no_grad():
        ref = module(*inputs)
    ref = ref if isinstance(ref, tuple) else (ref,)
    out_names = spec.output_names or [f"out{i}" for i in range(len(ref))]

    torch.onnx.export(
        module, inputs, path,
        input_names=input_names, output_names=out_names,
        opset_version=opset, do_constant_folding=True,
        dynamic_axes=None,
        # Keep static opset-13 Hailo handoffs on the legacy exporter: the pinned
        # suite environment intentionally does not depend on onnxscript.
        dynamo=False,
    )

    import onnx
    m = onnx.load(path)
    onnx.checker.check_model(m)

    simplified = False
    try:
        from onnxsim import simplify
        sm, ok = simplify(m)
        if ok:
            onnx.save(sm, path)
            m = sm
            simplified = True
    except ImportError:
        pass

    max_abs = None
    ort_ok = None
    if check_ort:
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        feed = {n: x.detach().cpu().numpy() for n, x in zip(input_names, inputs)}
        ort_out = sess.run(out_names, feed)
        max_abs = max(
            float(np.abs(r.detach().cpu().numpy() - o).max()) for r, o in zip(ref, ort_out)
        )
        ort_ok = bool(max_abs < 1e-3)

    op_types = [n.op_type for n in m.graph.node]
    risky, gdn = _risky_counts(op_types)

    return ExportResult(
        name=spec.name,
        path=os.path.abspath(path),
        nodes=len(m.graph.node),
        size_kb=os.path.getsize(path) / 1024,
        simplified=simplified,
        max_abs=max_abs,
        ops=sorted(set(op_types)),
        risky_ops=risky,
        gdn_ops=gdn,
        in_shapes=[list(x.shape) for x in inputs],
        out_shapes=[list(t.shape) for t in ref],
        input_names=list(input_names),
        output_names=list(out_names),
        sha256=_sha256_file(path),
        opset=opset,
        hailo_risk=spec.hailo_risk,
        risk_note=spec.risk_note,
        ort_ok=ort_ok,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student-ckpt", default=_DEFAULT_CKPT)
    ap.add_argument("--shape", type=int, nargs=4, default=[1, 3, 512, 512],
                    help="network input N C H W")
    ap.add_argument("--out-dir", default=_DEFAULT_OUT)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--only", nargs="*", default=None, help="subset of module names")
    ap.add_argument("--no-ort", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="fail if any module export/ORT check fails (formal handoff mode)")
    ap.add_argument("--gfa-mode", choices=["original", "group_softmax", "memory_softmax"],
                    default=None,
                    help="override GFA only for a deployment candidate; defaults to checkpoint config")
    args = ap.parse_args()

    torch.manual_seed(0)
    # Export does not need the training dataset stack pulled in by tools/_common.
    # Keep this entry point usable in the isolated Hailo suite container.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student, act, cfg = load_student(args.student_ckpt, device=device, gfa_mode=args.gfa_mode)
    gfa_mode = cfg.model.backbone.args.get("gfa_mode", "original")
    if args.gfa_mode is not None and gfa_mode != args.gfa_mode:
        raise SystemExit(f"gfa_mode mismatch after override: cfg={gfa_mode!r} requested={args.gfa_mode!r}")
    _, _, H, W = args.shape
    specs = build_specs(student, H, W, cfg)
    if args.only:
        wanted = set(args.only)
        specs = [s for s in specs if s.name in wanted]
        missing = wanted - {s.name for s in specs}
        if missing:
            raise SystemExit(f"unknown module(s): {sorted(missing)}")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"== fused mid per-module ONNX (opset {args.opset}, act={act}) ==")
    print(f"   ckpt: {args.student_ckpt}")
    print(f"   gfa_mode: {gfa_mode}")
    print(f"   net input: {tuple(args.shape)}")
    print(f"   out-dir: {os.path.abspath(args.out_dir)}\n")

    results: list[ExportResult] = []
    failures: list[str] = []
    for spec in specs:
        try:
            r = export_one(spec, args.out_dir, args.opset, check_ort=not args.no_ort)
            results.append(r)
            flag = "ok "
            if r.max_abs is not None:
                flag = "ok " if r.ort_ok else "CHK"
            if r.ort_ok is False:
                failures.append(f"{r.name}: ORT max_abs={r.max_abs}")
            gdn_flag = " GDN!" if r.gdn_ops else ""
            print(f"[{flag}] {r.name:22s} nodes={r.nodes:4d} {r.size_kb:8.1f}KB "
                  f"sim={'Y' if r.simplified else 'n'} risk={r.hailo_risk:6s}{gdn_flag}")
            if r.max_abs is not None:
                print(f"       max|torch-ort|={r.max_abs:.2e}  in={r.in_shapes} out={r.out_shapes}")
                print(f"       sha256={r.sha256}")
            if r.gdn_ops:
                print(f"       gdn_ops={r.gdn_ops}")
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"[ERR] {spec.name:22s} {err}")
            failures.append(f"{spec.name}: {err}")
            results.append(ExportResult(
                name=spec.name, path="", nodes=0, size_kb=0.0, simplified=False,
                max_abs=None, ops=[], input_names=list(spec.input_names),
                output_names=list(spec.output_names or []),
                hailo_risk=spec.hailo_risk, risk_note=spec.risk_note, error=err,
            ))
            if args.strict:
                break

    expected_names = [s.name for s in specs]
    produced = {r.name for r in results if not r.error and r.path}
    if args.strict and (failures or produced != set(expected_names)):
        err_path = os.path.join(args.out_dir, "export_failures.json")
        with open(err_path, "w") as f:
            json.dump({"failures": failures, "produced": sorted(produced)}, f, indent=2)
        raise SystemExit(
            f"strict export failed ({len(failures)} issue(s)); see {err_path}: {failures}"
        )

    manifest = {
        "ckpt": os.path.abspath(args.student_ckpt),
        "ckpt_sha256": _sha256_file(args.student_ckpt) if os.path.isfile(args.student_ckpt) else None,
        "act": act,
        "split": cfg.student.split,
        "bottleneck": cfg.bottleneck.name,
        "gfa_mode": gfa_mode,
        "net_input_nchw": list(args.shape),
        "opset": args.opset,
        "strict": bool(args.strict),
        "note": (
            "Formal per-module static ONNX handoff for Hailo PTQ; "
            "GDN ops remain high-risk for parse"
        ),
        "modules": [],
    }
    for r in results:
        if r.error:
            continue
        manifest["modules"].append({
            "name": r.name,
            "onnx": r.path,
            "sha256": r.sha256,
            "nodes": r.nodes,
            "size_kb": round(r.size_kb, 1),
            "opset": r.opset,
            "input_names": r.input_names,
            "output_names": r.output_names,
            "in_shapes_nchw": r.in_shapes,
            "out_shapes_nchw": r.out_shapes,
            "ops": r.ops,
            "risky_ops": r.risky_ops,
            "gdn_ops": r.gdn_ops,
            "hailo_risk": r.hailo_risk,
            "risk_note": r.risk_note,
            "ort_max_abs": r.max_abs,
            "ort_ok": r.ort_ok,
            "simplified": r.simplified,
        })

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n== manifest -> {os.path.abspath(manifest_path)} ==")
    print("\n== op coverage (GDN/attention flagged) ==")
    for r in results:
        if r.error:
            continue
        tags = []
        if r.gdn_ops:
            tags.append("GDN/IGDN")
        if any(k in ATTN_OPS for k in r.ops):
            tags.append("ATTN")
        if "Resize" in r.ops:
            tags.append("Resize")
        tag = f" [{', '.join(tags)}]" if tags else ""
        print(f"  {r.name:22s} {', '.join(r.ops)}{tag}")

    print("\n== NEXT (Hailo suite container) ==")
    print("  See deploy/hailo/README_fused_modules.md")
    print("  python deploy/hailo/parse_modules_fused.py --onnx-dir <shared>/onnx/modules")


if __name__ == "__main__":
    main()
