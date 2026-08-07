#!/usr/bin/env python
"""Export the fused mid-split SENDER encode subgraph to static ONNX (B-arm §9.2).

Subgraph = RTFormer sender (stem..stage3, pure conv) + fused bottleneck encode
(align_h / g_a / h_a). Receiver tail and entropy coding are out of scope — the
ONNX outputs are the main latent y and hyper latent z tensors fed to CPU rANS.

  python tools/export_fused_sender.py --help
  python tools/export_fused_sender.py \
    --student-ckpt /data/sidney/checkpoints/semantic-sensing/students/midf_b0.01_s2.pth
  # convbn weights (after B-arm training):
  python tools/export_fused_sender.py \
    --student-ckpt .../students/midf_b0.01_convbn_s2.pth

Pipeline (host, no Hailo SDK): PyTorch -> ONNX opset 13 -> onnxsim -> optional ORT check.
Hand the ONNX to ClientRunner.translate_onnx_model inside the Hailo suite container.
"""
import argparse
import os

from _common import pick_device  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import _compat  # noqa: F401
from src.models import build_model
from src.models.student import build_student

_DEFAULT_CKPT = "/data/sidney/checkpoints/semantic-sensing/students/midf_b0.01_s2.pth"
_DEFAULT_OUT = "/data/sidney/checkpoints/semantic-sensing/hailo/onnx/fused_sender.onnx"


class FusedSenderEncode(nn.Module):
    """mid sender + fused bottleneck encode path (no receiver, no entropy)."""
    def __init__(self, backbone, bottleneck):
        super().__init__()
        self.backbone = backbone
        self.align_h = bottleneck.align_h
        self.g_a = bottleneck.g_a
        self.h_a = bottleneck.h_a

    def forward(self, image):
        x_h, x_l = self.backbone.forward_sender_mid(image)
        a = self.align_h(x_h)
        if a.shape[-2:] != x_l.shape[-2:]:
            a = F.interpolate(a, size=x_l.shape[-2:], mode="bilinear", align_corners=False)
        y = self.g_a(torch.cat([a, x_l], dim=1))
        z = self.h_a(y)
        return y, z


def load_fused_sender(ckpt_path, device="cpu"):
    sd = torch.load(ckpt_path, map_location="cpu")
    cfg = sd["cfg"]
    cfg.model.backbone.args.pretrained = None
    teacher = build_model(cfg.model).eval()
    student = build_student(cfg, teacher).eval()
    student.load_state_dict(sd["student"])
    assert cfg.student.split == "mid", f"expected mid split, got {cfg.student.split!r}"
    assert cfg.bottleneck.name == "rtformer_fused", \
        f"expected rtformer_fused bottleneck, got {cfg.bottleneck.name!r}"
    mod = FusedSenderEncode(student.teacher.backbone, student.bottleneck).eval().to(device)
    act = cfg.bottleneck.args.get("act", "convbn")
    return mod, act, cfg


def export_onnx(module, shape, out_path, opset, check_ort):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    dummy = torch.randn(*shape)
    input_names = ["image"]
    output_names = ["latent_y", "latent_z"]

    with torch.no_grad():
        ref = module(dummy)
    ref = ref if isinstance(ref, tuple) else (ref,)

    torch.onnx.export(
        module, dummy, out_path,
        input_names=input_names, output_names=output_names,
        opset_version=opset, do_constant_folding=True,
        dynamic_axes=None,
    )

    import onnx
    m = onnx.load(out_path)
    onnx.checker.check_model(m)

    simplified = False
    try:
        from onnxsim import simplify
        sm, ok = simplify(m)
        if ok:
            onnx.save(sm, out_path)
            m = sm
            simplified = True
    except ImportError:
        pass

    max_abs = None
    if check_ort:
        import onnxruntime as ort
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        feed = {input_names[0]: dummy.numpy()}
        ort_out = sess.run(output_names, feed)
        max_abs = max(float(np.abs(r.numpy() - o).max()) for r, o in zip(ref, ort_out))

    n_nodes = len(m.graph.node)
    size_kb = os.path.getsize(out_path) / 1024
    op_types = sorted({n.op_type for n in m.graph.node})
    return dict(path=out_path, nodes=n_nodes, size_kb=size_kb, simplified=simplified,
                max_abs=max_abs, ops=op_types, in_shape=tuple(shape),
                out_shapes=[tuple(t.shape) for t in ref])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student-ckpt", default=_DEFAULT_CKPT,
                    help="fused student .pth (gdn ok for structure smoke; convbn for deploy)")
    ap.add_argument("--shape", type=int, nargs=4, default=[1, 3, 512, 512],
                    help="static input N C H W (default 512x512)")
    ap.add_argument("--out", default=_DEFAULT_OUT)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--no-ort", action="store_true", help="skip onnxruntime numeric check")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = pick_device()
    module, act, cfg = load_fused_sender(args.student_ckpt, device=device)
    print(f"== fused sender ONNX export (opset {args.opset}, act={act}) ==")
    print(f"   ckpt: {args.student_ckpt}")
    print(f"   input: {tuple(args.shape)}")

    r = export_onnx(module.cpu(), tuple(args.shape), args.out, args.opset,
                    check_ort=not args.no_ort)
    flag = "ok "
    if r["max_abs"] is not None:
        flag = "ok " if r["max_abs"] < 1e-3 else "CHK"
        ort_line = f"max|torch-ort|={r['max_abs']:.2e}"
    else:
        ort_line = "ort=skipped"
    print(f"[{flag}] nodes={r['nodes']:4d} {r['size_kb']:8.1f}KB "
          f"sim={'Y' if r['simplified'] else 'n'} {ort_line}")
    print(f"   out={r['out_shapes']}  ops={', '.join(r['ops'])}")
    print(f"   -> {os.path.abspath(r['path'])}")
    print("\n== NEXT (Hailo suite container) ==")
    print("  runner = ClientRunner(hw_arch='hailo8')")
    print(f"  runner.translate_onnx_model('{r['path']}', 'fused_sender')")
    print("  runner.load_model_script(open('deploy/hailo/fused_sender.alls').read())")
    print("  runner.optimize(calib_nhwc)  # tools/prepare_hailo_calib.py")
    print("  runner.compile()")


if __name__ == "__main__":
    main()
