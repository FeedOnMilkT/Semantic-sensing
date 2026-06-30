#!/usr/bin/env python
"""Per-module ONNX export + sanity for MiT-B0 (SegFormer encoder) — host-side
front of the Hailo DFC chain, mirroring tools/export_modules.py (which is RTFormer).

Why this exists: we picked MiT-B0/SegFormer as the depth-quality mainline, but
MiT is a TRANSFORMER — its risk operators (LayerNorm, dynamic activation@activation
MatMul, Softmax, 5-D permute, token<->spatial reshapes, GELU/Erf) are exactly the
ones an edge NPU like Hailo-8 struggles with. RTFormer was chosen originally
BECAUSE it avoids these. So before committing to MiT for deployment we must check
whether its modules even translate.

This runs the FIRST HALF (no board / no Hailo suite container):
    PyTorch module --> static ONNX (opset 13) --> onnxsim --> onnxruntime numeric check
and reports the op set per module so it can be checked against the Hailo §7
Supported-Layers table. The actual ClientRunner.translate_onnx_model / optimize /
compile is the NEXT step, inside the suite container.

Risk order (SegFormer MiT-B0):
  1. patch_embed   OverlapPatchEmbed: conv + flatten/transpose + LayerNorm   (tests LN export)
  2. attn_sr       EfficientAttention WITH spatial-reduction (stage1, 2 heads) #1 risk: SR-conv,
                   LayerNorm, q@k^T & attn@v dynamic MatMul, Softmax, 5-D permute
  3. attn_full     EfficientAttention sr_ratio=1 (stage4, 8 heads)            full attention, no-SR branch
  4. mix_ffn       MixFFN: dwconv-in-sequence reshapes + GELU
  5. block         one full transformer Block (2x LayerNorm + attn + ffn + residual)
  6. full_backbone whole MiT-B0 (4 stages)

  python tools/export_modules_mit.py
  python tools/export_modules_mit.py --only attn_sr block
  python tools/export_modules_mit.py --shape 1 3 512 1024 --out-dir deploy/onnx_mit
"""
import argparse
import os

from _common import pick_device  # noqa: F401  (adds repo root to sys.path)
import numpy as np
import torch

from src.models.backbones.mit_b0 import (
    MiTB0, OverlapPatchEmbed, EfficientAttention, MixFFN, Block,
)

# MiT-B0 widths (configs/base.yaml mit_b0 defaults).
EMBED = (32, 64, 160, 256)
HEADS = (1, 2, 5, 8)
SR = (8, 4, 2, 1)
MLP = (4, 4, 4, 4)

# Hailo-8 risk op_types (guide §7): these don't map 1:1 to an edge NPU.
RISK_OPS = {"MatMul", "Softmax", "Erf", "ReduceMean", "ReduceL2",
            "Einsum", "LayerNormalization", "Gelu", "InstanceNormalization"}


class _SeqWrap(torch.nn.Module):
    """Wrap a module whose forward is f(x, H, W) into f(x) with H,W frozen, and
    return only tensor outputs (drop the int H,W that patch_embed emits)."""
    def __init__(self, mod, H, W, tensor_only_first=False):
        super().__init__()
        self.mod, self.H, self.W = mod, H, W
        self.tensor_only_first = tensor_only_first

    def forward(self, x):
        out = self.mod(x, self.H, self.W) if not self.tensor_only_first else self.mod(x)
        if isinstance(out, tuple):
            return out[0]            # patch_embed -> (tensor, H, W); keep tensor
        return out


class _BackboneTuple(torch.nn.Module):
    KEYS = ("s4", "s8", "s16", "s32")

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)
        return tuple(out[k] for k in self.KEYS)


def build_specs(H, W):
    """name -> (module, example_inputs tuple, input_names). Token counts derive
    from a HxW net input at each stage's stride."""
    def hw(stride):
        return H // stride, W // stride

    def t(*shape):
        return torch.randn(*shape)

    # stage strides: s4 / s8 / s16 / s32
    h1, w1 = hw(8)       # stage1 (s8): SR attn representative (2 heads)
    h3, w3 = hw(32)      # stage4 (s32): no-SR full attention (8 heads)
    h0, w0 = hw(4)       # stage0 (s4): patch_embed output res

    # stage1 SR attention: dim=64, heads=2, sr=4
    attn_sr = EfficientAttention(EMBED[1], HEADS[1], SR[1]).eval()
    # stage4 full attention: dim=256, heads=8, sr=1 (no sr branch)
    attn_full = EfficientAttention(EMBED[3], HEADS[3], SR[3]).eval()
    mix = MixFFN(EMBED[1], EMBED[1] * MLP[1]).eval()
    blk = Block(EMBED[1], HEADS[1], MLP[1], SR[1]).eval()

    return {
        # 1. patch embed (stage0): conv + flatten/transpose + LayerNorm
        "patch_embed": (
            _SeqWrap(OverlapPatchEmbed(7, 4, 3, EMBED[0]).eval(), None, None, tensor_only_first=True),
            (t(1, 3, H, W),), ["image"],
        ),
        # 2. #1 RISK: efficient attention WITH spatial reduction (stage1)
        "attn_sr": (
            _SeqWrap(attn_sr, h1, w1),
            (t(1, h1 * w1, EMBED[1]),), ["tokens"],
        ),
        # 3. efficient attention, sr_ratio=1 -> full self-attention (stage4)
        "attn_full": (
            _SeqWrap(attn_full, h3, w3),
            (t(1, h3 * w3, EMBED[3]),), ["tokens"],
        ),
        # 4. Mix-FFN: dwconv-in-sequence + GELU
        "mix_ffn": (
            _SeqWrap(mix, h1, w1),
            (t(1, h1 * w1, EMBED[1]),), ["tokens"],
        ),
        # 5. one full transformer block (2x LayerNorm + attn + ffn + residual)
        "block": (
            _SeqWrap(blk, h1, w1),
            (t(1, h1 * w1, EMBED[1]),), ["tokens"],
        ),
        # 6. whole MiT-B0 backbone (dict -> tuple)
        "full_backbone": (
            _BackboneTuple(MiTB0().eval()),
            (t(1, 3, H, W),), ["image"],
        ),
    }


def export_one(name, module, inputs, input_names, out_dir, opset):
    path = os.path.join(out_dir, f"{name}.onnx")
    with torch.no_grad():
        ref = module(*inputs)
    ref = ref if isinstance(ref, tuple) else (ref,)
    out_names = [f"out{i}" for i in range(len(ref))]

    torch.onnx.export(
        module, inputs, path,
        input_names=input_names, output_names=out_names,
        opset_version=opset, do_constant_folding=True,
        dynamic_axes=None,  # static shape — required by Hailo DFC
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

    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feed = {n: x.numpy() for n, x in zip(input_names, inputs)}
    ort_out = sess.run(out_names, feed)
    max_abs = max(float(np.abs(r.numpy() - o).max()) for r, o in zip(ref, ort_out))

    op_types = sorted({n.op_type for n in m.graph.node})
    return dict(name=name, path=path, nodes=len(m.graph.node),
                size_kb=os.path.getsize(path) / 1024, simplified=simplified,
                max_abs=max_abs, ops=op_types,
                in_shapes=[tuple(x.shape) for x in inputs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=int, nargs=4, default=[1, 3, 512, 1024])
    ap.add_argument("--out-dir", default="deploy/onnx_mit")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    os.makedirs(args.out_dir, exist_ok=True)
    _, _, H, W = args.shape
    specs = build_specs(H, W)
    names = args.only or list(specs)

    print(f"== MiT-B0 per-module ONNX export (opset {args.opset}, net input {tuple(args.shape)}) ==")
    print(f"   out-dir: {os.path.abspath(args.out_dir)}\n")
    rows = []
    for name in names:
        module, inputs, input_names = specs[name]
        try:
            r = export_one(name, module, inputs, input_names, args.out_dir, args.opset)
            rows.append(r)
            flag = "ok " if r["max_abs"] < 1e-3 else "CHK"
            print(f"[{flag}] {name:14s} nodes={r['nodes']:4d} {r['size_kb']:9.1f}KB "
                  f"sim={'Y' if r['simplified'] else 'n'} "
                  f"max|torch-ort|={r['max_abs']:.2e}  in={r['in_shapes']}")
        except Exception as e:
            print(f"[ERR] {name:14s} {type(e).__name__}: {e}")

    print("\n== op coverage (★ = Hailo-8 risk op, check guide §7 Supported Layers) ==")
    for r in rows:
        marked = [f"{o}★" if o in RISK_OPS else o for o in r["ops"]]
        print(f"  {r['name']:14s} {', '.join(marked)}")

    risky = sorted({o for r in rows for o in r["ops"] if o in RISK_OPS})
    print(f"\n  Hailo-risk ops present across modules: {risky or 'NONE'}")
    print("\n== NEXT (Hailo suite container, not this env) ==")
    print("  hailo_sdk_client is NOT installed here. Inside the suite container, per module:")
    print("    runner = ClientRunner(hw_arch='hailo8')")
    print("    runner.translate_onnx_model('<module>.onnx', '<module>')   # <-- where unsupported ops surface")


if __name__ == "__main__":
    main()
