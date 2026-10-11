#!/usr/bin/env python
"""Per-module ONNX export + sanity — host-side front of the Hailo DFC chain.

Demo 3 de-risk, modular. This runs the FIRST HALF of the deployment pipeline that
does NOT need the board or the Hailo suite container:

    PyTorch module --> static ONNX (opset 13) --> onnxsim --> onnxruntime numeric check

It isolates each RTFormer risk operator (deplyment-demo/hailo_deployment_guide.md
§7) into its own tiny graph, in the risk order of demo3_prompt.md "落地顺序":

    1. conv_sanity     stem+stage1~3 (pure conv)      環境/导出链路 sanity, zero op-risk
    2. gfa             GFA double-normalization        #1 risk — softmax(N) + grouped L1
    3. cross_res_attn  Cross-resolution Attention      standard softmax MHA (Hailo 一等支持)
    4. rtformer_stage  one stepped RTFormer block       GFA + CA + 2x FFN together
    5. full_backbone   whole RTFormer                   整网导出, feeds ClientRunner

Each module's clean static ONNX is exactly what you hand to `ClientRunner.
translate_onnx_model(...)` inside the Hailo suite container (the actual DFC
parse/optimize/compile is the NEXT step — see the printed handoff note).

  python tools/export_modules.py                       # all modules, 512x1024
  python tools/export_modules.py --only gfa cross_res_attn
  python tools/export_modules.py --shape 1 3 512 1024 --out-dir deploy/onnx
"""
import argparse
import os

from _common import pick_device  # noqa: F401  (adds repo root to sys.path)
import numpy as np
import torch

from src.models.backbones.rtformer import (
    RTFormer, EABlock, ExternalAttention,
)

# RTFormer-Base widths (configs/cityscapes_mt_rtformer.yaml: args={}).
C = dict(c1=64, c2=128, c3=256, c4=512, mem=128, groups=8, cross_size=12)


class ConvTrunk(torch.nn.Module):
    """Official stem through stage 3, the pure-convolution sender subgraph."""
    def __init__(self, rt: RTFormer):
        super().__init__()
        self.rt = rt

    def forward(self, x):
        return self.rt.forward_sender_mid(x)


def build_specs(H, W):
    """Each spec: name -> (module, example_inputs tuple, input_names).
    Spatial sizes derived from a HxW network input at the module's stride."""
    h8, w8 = H // 8, W // 8        # high branch (1/8)
    h32, w32 = H // 32, W // 32    # low branch (1/32, RTFormer stage4/5)

    rt = RTFormer().eval()         # one shared instance (faithful widths)

    def t(*shape):
        return torch.randn(*shape)

    return {
        # 1. pure-conv sanity: isolate the export/onnxsim/ORT environment, zero op-risk
        "conv_sanity": (
            ConvTrunk(rt).eval(),
            (t(1, 3, H, W),),
            ["image"],
        ),
        # 2. #1 RISK: GFA grouped double-normalization (low branch @1/32, c4)
        "gfa": (
            ExternalAttention(C["c4"], C["c4"], C["mem"],
                              num_heads=C["groups"]).eval(),
            (t(1, C["c4"], h32, w32),),
            ["low_feat"],
        ),
        # 3. official cross attention uses pooled low-branch K/V and grouped conv
        "cross_res_attn": (
            EABlock((C["c2"], C["c4"]), (C["c2"], C["c4"]),
                    num_heads=C["groups"], cross_size=C["cross_size"],
                    use_injection=False).eval(),
            (t(1, C["c2"], h8, w8), t(1, C["c4"], h32, w32)),
            ["high_feat", "low_feat"],
        ),
        # 4. stage-4 transition includes official low stride-2 projection and injection.
        "rtformer_stage": (
            EABlock((C["c2"], C["c3"]), (C["c2"], C["c4"]),
                    num_heads=C["groups"], cross_size=C["cross_size"],
                    use_injection=True).eval(),
            (t(1, C["c2"], h8, w8), t(1, C["c3"], H // 16, W // 16)),
            ["high_feat", "low_feat"],
        ),
        # 5. whole backbone (dict output -> tuple-wrapped for a clean graph)
        "full_backbone": (
            _BackboneTuple(rt).eval(),
            (t(1, 3, H, W),),
            ["image"],
        ),
    }


class _BackboneTuple(torch.nn.Module):
    """RTFormer returns a dict; wrap to a fixed-order tuple for ONNX."""
    KEYS = ("s4", "s8", "s16", "s32")

    def __init__(self, rt):
        super().__init__()
        self.rt = rt

    def forward(self, x):
        out = self.rt(x)
        return tuple(out[k] for k in self.KEYS)


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
            simplified = True
    except ImportError:
        pass

    # numeric parity: onnxruntime vs torch (the NATIVE-emulator equivalent here)
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feed = {n: x.numpy() for n, x in zip(input_names, inputs)}
    ort_out = sess.run(out_names, feed)
    max_abs = max(float(np.abs(r.numpy() - o).max()) for r, o in zip(ref, ort_out))

    n_nodes = len(m.graph.node)
    size_kb = os.path.getsize(path) / 1024
    op_types = sorted({n.op_type for n in m.graph.node})
    return dict(name=name, path=path, nodes=n_nodes, size_kb=size_kb,
                simplified=simplified, max_abs=max_abs, ops=op_types,
                in_shapes=[tuple(x.shape) for x in inputs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=int, nargs=4, default=[1, 3, 512, 1024],
                    help="network input N C H W (default paper crop 512x1024)")
    ap.add_argument("--out-dir", default="deploy/onnx")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--only", nargs="*", default=None, help="subset of module names")
    args = ap.parse_args()

    torch.manual_seed(0)
    os.makedirs(args.out_dir, exist_ok=True)
    _, _, H, W = args.shape
    specs = build_specs(H, W)
    names = args.only or list(specs)

    print(f"== per-module ONNX export (opset {args.opset}, net input {tuple(args.shape)}) ==")
    print(f"   out-dir: {os.path.abspath(args.out_dir)}\n")
    rows = []
    for name in names:
        module, inputs, input_names = specs[name]
        try:
            r = export_one(name, module, inputs, input_names, args.out_dir, args.opset)
            rows.append(r)
            flag = "ok " if r["max_abs"] < 1e-3 else "CHK"
            print(f"[{flag}] {name:14s} nodes={r['nodes']:4d} {r['size_kb']:8.1f}KB "
                  f"sim={'Y' if r['simplified'] else 'n'} "
                  f"max|torch-ort|={r['max_abs']:.2e}  in={r['in_shapes']}")
        except Exception as e:
            print(f"[ERR] {name:14s} {type(e).__name__}: {e}")

    print("\n== op coverage (check against guide §7 Supported Layers) ==")
    for r in rows:
        print(f"  {r['name']:14s} {', '.join(r['ops'])}")

    print("\n== NEXT (Hailo suite container, not this env) ==")
    print("  hailo_sdk_client is NOT installed here — these ONNX are DFC-ready inputs.")
    print("  Inside the suite container, per module:")
    print("    runner = ClientRunner(hw_arch='hailo8')")
    print("    runner.translate_onnx_model('<module>.onnx', '<module>')  # parse")
    print("    runner.save_har('<module>_hailo_model.har')")
    print("    # then optimize(calib) -> QUANTIZED check -> compile() -> HEF")


if __name__ == "__main__":
    main()
