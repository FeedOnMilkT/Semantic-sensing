#!/usr/bin/env python3
"""Demo 3 · Hailo DFC parse de-risk — RUN INSIDE the Hailo suite container.

Checkpoint #1 (deplyment-demo/CLAUDE.md): do RTFormer's key blocks — especially
the GFA grouped double-normalization (guide §7 唯一未明确祝福的算子) — pass DFC
parse on hw_arch="hailo8"?

This ONLY does阶段1 Translation (ONNX -> HAR). No optimize/compile yet — isolate
the parse question first (落地顺序 step 2-3). Modules are tried in risk order; one
failure does not stop the others, so you get a full pass/fail matrix in one run.

  # inside container: /local/shared_with_docker
  python onnx/../parse_modules.py            # if you copied it next to onnx/
  python parse_modules.py --onnx-dir onnx --out-dir har

Each module that parses -> <name>_hailo_model.har saved for the next stage.
"""
import argparse
import os
import traceback

# Risk order (guide §7 + demo3_prompt 落地顺序). Standalone static-shape ONNX from
# backbone-demo/tools/export_modules.py — single input, no dynamic axes, so
# translate_onnx_model should auto-infer start/end nodes.
MODULES = [
    ("conv_sanity",    "纯卷积 sanity，隔离环境，零算子风险"),
    ("gfa",            "#1 风险 GFA double-norm (softmax_N + grouped L1)"),
    ("cross_res_attn", "cross-res 注意力，裸 matmul→MHA 映射 (要盯)"),
    ("rtformer_stage", "一个完整 stepped block (GFA+CA+2xFFN)"),
    ("full_backbone",  "整网 (DAPPM/上采样/动态轴残留也一起验)"),
]


def parse_one(runner_cls, name, onnx_dir, out_dir):
    onnx_path = os.path.join(onnx_dir, f"{name}.onnx")
    if not os.path.isfile(onnx_path):
        return name, "SKIP", f"missing {onnx_path}", None
    runner = runner_cls(hw_arch="hailo8")          # Hailo-8 -> DFC 3.x (guide §1)
    try:
        # guide §4: for clean static ONNX, no start/end/net_input_shapes needed.
        runner.translate_onnx_model(onnx_path, name)
    except Exception:
        # fallback: some versions want the model name positional only / retry bare
        runner.translate_onnx_model(onnx_path)
    har = os.path.join(out_dir, f"{name}_hailo_model.har")
    runner.save_har(har)
    # report the parsed HN op/layer types so you can eyeball against Supported Layers
    try:
        layers = runner.get_hn_dict().get("layers", {})
        op_types = sorted({v.get("type", "?") for v in layers.values()})
    except Exception:
        op_types = []
    return name, "PASS", har, op_types


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default="onnx")
    ap.add_argument("--out-dir", default="har")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    from hailo_sdk_client import ClientRunner   # provided by the suite image

    os.makedirs(args.out_dir, exist_ok=True)
    names = args.only or [m for m, _ in MODULES]
    desc = dict(MODULES)

    print(f"== Hailo DFC parse (hw_arch=hailo8) · onnx={args.onnx_dir} -> har={args.out_dir} ==\n")
    results = []
    for name in names:
        print(f"--- {name}: {desc.get(name,'')}")
        try:
            r = parse_one(ClientRunner, name, args.onnx_dir, args.out_dir)
        except Exception as e:
            print(traceback.format_exc())
            r = (name, "FAIL", f"{type(e).__name__}: {e}", None)
        results.append(r)
        nm, status, info, ops = r
        print(f"    [{status}] {info}")
        if ops:
            print(f"    HN ops: {', '.join(ops)}")
        print()

    print("== SUMMARY ==")
    for nm, status, info, _ in results:
        mark = {"PASS": "✅", "FAIL": "❌", "SKIP": "⚠️"}.get(status, "?")
        print(f"  {mark} {nm:14s} {status}")
    print("\nNext per passing module: load_model_script(.alls) -> optimize(calib) ->")
    print("QUANTIZED emulator 验掉点 -> compile(). FAILs -> hailo visualizer <har> --no-browser.")


if __name__ == "__main__":
    main()
