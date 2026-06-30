#!/usr/bin/env python
"""Static-shape ONNX export — groundwork for the Hailo DFC chain (CLAUDE.md
acceptance #4). Wraps the dict-returning model into a fixed-order tuple output so
the graph is clean; static input shape (no dynamic axes), opset 13.

  python tools/export_onnx.py configs/cityscapes_multitask.yaml --shape 1 3 512 1024 --out model.onnx [--simplify] [--ckpt ...]
"""
import argparse

from _common import tasks_from_cfg, pick_device
import torch
import torch.nn as nn

from src.utils import load_config, get_logger
from src.models import build_model

log = get_logger("export")


class TupleWrap(nn.Module):
    """Return head outputs as a fixed-order tuple (ONNX-friendly)."""
    def __init__(self, model, tasks):
        super().__init__()
        self.model = model
        self.tasks = list(tasks)

    def forward(self, x):
        out = self.model(x, tasks=self.tasks)
        return tuple(out[t] for t in self.tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--shape", type=int, nargs=4, default=[1, 3, 512, 1024])
    ap.add_argument("--out", default="model.onnx")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--simplify", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    tasks = tasks_from_cfg(cfg)
    model = build_model(cfg.model).eval()
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])

    wrapped = TupleWrap(model, tasks).eval()
    dummy = torch.randn(*args.shape)
    log.info(f"exporting tasks={tasks} shape={tuple(args.shape)} opset={args.opset}")

    torch.onnx.export(
        wrapped, dummy, args.out,
        input_names=["image"], output_names=[f"out_{t}" for t in tasks],
        opset_version=args.opset, do_constant_folding=True,
        dynamic_axes=None,  # static shape for Hailo DFC
    )

    import onnx
    m = onnx.load(args.out)
    onnx.checker.check_model(m)
    log.info(f"ONNX check OK -> {args.out}")

    if args.simplify:
        try:
            from onnxsim import simplify
            sm, ok = simplify(m)
            assert ok, "onnxsim returned invalid model"
            onnx.save(sm, args.out)
            log.info("onnxsim: simplified")
        except ImportError:
            log.info("onnxsim not installed; skipping (pip install onnxsim)")


if __name__ == "__main__":
    main()
