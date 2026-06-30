#!/usr/bin/env python
"""Evaluate a checkpoint.  python tools/eval.py configs/cityscapes_seg.yaml --ckpt checkpoints/model.pth"""
import argparse

from _common import make_loader, tasks_from_cfg, pick_device
import torch

from src.utils import load_config, get_logger
from src.models import build_model
from src.engine import Evaluator

log = get_logger("eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--slide", action="store_true",
                    help="sliding-window seg eval at native resolution (no aspect "
                         "distortion); forces val size=null + batch 1")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--stride", type=int, default=341)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    device = pick_device()
    tasks = tasks_from_cfg(cfg)

    model = build_model(cfg.model).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    log.info(f"loaded {args.ckpt}; tasks={tasks}")

    slide = None
    if args.slide:
        cfg.data.val.args.size = None           # native resolution -> needs batch 1
        slide = {"crop": args.crop, "stride": args.stride}
        log.info(f"slide eval: crop={args.crop} stride={args.stride} (native res, batch 1)")
    eval_bs = 1 if args.slide else cfg.train.batch_size
    val_loader = make_loader(cfg.data.val, eval_bs, shuffle=False)
    ev = Evaluator(tasks, cfg.eval.get("meters", {}), device=device)
    metrics = ev.evaluate(model, val_loader, slide=slide)
    for t, m in metrics.items():
        log.info(f"[{t}] " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))


if __name__ == "__main__":
    main()
