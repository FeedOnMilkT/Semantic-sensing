#!/usr/bin/env python
"""Probe peak VRAM + throughput per per-GPU batch size for RTFormer single-task
seg, so we can pick the batch for the 4-GPU DDP run. Single-GPU, ~20 steps each.
Run: CUDA_VISIBLE_DEVICES=4 python tools/probe_batch.py
"""
import time

from _common import make_loader, tasks_from_cfg
import torch

from src.utils import load_config
from src.models import build_model
from src.losses import build_loss
from src.engine import Trainer

CFG = "configs/cityscapes_seg.yaml"
OVERRIDES = [
    "model.backbone.name=rtformer",
    "model.backbone.args.pretrained=/data/sidney/checkpoints/semantic-sensing/pretrained/rtformer_in1k.pth",
]
DEVICE = "cuda"
STEPS = 20

for bs in [4, 8, 16, 32]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cfg = load_config(CFG, OVERRIDES)
    tasks = tasks_from_cfg(cfg)
    try:
        model = build_model(cfg.model)
        loss = build_loss(cfg.loss, tasks)
        loader = make_loader(cfg.data.train, bs, shuffle=True, num_workers=4)
        tr = Trainer(cfg.train, model, loss, device=DEVICE, rank=0, world_size=1, local_rank=0)
        t0 = time.time()
        tr.fit(loader, epochs=100, max_steps=STEPS, log_every=1000)
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[PROBE] bs={bs:>2}  peak={peak:5.1f} GB  {1000 * dt / STEPS:6.0f} ms/iter  "
              f"(eff_bs_4card={bs * 4}, ~{2975 // (bs * 4)} iters/epoch)", flush=True)
        del model, loss, loader, tr
    except RuntimeError as e:
        print(f"[PROBE] bs={bs:>2}  FAILED: {str(e)[:120]}", flush=True)
    torch.cuda.empty_cache()
