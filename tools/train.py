#!/usr/bin/env python
"""Train a MultiTaskNet from a YAML config.

Single-GPU:
  python tools/train.py configs/cityscapes_multitask.yaml

Multi-GPU (DDP, e.g. 2 cards, batch_size is PER-GPU):
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501 \
      tools/train.py configs/cityscapes_mt_rtformer.yaml --out checkpoints/rtformer.pth
"""
import argparse
import os

from _common import make_loader, tasks_from_cfg, pick_device
import torch
import torch.distributed as dist

from src.utils import load_config, get_logger
from src.models import build_model
from src.losses import build_loss
from src.engine import Trainer, Evaluator

log = get_logger("train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], help="dotlist overrides, e.g. train.epochs=5")
    ap.add_argument("--out", default="checkpoints/model.pth")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="resume from <out>.last.pth if it exists")
    args = ap.parse_args()
    ckpt_path = os.path.splitext(args.out)[0] + ".last.pth"   # rolling resume checkpoint

    # ---- distributed setup (torchrun sets WORLD_SIZE/RANK/LOCAL_RANK) ----
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = pick_device()
    is_main = rank == 0

    cfg = load_config(args.config, args.set)
    tasks = tasks_from_cfg(cfg)
    if is_main:
        log.info(f"tasks={tasks} backbone={cfg.model.backbone.name} device={device} "
                 f"world_size={world_size} per_gpu_bs={cfg.train.batch_size}")

    model = build_model(cfg.model)
    loss = build_loss(cfg.loss, tasks)
    if is_main:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        log.info(f"model params: {n_params:.2f}M")

    train_loader = make_loader(cfg.data.train, cfg.train.batch_size, shuffle=True,
                               distributed=distributed)
    trainer = Trainer(cfg.train, model, loss, device=device,
                      rank=rank, world_size=world_size, local_rank=local_rank)

    # resume: every rank loads the same checkpoint so DDP weights stay in sync
    start_epoch = 0
    if args.resume and os.path.exists(ckpt_path):
        start_epoch, gstep = trainer.load_ckpt(ckpt_path)
        if is_main:
            log.info(f"resumed from {ckpt_path} -> start_epoch={start_epoch} (global_step={gstep})")
    elif args.resume and is_main:
        log.info(f"--resume set but no checkpoint at {ckpt_path}; starting fresh")

    sampler = getattr(train_loader, "sampler", None) if distributed else None
    trainer.fit(train_loader, epochs=cfg.train.epochs, max_steps=args.max_steps,
                log_every=cfg.train.get("log_every", 50), sampler=sampler,
                start_epoch=start_epoch, ckpt_path=ckpt_path,
                save_every=cfg.train.get("save_every_epochs", 1))

    if is_main:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": cfg}, args.out)
        log.info(f"saved checkpoint -> {args.out}")

    # ---- eval on rank 0 only (full val set, single GPU) ----
    if not args.no_eval and cfg.data.get("val"):
        if distributed:
            dist.barrier()
        if is_main:
            eval_bs = cfg.eval.get("batch_size", cfg.train.batch_size)
            val_loader = make_loader(cfg.data.val, eval_bs, shuffle=False)
            ev = Evaluator(tasks, cfg.eval.get("meters", {}), device=device)
            log.info(f"eval: {ev.evaluate(model, val_loader)}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
