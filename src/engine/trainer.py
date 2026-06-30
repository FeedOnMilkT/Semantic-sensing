"""Hand-written training loop (pure PyTorch, no Lightning).

Conditional multi-task: each step forwards only the batch's active tasks and the
MultiTaskLoss computes loss only where GT exists. Adding/removing a task in the
YAML `heads:` therefore needs zero trainer changes.

Multi-GPU (optional, gated): if launched under torchrun (WORLD_SIZE>1), the
model+loss are wrapped in DDP so every rank's gradients sync — including the
uncertainty-weighting log_vars, which live in the loss module (that's why model
AND loss go inside one wrapped module). Single-GPU path is untouched: no DDP, no
process group. Rank 0 owns the progress bar, logging, checkpointing and eval.
"""
import math
import os
import sys
import time

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from tqdm import tqdm
except Exception:  # tqdm optional
    tqdm = None

from ..utils.logging import get_logger

log = get_logger("trainer")


class _ModelLoss(nn.Module):
    """Bundle model + loss so DDP wraps (and syncs) both sets of parameters."""
    def __init__(self, model, loss):
        super().__init__()
        self.model = model
        self.loss = loss

    def forward(self, x, targets, tasks):
        preds = self.model(x, tasks=tasks)
        return self.loss(preds, targets, tasks)   # (total_loss, parts)


def build_optimizer(cfg, params):
    name = cfg.get("name", "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.get("lr", 6e-5), weight_decay=cfg.get("weight_decay", 0.01))
    if name == "sgd":
        return torch.optim.SGD(params, lr=cfg.get("lr", 0.01), momentum=cfg.get("momentum", 0.9),
                               weight_decay=cfg.get("weight_decay", 5e-4))
    raise KeyError(f"unknown optimizer: {name}")


def poly_lr(step, total, base_lr, power=0.9, warmup=0):
    if warmup and step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * (1 - t) ** power


def _fmt_eta(seconds):
    seconds = int(max(0, seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


class Trainer:
    def __init__(self, cfg, model, loss, device="cuda", rank=0, world_size=1, local_rank=0):
        self.cfg = cfg
        self.device = device
        self.rank, self.world_size, self.is_main = rank, world_size, rank == 0
        self.distributed = world_size > 1
        # keep unwrapped refs for eval / checkpoint
        self.model = model.to(device)
        self.loss = loss.to(device)
        # opt-in SyncBN (train.sync_bn): for small per-GPU batch (paper-faithful
        # eff-batch via more cards), sync BN stats across ranks so BN sees the full
        # effective batch instead of the tiny per-GPU one. No-op when not distributed.
        if self.distributed and cfg.get("sync_bn", False):
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        core = _ModelLoss(self.model, self.loss)
        if self.distributed:
            self.core = DDP(core, device_ids=[local_rank], output_device=local_rank,
                            find_unused_parameters=True)
        else:
            self.core = core
        self.opt = build_optimizer(cfg.get("optimizer", {}), self.core.parameters())
        self.base_lr = self.opt.param_groups[0]["lr"]
        self.amp = bool(cfg.get("amp", False))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.sched = cfg.get("scheduler", "poly")
        self.grad_clip = cfg.get("grad_clip", 0.0)

    def _set_lr(self, step, total):
        if self.sched == "poly":
            lr = poly_lr(step, total, self.base_lr, warmup=self.cfg.get("warmup", 0))
        elif self.sched == "cosine":
            lr = 0.5 * self.base_lr * (1 + math.cos(math.pi * step / max(1, total)))
        else:
            return self.base_lr
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr

    def save_ckpt(self, path, epoch, global_step):
        """Resume-complete snapshot: model + loss (uncertainty log_vars) + optimizer
        + AMP scaler + position. Atomic (tmp -> replace) so a crash mid-write can't
        corrupt the file."""
        ckpt = {"model": self.model.state_dict(), "loss": self.loss.state_dict(),
                "opt": self.opt.state_dict(), "scaler": self.scaler.state_dict(),
                "epoch": epoch, "global_step": global_step}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        torch.save(ckpt, tmp)
        os.replace(tmp, path)

    def load_ckpt(self, path):
        """Restore all state in-place; return (next_epoch, global_step) to resume from."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.loss.load_state_dict(ckpt["loss"])
        self.opt.load_state_dict(ckpt["opt"])
        self.scaler.load_state_dict(ckpt["scaler"])
        return int(ckpt.get("epoch", 0)), int(ckpt.get("global_step", 0))

    def fit(self, loader, epochs=1, max_steps=None, log_every=10, sampler=None,
            start_epoch=0, ckpt_path=None, save_every=1):
        spe = len(loader)                          # steps per epoch (per rank)
        total = (max_steps if max_steps else epochs * spe)
        step = start_epoch * spe                   # global step continues across resume
        run_steps = 0                              # steps THIS run (for honest ETA on resume)
        t0 = time.time()
        self.core.train()
        use_bar = tqdm is not None and self.is_main and sys.stderr.isatty()
        pbar = tqdm(total=total, initial=step, dynamic_ncols=True, desc="train") if use_bar else None
        for ep in range(start_epoch, epochs):
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(ep)                   # reshuffle shards per epoch (DDP); deterministic -> resume-safe
            for batch in loader:
                if max_steps and step >= max_steps:
                    break
                lr = self._set_lr(step, total)
                x = batch["image"].to(self.device, non_blocking=True)
                tasks = batch["tasks"]
                targets = {t: batch["targets"][t].to(self.device, non_blocking=True) for t in tasks}

                self.opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.amp):
                    total_loss, parts = self.core(x, targets, tasks)
                self.scaler.scale(total_loss).backward()
                if self.grad_clip:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.core.parameters(), self.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()

                if self.is_main:
                    run_steps += 1
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(loss=f"{float(total_loss):.3f}", lr=f"{lr:.1e}")
                    if step % log_every == 0:
                        eta = (time.time() - t0) / max(1, run_steps) * (total - step - 1)
                        parts_str = " ".join(f"{k}={v:.3f}" for k, v in parts.items())
                        pct = 100.0 * (step + 1) / total
                        log.info(f"ep{ep} [{step + 1}/{total} {pct:4.1f}%] lr={lr:.2e} "
                                 f"loss={float(total_loss):.3f} [{parts_str}] eta={_fmt_eta(eta)}")
                step += 1
            # end of epoch: checkpoint for resume (rank 0, real runs only)
            if self.is_main and ckpt_path and not max_steps and (ep + 1) % save_every == 0:
                self.save_ckpt(ckpt_path, epoch=ep + 1, global_step=step)
                log.info(f"[ckpt] saved last @ epoch {ep + 1} -> {ckpt_path}")
            if max_steps and step >= max_steps:
                break
        if pbar is not None:
            pbar.close()
        return self.model
