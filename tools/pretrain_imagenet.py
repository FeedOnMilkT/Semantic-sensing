#!/usr/bin/env python
"""ImageNet-1K pretraining for the RTFormer backbone (so it can be loaded as an
ImageNet-pretrained backbone for segmentation via backbone.args.pretrained).

Trains RTFormer's trunk (stem->stage5, via forward_trunk -> s32) + a linear
classifier. DAPPM is NOT trained here (it's a seg-head module). DDP, AMP, cosine
LR with warmup, label smoothing, per-epoch checkpoint + --resume.

Launch (4 GPUs, batch is PER-GPU):
  HF_HOME=/data/sidney/hf-cache CUDA_VISIBLE_DEVICES=0,1,4,5 \
    torchrun --nproc_per_node=4 --master_port=29701 tools/pretrain_imagenet.py \
    --epochs 100 --batch-size 512 --lr 2e-3 --resume \
    --out /data/sidney/checkpoints/semantic-sensing/pretrained/rtformer_in1k.pth
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as T

from src.models.backbones import RTFormer
from src.utils.logging import get_logger

log = get_logger("pretrain")

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class ClsModel(nn.Module):
    """RTFormer trunk (s32) -> global avg pool -> linear classifier."""
    def __init__(self, num_classes=1000):
        super().__init__()
        self.backbone = RTFormer()                       # Base defaults
        c32 = self.backbone.out_channels["s32"]          # 512
        self.head = nn.Linear(c32, num_classes)

    def forward(self, x):
        f = self.backbone.forward_trunk(x)               # B, 512, H/32, W/32
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)       # B, 512
        return self.head(f)


class HFImageNet(Dataset):
    """Wrap the HF imagenet-1k split as a torch Dataset with torchvision transforms."""
    def __init__(self, split, tf):
        from datasets import load_dataset
        from huggingface_hub import get_token
        # datasets 5.0 doesn't auto-pick the ambient token for this gated repo -> pass it explicitly
        self.ds = load_dataset("ILSVRC/imagenet-1k", split=split, token=get_token())
        self.tf = tf

    def __len__(self):
        return self.ds.num_rows

    def __getitem__(self, i):
        ex = self.ds[i]
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.tf(img), int(ex["label"])


def build_transforms(img_size):
    train = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.4),
        T.ToTensor(), T.Normalize(_MEAN, _STD),
    ])
    val = T.Compose([
        T.Resize(int(img_size * 256 / 224)), T.CenterCrop(img_size),
        T.ToTensor(), T.Normalize(_MEAN, _STD),
    ])
    return train, val


def cosine_lr(step, total, base_lr, warmup, min_lr=1e-5):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def fmt_eta(sec):
    sec = int(max(0, sec)); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item(); total += y.numel()
    t = torch.tensor([correct, total], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(t)
    model.train()
    return (t[0] / t[1]).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=512, help="PER-GPU")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="if >0, also keep a permanent backbone snapshot every N epochs")
    ap.add_argument("--out", default="/data/sidney/checkpoints/semantic-sensing/pretrained/rtformer_in1k.pth")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    ckpt_path = os.path.splitext(args.out)[0] + ".last.pth"

    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed = world > 1
    if distributed:
        dist.init_process_group("nccl"); torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cuda"
    is_main = rank == 0

    tf_train, tf_val = build_transforms(args.img_size)
    if is_main:
        log.info(f"loading imagenet-1k (HF_HOME={os.environ.get('HF_HOME')}) ...")
    train_ds = HFImageNet("train", tf_train)
    val_ds = HFImageNet("validation", tf_val)
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              shuffle=train_sampler is None, num_workers=args.workers,
                              pin_memory=True, drop_last=True, persistent_workers=True)
    # val loader: non-persistent + few workers. With persistent_workers the val
    # pool would hold shm batches for the whole run on top of the train pool, and
    # at batch*4-procs that overflows the container's 64G /dev/shm (-> SIGBUS).
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=2, pin_memory=True, persistent_workers=False)
    spe = len(train_loader)
    total_steps = args.epochs * spe
    warmup_steps = args.warmup_epochs * spe

    model = ClsModel().to(device)
    core = DDP(model, device_ids=[local_rank], find_unused_parameters=True) if distributed else model
    opt = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if is_main:
        n = sum(p.numel() for p in model.parameters()) / 1e6
        log.info(f"ClsModel params: {n:.2f}M | per-gpu bs={args.batch_size} world={world} "
                 f"global_bs={args.batch_size*world} steps/epoch={spe} total_steps={total_steps}")

    start_epoch = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"]
        if is_main:
            log.info(f"resumed from {ckpt_path} -> start_epoch={start_epoch} best_acc={ck.get('best_acc',0):.4f}")
    best_acc = 0.0

    step = start_epoch * spe
    t0 = time.time(); run_steps = 0
    core.train()
    for ep in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)
        for x, y in train_loader:
            lr = cosine_lr(step, total_steps, args.lr, warmup_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = crit(core(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            if is_main:
                run_steps += 1
                if step % 50 == 0:
                    eta = (time.time() - t0) / max(1, run_steps) * (total_steps - step - 1)
                    log.info(f"ep{ep} [{step+1}/{total_steps} {100.0*(step+1)/total_steps:4.1f}%] "
                             f"lr={lr:.2e} loss={float(loss):.3f} eta={fmt_eta(eta)}")
            step += 1
        # ---- end epoch: eval + checkpoint (rank0) ----
        acc = -1.0
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
            acc = evaluate(core, val_loader, device)
        if is_main:
            if acc >= 0:
                log.info(f"[eval] epoch {ep+1} top1={acc:.4f}")
                best_acc = max(best_acc, acc)
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            tmp = ckpt_path + ".tmp"
            torch.save({"model": model.state_dict(), "backbone": model.backbone.state_dict(),
                        "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": ep + 1, "best_acc": best_acc}, tmp)
            os.replace(tmp, ckpt_path)
            # also save a backbone-only file (what segmentation loads) on new best / last epoch
            if acc >= best_acc or ep + 1 == args.epochs:
                torch.save({"backbone": model.backbone.state_dict()}, args.out)
            # permanent periodic snapshot (kept, not overwritten) so we have several ckpts
            if args.snapshot_every > 0 and (ep + 1) % args.snapshot_every == 0:
                snap = os.path.splitext(args.out)[0] + f".ep{ep+1}.pth"
                torch.save({"backbone": model.backbone.state_dict(),
                            "epoch": ep + 1, "acc": acc, "best_acc": best_acc}, snap)
                log.info(f"[snapshot] saved -> {snap}")
            log.info(f"[ckpt] saved epoch {ep+1} -> {ckpt_path} (best_acc={best_acc:.4f})")
    if distributed:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
