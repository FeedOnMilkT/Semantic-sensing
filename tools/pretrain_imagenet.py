#!/usr/bin/env python
"""ImageNet-1K classification pretraining for the RTFormer backbone.

Trains RTFormer's trunk (stem->stage5, via forward_trunk -> s32) + a linear
classifier. DAPPM is not part of the pretraining forward and is excluded from the
exported backbone checkpoint. DDP, BF16/FP16 autocast, cosine LR with warmup,
label smoothing, per-epoch resume checkpoint, and best-validation backbone export.

Launch (4 GPUs, batch is PER-GPU):
  HF_HOME=/data/sidney/hf-cache CUDA_VISIBLE_DEVICES=0,1,4,5 \
    torchrun --nproc_per_node=4 --master_port=29701 tools/pretrain_imagenet.py \
    --epochs 100 --batch-size 128 --lr 1e-4 --precision bf16 --resume \
    --out /data/sidney/checkpoints/semantic-sensing/pretrained/rtformer_in1k.pth
"""
import argparse
import math
import os
import random
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
from PIL import Image

from src.models.backbones import RTFormer
from src.utils.logging import get_logger

log = get_logger("pretrain")

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_IMAGENET1K_ROWS = {"train": 1_281_167, "validation": 50_000}


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
        token = get_token()
        if not token:
            raise RuntimeError("ImageNet-1K requires an authorized Hugging Face token; run `hf auth login` first.")
        self.ds = load_dataset("ILSVRC/imagenet-1k", split=split, token=token)
        self.tf = tf

    def __len__(self):
        return self.ds.num_rows

    @property
    def label_names(self):
        feature = self.ds.features.get("label")
        return getattr(feature, "names", None)

    def __getitem__(self, i):
        ex = self.ds[i]
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.tf(img), int(ex["label"])


class SyntheticImageNet(Dataset):
    """Tiny deterministic fixture for the real CLI dry-run branch; never for training."""
    def __init__(self, split, tf):
        self.n = 4
        self.tf = tf
        self.label_names = [f"class_{i}" for i in range(1000)]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        image = Image.new("RGB", (256, 256), color=((i * 37) % 256, (i * 71) % 256, (i * 113) % 256))
        return self.tf(image), i


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


def autocast_context(device, precision):
    device_type = torch.device(device).type
    if precision == "fp32" or device_type not in ("cuda", "mps"):
        return torch.autocast(device_type=device_type, enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device_type, dtype=dtype)


def _backbone_for_export(backbone):
    # DAPPM is a segmentation head and is not executed by forward_trunk().
    return {k: v for k, v in backbone.state_dict().items() if not k.startswith("dappm.")}


@torch.no_grad()
def evaluate(model, loader, device, precision="fp32"):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_context(device, precision):
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item(); total += y.numel()
    t = torch.tensor([correct, total], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(t)
    model.train()
    return (t[0] / t[1]).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100,
                    help="100 is an engineering default; the paper does not specify ImageNet pretraining epochs")
    ap.add_argument("--batch-size", type=int, default=128, help="PER-GPU; server prompt targets effective batch 128")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="engineering default scaled from the repository's larger-batch run; not paper-specified")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16",
                    help="BF16 autocast is the requested server setting; images remain RGB/FP32 inputs")
    ap.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="if >0, also keep a permanent backbone snapshot every N epochs")
    ap.add_argument("--out", default="/data/sidney/checkpoints/semantic-sensing/pretrained/rtformer_in1k.pth")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="load real gated data and run one train + validation batch, then exit without checkpoints")
    ap.add_argument("--prepare-data-only", action="store_true",
                    help="download/verify both gated ImageNet splits and exit without creating a model")
    ap.add_argument("--synthetic-data", action="store_true",
                    help="use a four-sample local fixture; only allowed together with --dry-run")
    args = ap.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.eval_every < 1 or args.workers < 0:
        ap.error("epochs, batch-size and eval-every must be positive; workers must be nonnegative")
    if args.synthetic_data and not args.dry_run:
        ap.error("--synthetic-data is only allowed with --dry-run; it must never be used for training")
    if args.synthetic_data and args.prepare_data_only:
        ap.error("--synthetic-data cannot be combined with --prepare-data-only")
    if args.dry_run and args.resume:
        ap.error("--resume cannot be combined with --dry-run")
    ckpt_path = os.path.splitext(args.out)[0] + ".last.pth"

    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed = world > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun DDP pretraining requires CUDA on every worker")
        dist.init_process_group("nccl"); torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        if args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but MPS is unavailable")
    if args.precision == "bf16" and torch.device(device).type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 was requested, but this CUDA device does not report BF16 support")
    if args.precision == "fp16" and torch.device(device).type != "cuda":
        raise RuntimeError("FP16 autocast is currently supported only on CUDA")
    is_main = rank == 0

    tf_train, tf_val = build_transforms(args.img_size)
    if is_main:
        source = "synthetic fixture" if args.synthetic_data else f"ImageNet-1K (HF_HOME={os.environ.get('HF_HOME')})"
        log.info(f"loading {source} ...")
    if args.synthetic_data:
        train_ds = SyntheticImageNet("train", tf_train)
        val_ds = SyntheticImageNet("validation", tf_val)
    else:
        train_ds = HFImageNet("train", tf_train)
        val_ds = HFImageNet("validation", tf_val)
    for split_name, dataset in (("train", train_ds), ("validation", val_ds)):
        names = dataset.label_names
        if args.synthetic_data:
            if len(dataset) == 0 or names is None or len(names) != 1000:
                raise RuntimeError(f"bad synthetic ImageNet {split_name} fixture")
        elif len(dataset) != _IMAGENET1K_ROWS[split_name] or names is None or len(names) != 1000:
            raise RuntimeError(f"unexpected ImageNet {split_name} metadata: rows={len(dataset)}, classes="
                               f"{None if names is None else len(names)} (expected "
                               f"{_IMAGENET1K_ROWS[split_name]} rows / 1000 classes)")
    if train_ds.label_names != val_ds.label_names:
        raise RuntimeError("ImageNet train and validation class-name metadata differs")
    if is_main:
        source = "synthetic fixture" if args.synthetic_data else "ImageNet HF"
        log.info(f"{source} rows: train={len(train_ds):,}, validation={len(val_ds):,}; "
                 f"classes={len(train_ds.label_names) if train_ds.label_names else 'metadata unavailable'}")
    if args.prepare_data_only:
        if is_main:
            log.info("ImageNet split download/metadata check complete; no model or checkpoint created")
        if distributed:
            dist.barrier(); dist.destroy_process_group()
        return
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              shuffle=train_sampler is None, num_workers=args.workers,
                              pin_memory=torch.device(device).type == "cuda", drop_last=True,
                              persistent_workers=args.workers > 0)
    # val loader: non-persistent + few workers. With persistent_workers the val
    # pool would hold shm batches for the whole run on top of the train pool, and
    # at batch*4-procs that overflows the container's 64G /dev/shm (-> SIGBUS).
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=min(args.workers, 2),
                            pin_memory=torch.device(device).type == "cuda", persistent_workers=False)
    spe = len(train_loader)
    total_steps = args.epochs * spe
    warmup_steps = args.warmup_epochs * spe

    model = ClsModel().to(device)
    core = DDP(model, device_ids=[local_rank], find_unused_parameters=True) if distributed else model
    opt = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler_enabled = args.precision == "fp16" and torch.device(device).type == "cuda"
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    else:  # compatibility with older local CPU environments; server should use current CUDA PyTorch
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if is_main:
        n = sum(p.numel() for p in model.parameters()) / 1e6
        log.info(f"ClsModel params: {n:.2f}M | per-gpu bs={args.batch_size} world={world} "
                 f"global_bs={args.batch_size*world} steps/epoch={spe} total_steps={total_steps}")

    if args.dry_run:
        if torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(device=device)
        core.train()
        x, y = next(iter(train_loader))
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast_context(device, args.precision):
            loss = crit(core(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite ImageNet dry-run train loss: {loss.detach()}")
        if scaler.is_enabled():
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        core.eval()
        vx, vy = next(iter(val_loader))
        vx, vy = vx.to(device, non_blocking=True), vy.to(device, non_blocking=True)
        with torch.no_grad(), autocast_context(device, args.precision):
            vp = core(vx).argmax(1)
        counts = torch.tensor([(vp == vy).sum().item(), vy.numel()], device=device, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(counts)
        val_acc = (counts[0] / counts[1]).item()
        if is_main:
            memory = (f" peak_allocated={torch.cuda.max_memory_allocated(device) / 2**30:.2f}GiB"
                      f" peak_reserved={torch.cuda.max_memory_reserved(device) / 2**30:.2f}GiB"
                      if torch.device(device).type == "cuda" else "")
            log.info(f"[dry-run] train_loss={float(loss):.4f} val_batch_top1={val_acc:.4f}; "
                     f"real data, forward/backward, no checkpoint written{memory}")
        if distributed:
            dist.barrier(); dist.destroy_process_group()
        return

    start_epoch = 0
    best_acc = 0.0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck.get("world_size", world) != world or ck.get("steps_per_epoch", spe) != spe:
            raise RuntimeError("resume checkpoint world size or steps/epoch differs from this run")
        saved_schedule = ck.get("schedule", {})
        requested_schedule = {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                              "weight_decay": args.weight_decay, "warmup_epochs": args.warmup_epochs,
                              "precision": args.precision}
        if saved_schedule and saved_schedule != requested_schedule:
            raise RuntimeError(f"resume schedule differs: saved={saved_schedule}, requested={requested_schedule}")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck.get("scaler", {}))
        start_epoch = ck["epoch"]
        best_acc = float(ck.get("best_acc", 0.0))
        rank_rng = ck.get("rng_by_rank", [])
        if rank_rng:
            state = rank_rng[rank]
            torch.set_rng_state(state["torch"].cpu())
            random.setstate(state["python"])
            if torch.device(device).type == "cuda" and "cuda" in state:
                torch.cuda.set_rng_state(state["cuda"], device=device)
        if is_main:
            log.info(f"resumed from {ckpt_path} -> start_epoch={start_epoch} best_acc={ck.get('best_acc',0):.4f}")

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
            with autocast_context(device, args.precision):
                loss = crit(core(x), y)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                opt.step()
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
            acc = evaluate(core, val_loader, device, args.precision)
        local_rng = {"torch": torch.get_rng_state(), "python": random.getstate()}
        if torch.device(device).type == "cuda":
            local_rng["cuda"] = torch.cuda.get_rng_state(device=device)
        rng_by_rank = [local_rng]
        if distributed:
            rng_by_rank = [None for _ in range(world)]
            dist.all_gather_object(rng_by_rank, local_rng)
        if is_main:
            is_best = acc >= 0 and acc >= best_acc
            if acc >= 0:
                log.info(f"[eval] epoch {ep+1} top1={acc:.4f}")
                best_acc = max(best_acc, acc)
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            tmp = ckpt_path + ".tmp"
            torch.save({"model": model.state_dict(), "backbone": _backbone_for_export(model.backbone),
                        "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": ep + 1, "best_acc": best_acc, "world_size": world,
                        "steps_per_epoch": spe,
                        "schedule": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                                     "weight_decay": args.weight_decay, "warmup_epochs": args.warmup_epochs,
                                     "precision": args.precision},
                        "rng_by_rank": rng_by_rank}, tmp)
            os.replace(tmp, ckpt_path)
            # Export only on validation improvement. ADE20K loads this wrapper directly.
            if is_best:
                export_tmp = args.out + ".tmp"
                torch.save({"backbone": _backbone_for_export(model.backbone),
                            "epoch": ep + 1, "acc": acc, "best_acc": best_acc}, export_tmp)
                os.replace(export_tmp, args.out)
            # permanent periodic snapshot (kept, not overwritten) so we have several ckpts
            if args.snapshot_every > 0 and (ep + 1) % args.snapshot_every == 0:
                snap = os.path.splitext(args.out)[0] + f".ep{ep+1}.pth"
                torch.save({"backbone": _backbone_for_export(model.backbone),
                            "epoch": ep + 1, "acc": acc, "best_acc": best_acc}, snap)
                log.info(f"[snapshot] saved -> {snap}")
            log.info(f"[ckpt] saved epoch {ep+1} -> {ckpt_path} (best_acc={best_acc:.4f})")
    if distributed:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
