#!/usr/bin/env python
"""ImageNet-1K classification pretraining for the RTFormer-Base trunk.

Uses the official supplement's 300-epoch AdamW/cosine recipe and augmentations.
Physical batch 128 x accumulation 8 gives the paper's effective batch 1024 on
one GPU. BF16 is the requested project execution override, not the paper's stated
precision. The temporary GAP+linear classifier pools both stage-5 streams; DAPPM
and segmentation heads are not part of ImageNet classification.
"""
import argparse
from contextlib import nullcontext
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
from timm.data import Mixup
from timm.data.transforms_factory import transforms_imagenet_train
import yaml
from PIL import Image

from src.models.backbones import RTFormer
from src.utils.logging import get_logger

log = get_logger("pretrain")

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_IMAGENET1K_ROWS = {"train": 1_281_167, "validation": 50_000}


class ClsModel(nn.Module):
    """RTFormer trunk -> GAP of both stage-5 streams -> temporary 1000-way head.

    The paper reports ImageNet classification but not its classifier-head design;
    pooling and concatenating both streams is an explicit engineering choice that
    ensures every trunk branch receives classification gradients.
    """
    def __init__(self, num_classes=1000, drop_path_rate=0.0,
                 use_injection=(True, False)):
        super().__init__()
        self.backbone = RTFormer(drop_path_rate=drop_path_rate,
                                 use_injection=tuple(use_injection))
        c_h = self.backbone.split_channels["mid"][0]
        c_l = self.backbone.out_channels["s32"]
        self.head = nn.Linear(c_h + c_l, num_classes)

    def forward(self, x):
        high, low = self.backbone.forward_trunk(x)
        high = F.adaptive_avg_pool2d(high, 1).flatten(1)
        low = F.adaptive_avg_pool2d(low, 1).flatten(1)
        return self.head(torch.cat((high, low), dim=1))


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


def build_transforms(img_size, recipe):
    # Supplement notation RandAug(9, 0.5) is interpreted with timm's standard
    # rand-m9-mstd0.5 policy (two ops/image); crop/interpolation are not specified
    # by the paper and use timm's ImageNet defaults (engineering choices).
    train = transforms_imagenet_train(
        img_size=img_size,
        auto_augment=recipe["rand_augment"],
        color_jitter=recipe["color_jitter"], re_prob=recipe["random_erasing"],
        # timm disables color jitter whenever RandAugment is active unless this
        # override is set; the supplement lists both augmentations.
        force_color_jitter=True,
        re_mode="const", re_count=1, interpolation=recipe["interpolation"],
        mean=_MEAN, std=_STD,
    )
    val = T.Compose([
        T.Resize(int(img_size * 256 / 224), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(), T.Normalize(_MEAN, _STD),
    ])
    return train, val


def cosine_lr(step, total, base_lr, warmup, min_lr=5e-6, warmup_start_lr=5e-7):
    if step < warmup:
        return warmup_start_lr + (base_lr - warmup_start_lr) * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup - 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def fmt_eta(sec):
    sec = int(max(0, sec)); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _schedule_state(args, recipe, world, effective_batch, updates_per_epoch):
    """Capture every setting that changes data, augmentation, or update timing."""
    recipe_fields = (
        "drop_path_rate", "use_injection", "rand_augment", "color_jitter",
        "random_erasing", "interpolation", "mixup_alpha", "cutmix_alpha",
        "mixup_cutmix_switch_prob", "label_smoothing", "num_classes",
    )
    return {
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay, "warmup_epochs": args.warmup_epochs,
        "warmup_start_lr": args.warmup_start_lr, "min_lr": args.min_lr,
        "img_size": args.img_size, "precision": args.precision,
        "accumulation_steps": args.accumulation_steps, "world_size": world,
        "effective_batch": effective_batch,
        "updates_per_epoch": updates_per_epoch,
        "recipe": {key: recipe[key] for key in recipe_fields},
    }


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
    default_config = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "configs", "imagenet_rtformer_original.yaml")
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=default_config)
    boot_args, _ = bootstrap.parse_known_args()
    with open(boot_args.config, encoding="utf-8") as f:
        recipe = yaml.safe_load(f)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=boot_args.config)
    ap.add_argument("--epochs", type=int, default=recipe["epochs"],
                    help="official supplement ImageNet recipe: 300 epochs")
    ap.add_argument("--batch-size", type=int, default=recipe["batch_size"],
                    help="physical batch per rank; default 128 x 8 accumulation = 1024 on one GPU")
    ap.add_argument("--accumulation-steps", type=int, default=recipe["accumulation_steps"],
                    help="microbatches per optimizer update; paper effective batch is 1024")
    ap.add_argument("--lr", type=float, default=recipe["lr"])
    ap.add_argument("--weight-decay", type=float, default=recipe["weight_decay"])
    ap.add_argument("--warmup-epochs", type=int, default=recipe["warmup_epochs"])
    ap.add_argument("--warmup-start-lr", type=float, default=recipe["warmup_start_lr"])
    ap.add_argument("--min-lr", type=float, default=recipe["min_lr"])
    ap.add_argument("--img-size", type=int, default=recipe["img_size"])
    ap.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default=recipe["precision"],
                    help="BF16 autocast is the requested server setting; images remain RGB/FP32 inputs")
    ap.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default=recipe["device"])
    ap.add_argument("--workers", type=int, default=recipe["workers"])
    ap.add_argument("--eval-every", type=int, default=recipe["eval_every"])
    ap.add_argument("--snapshot-every", type=int, default=recipe["snapshot_every"],
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
    if (args.epochs < 1 or args.batch_size < 1 or args.accumulation_steps < 1
            or args.eval_every < 1 or args.workers < 0):
        ap.error("epochs, batch-size, accumulation-steps and eval-every must be positive; workers must be nonnegative")
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
    effective_batch = args.batch_size * world * args.accumulation_steps
    if not args.dry_run and effective_batch != recipe["effective_batch_size"]:
        raise RuntimeError(
            f"effective batch must match configured {recipe['effective_batch_size']}; got "
            f"{args.batch_size}*{world}*{args.accumulation_steps}={effective_batch}. "
            "Adjust --accumulation-steps explicitly.")

    tf_train, tf_val = build_transforms(args.img_size, recipe)
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
    updates_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
    total_steps = args.epochs * updates_per_epoch
    warmup_steps = args.warmup_epochs * updates_per_epoch

    model = ClsModel(num_classes=recipe["num_classes"],
                     drop_path_rate=recipe["drop_path_rate"],
                     use_injection=recipe["use_injection"]).to(device)
    core = DDP(model, device_ids=[local_rank], find_unused_parameters=True) if distributed else model
    opt = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            betas=tuple(recipe["betas"]))
    scaler_enabled = args.precision == "fp16" and torch.device(device).type == "cuda"
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    else:  # compatibility with older local CPU environments; server should use current CUDA PyTorch
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    mixup_fn = Mixup(mixup_alpha=recipe["mixup_alpha"], cutmix_alpha=recipe["cutmix_alpha"],
                     prob=1.0, switch_prob=recipe["mixup_cutmix_switch_prob"], mode="batch",
                     label_smoothing=recipe["label_smoothing"],
                     num_classes=recipe["num_classes"])
    if is_main:
        n = sum(p.numel() for p in model.parameters()) / 1e6
        log.info(f"ClsModel params: {n:.2f}M | per-rank microbatch={args.batch_size} world={world} "
                 f"accumulation={args.accumulation_steps} effective_bs={effective_batch} "
                 f"updates/epoch={updates_per_epoch} total_updates={total_steps}")

    if args.dry_run:
        if torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(device=device)
        core.train()
        x, y = next(iter(train_loader))
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        x, targets = mixup_fn(x, y)
        with autocast_context(device, args.precision):
            logits = core(x)
            loss = -(targets * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()
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
            log.info(f"[dry-run] source={'synthetic fixture' if args.synthetic_data else 'ImageNet HF'} "
                     f"train_loss={float(loss.detach()):.4f} val_batch_top1={val_acc:.4f}; "
                     f"forward/backward complete, no checkpoint written{memory}")
        if distributed:
            dist.barrier(); dist.destroy_process_group()
        return

    start_epoch = 0
    best_acc = 0.0
    global_step = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if (ck.get("world_size", world) != world
                or ck.get("updates_per_epoch", updates_per_epoch) != updates_per_epoch):
            raise RuntimeError("resume checkpoint world size or optimizer updates/epoch differs")
        saved_schedule = ck.get("schedule", {})
        requested_schedule = _schedule_state(args, recipe, world, effective_batch, updates_per_epoch)
        if saved_schedule and saved_schedule != requested_schedule:
            raise RuntimeError(f"resume schedule differs: saved={saved_schedule}, requested={requested_schedule}")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck.get("scaler", {}))
        start_epoch = ck["epoch"]
        global_step = int(ck.get("global_step", start_epoch * updates_per_epoch))
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

    t0 = time.time(); run_steps = 0
    core.train()
    for ep in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)
        opt.zero_grad(set_to_none=True)
        window_samples = 0
        window_loss = 0.0
        for micro_idx, (x, y) in enumerate(train_loader):
            window_start = micro_idx % args.accumulation_steps == 0
            update_now = ((micro_idx + 1) % args.accumulation_steps == 0
                          or micro_idx + 1 == len(train_loader))
            if window_start:
                lr = cosine_lr(global_step, total_steps, args.lr, warmup_steps,
                               min_lr=args.min_lr, warmup_start_lr=args.warmup_start_lr)
                for group in opt.param_groups:
                    group["lr"] = lr
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            x, targets = mixup_fn(x, y)
            sync_context = (core.no_sync() if distributed and not update_now else nullcontext())
            with sync_context:
                with autocast_context(device, args.precision):
                    logits = core(x)
                    loss_sum = -(targets * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).sum()
                if not torch.isfinite(loss_sum):
                    raise FloatingPointError(f"non-finite ImageNet loss at epoch {ep + 1}")
                if scaler.is_enabled():
                    scaler.scale(loss_sum).backward()
                else:
                    loss_sum.backward()
            window_samples += y.numel()
            window_loss += float(loss_sum.detach())
            if update_now:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                for parameter in core.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(window_samples)
                if scaler.is_enabled():
                    old_scale = scaler.get_scale()
                    scaler.step(opt)
                    scaler.update()
                    did_update = scaler.get_scale() >= old_scale
                else:
                    opt.step()
                    did_update = True
                opt.zero_grad(set_to_none=True)
                if is_main and did_update:
                    run_steps += 1
                    if global_step % 50 == 0:
                        eta = (time.time() - t0) / max(1, run_steps) * (total_steps - global_step - 1)
                        log.info(f"ep{ep + 1} [{global_step + 1}/{total_steps} "
                                 f"{100.0 * (global_step + 1) / total_steps:4.1f}%] "
                                 f"lr={lr:.2e} loss={window_loss / window_samples:.3f} "
                                 f"window_samples={window_samples} eta={fmt_eta(eta)}")
                if did_update:
                    global_step += 1
                elif is_main:
                    log.warning(f"[amp] skipped optimizer update at epoch {ep + 1}; "
                                "global scheduler step was not advanced")
                window_samples = 0
                window_loss = 0.0
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
                        "global_step": global_step,
                        "updates_per_epoch": updates_per_epoch,
                        "schedule": _schedule_state(args, recipe, world, effective_batch,
                                                     updates_per_epoch),
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
