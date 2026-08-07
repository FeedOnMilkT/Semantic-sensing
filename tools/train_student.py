#!/usr/bin/env python
"""Train a Phase-2 split-inference student (one beta) and report its bpp-mIoU point.

  # quick wiring smoke (synthetic data, random teacher, CPU-safe):
  python tools/train_student.py configs/student_rtformer_late.yaml --smoke

  # real run: freeze a trained RTFormer teacher, train the bottleneck at one beta:
  CUDA_VISIBLE_DEVICES=0 python tools/train_student.py configs/student_rtformer_early.yaml \
      --teacher-ckpt /data/sidney/checkpoints/semantic-sensing/ade20k/ade20k_rtformer_bs32_lr4e4.pth \
      --beta 0.01 --out /data/sidney/checkpoints/semantic-sensing/students/early_b0.01.pth

A bpp-mIoU curve = sweep --beta across runs.
NOTE: forward/backward run on GPU fine; the rANS coder (compress/decompress) is CPU-only
and is exercised by tools/smoke_student_split.py, not here (this tool uses estimated bpp).
"""
import argparse
import copy
import os
import random

from _common import make_loader, pick_device
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from src import _compat  # noqa: F401  shim before CompressAI
from src.utils import load_config, get_logger
from src.models import build_model
from src.models.student import build_student
from src.engine import StudentTrainer, Stage2Trainer, evaluate_student
from src.datasets import collate
from src.datasets.synthetic import SyntheticMultiTask

log = get_logger("train_student")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="student config (configs/student_rtformer_{early,late}.yaml)")
    ap.add_argument("--teacher-ckpt", default=None, help="trained RTFormer teacher .pth (frozen)")
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2],
                    help="1 = train bottleneck (feature-MSE + beta*rate); 2 = freeze bottleneck, "
                         "fine-tune receiver tail (CE+Dice)")
    ap.add_argument("--student-ckpt", default=None, help="Stage-1 student .pth to load before Stage 2")
    ap.add_argument("--reference-student-ckpt", default=None,
                    help="frozen original-receiver checkpoint for controlled reference KD")
    ap.add_argument("--beta", type=float, default=0.01, help="rate weight (sweep this for the curve)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--aux-lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=None,
                    help="train epochs. Default = per-stage config (Stage1 student.stage1_epochs / "
                         "Stage2 stage2.epochs), NEVER a silent 1. --smoke uses 1 (synthetic wiring check only).")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--task-weight", type=float, default=0.0, help=">0 adds task loss (ICM arm)")
    ap.add_argument("--batch-size", type=int, default=None, help="override per-step batch size")
    ap.add_argument("--val-limit", type=int, default=None, help="eval on first N val images (sanity speed)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=20260717,
                    help="random seed for Python, NumPy, PyTorch, and DataLoader shuffling")
    ap.add_argument("--smoke", action="store_true", help="synthetic data + random teacher (wiring check)")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    log.info(f"seed={args.seed}")

    device = "cpu" if args.smoke else pick_device()
    cfg = load_config(args.config)
    num_classes = int(cfg.get("num_classes", 150))
    ignore_index = int(cfg.get("ignore_index", 255))

    # ---- frozen teacher ----
    if args.smoke or args.teacher_ckpt:
        cfg.model.backbone.args.pretrained = None     # ckpt (or random for smoke) supplies weights
    teacher = build_model(cfg.model).to(device).eval()
    if args.teacher_ckpt:
        state = torch.load(args.teacher_ckpt, map_location="cpu")
        teacher.load_state_dict(state["model"])
        log.info(f"loaded teacher {args.teacher_ckpt}")
    student = build_student(cfg, teacher).to(device)
    if args.student_ckpt:                                 # resume a Stage-1 student (for Stage 2)
        sd = torch.load(args.student_ckpt, map_location="cpu")
        student.load_state_dict(sd["student"])
        log.info(f"loaded student {args.student_ckpt}")
    n_codec = sum(p.numel() for p in student.codec_parameters()) / 1e6
    unfreeze = bool(cfg.student.get("unfreeze_sender", False))
    n_sender = sum(p.numel() for p in student.sender_parameters()) / 1e6 if unfreeze else 0.0
    log.info(f"split={cfg.student.split} stage={args.stage} bottleneck={cfg.bottleneck.name} "
             f"in_ch={student.bottleneck.in_channels} codec_params={n_codec:.2f}M "
             f"unfreeze_sender={unfreeze} sender_params={n_sender:.2f}M device={device}")

    # ---- data ----
    if args.smoke:
        size = (64, 128)
        ds = SyntheticMultiTask(tasks=["seg"], size=size, num_classes=num_classes, length=16)
        train_loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate)
    else:
        bs = args.batch_size or cfg.train.get("batch_size", 8)
        train_loader = make_loader(cfg.data.train, bs, shuffle=True)
        if args.val_limit:                                # fast, teacher-comparable sanity eval
            from torch.utils.data import Subset
            from src.datasets import build_dataset
            full_val = build_dataset(cfg.data.val)
            n = min(args.val_limit, len(full_val))        # clamp: never index past the val set
            vds = Subset(full_val, range(n))
            val_loader = DataLoader(vds, batch_size=4, shuffle=False, collate_fn=collate)
        else:
            val_loader = make_loader(cfg.data.val, cfg.eval.get("batch_size", 4), shuffle=False)

    # ---- resolve epochs: real per-stage config default, NEVER a silent 1 ----
    if args.smoke:
        epochs = args.epochs or 1                     # synthetic wiring check only
    elif args.stage == 2:
        epochs = args.epochs or int(cfg.get("stage2", {}).get("epochs", 20))
    else:
        epochs = args.epochs or int(OmegaConf.select(cfg, "student.stage1_epochs") or 100)
    log.info(f"training stage={args.stage} epochs={epochs} (real schedule)")

    # ---- train ----
    if args.stage == 2:
        # resolve ${num_classes}/${ignore_index} against the full cfg BEFORE detaching
        s2cfg = OmegaConf.create(OmegaConf.to_container(cfg.get("stage2", {}), resolve=True))
        s2cfg.setdefault("num_classes", num_classes)
        s2cfg.setdefault("ignore_index", ignore_index)
        if "optimizer" not in s2cfg:
            s2cfg.optimizer = {"lr": args.lr, "weight_decay": 0.01}
        reference_student = None
        reference_kd = s2cfg.get("reference_kd", None) or {}
        if float(reference_kd.get("weight", 0.0)) > 0:
            if not args.reference_student_ckpt:
                raise ValueError("reference_kd requires --reference-student-ckpt")
            if cfg.student.split != "mid":
                raise ValueError("controlled reference_kd is only implemented for the mid split")
            reference_student = copy.deepcopy(student)
            reference_state = torch.load(
                args.reference_student_ckpt, map_location="cpu", weights_only=False)
            reference_student.load_state_dict(reference_state["student"], strict=True)
            reference_student.teacher.backbone.stage4.gfa.mode = "original"
            reference_student.teacher.backbone.stage5.gfa.mode = "original"
            reference_student.eval()
            for param in reference_student.parameters():
                param.requires_grad_(False)
            log.info("reference KD: original receiver, frozen/eval, same reconstruction; "
                     f"weight={float(reference_kd.weight):g} "
                     f"temperature={float(reference_kd.temperature):g} "
                     f"ckpt={args.reference_student_ckpt}")
        trainer = Stage2Trainer(
            s2cfg, student, device=device, reference_student=reference_student)
        trainer.fit(train_loader, epochs=epochs, max_steps=args.max_steps,
                    log_every=cfg.train.get("log_every", 20))
    else:
        tcfg = OmegaConf.create({
            "optimizer": {"lr": args.lr, "weight_decay": 0.0}, "aux_lr": args.aux_lr,
            "scheduler": "poly", "warmup": 0, "grad_clip": 1.0,
            "task_weight": args.task_weight, "ignore_index": ignore_index,
            "unfreeze_sender": unfreeze,
            "sender_lr": float(cfg.student.get("sender_lr", 1e-4)),
        })
        trainer = StudentTrainer(tcfg, student, beta=args.beta, device=device)
        trainer.fit(train_loader, epochs=epochs, max_steps=args.max_steps,
                    log_every=cfg.train.get("log_every", 20))

    # ---- eval: the bpp-mIoU point ----
    if not args.no_eval:
        m = evaluate_student(student, val_loader, num_classes, device=device, ignore_index=ignore_index)
        log.info(f"[POINT] split={cfg.student.split} beta={args.beta:g} "
                 f"bpp={m['bpp']:.4f} mIoU={m['mIoU']:.4f} pixel_acc={m['pixel_acc']:.4f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save({"student": student.state_dict(), "beta": args.beta, "cfg": cfg}, args.out)
        log.info(f"saved student -> {args.out}")


if __name__ == "__main__":
    main()
