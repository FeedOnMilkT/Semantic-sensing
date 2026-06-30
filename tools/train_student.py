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
import os

from _common import make_loader, pick_device
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from src import _compat  # noqa: F401  shim before CompressAI
from src.utils import load_config, get_logger
from src.models import build_model
from src.models.student import build_student
from src.engine import StudentTrainer, evaluate_student
from src.datasets import collate
from src.datasets.synthetic import SyntheticMultiTask

log = get_logger("train_student")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="student config (configs/student_rtformer_{early,late}.yaml)")
    ap.add_argument("--teacher-ckpt", default=None, help="trained RTFormer teacher .pth (frozen)")
    ap.add_argument("--beta", type=float, default=0.01, help="rate weight (sweep this for the curve)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--aux-lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--task-weight", type=float, default=0.0, help=">0 adds task loss (ICM arm)")
    ap.add_argument("--batch-size", type=int, default=None, help="override per-step batch size")
    ap.add_argument("--val-limit", type=int, default=None, help="eval on first N val images (sanity speed)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="synthetic data + random teacher (wiring check)")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

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
    n_codec = sum(p.numel() for p in student.codec_parameters()) / 1e6
    log.info(f"split={cfg.student.split} bottleneck={cfg.bottleneck.name} "
             f"in_ch={student.bottleneck.in_channels} codec_params={n_codec:.2f}M device={device}")

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
            vds = Subset(build_dataset(cfg.data.val), range(args.val_limit))
            val_loader = DataLoader(vds, batch_size=4, shuffle=False, collate_fn=collate)
        else:
            val_loader = make_loader(cfg.data.val, cfg.eval.get("batch_size", 4), shuffle=False)

    # ---- train ----
    tcfg = OmegaConf.create({
        "optimizer": {"lr": args.lr, "weight_decay": 0.0}, "aux_lr": args.aux_lr,
        "scheduler": "poly", "warmup": 0, "grad_clip": 1.0,
        "task_weight": args.task_weight, "ignore_index": ignore_index,
    })
    trainer = StudentTrainer(tcfg, student, beta=args.beta, device=device)
    trainer.fit(train_loader, epochs=args.epochs, max_steps=args.max_steps,
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
