#!/usr/bin/env python
"""Detailed ADE20K evaluation for a Phase-2 SplitStudent checkpoint."""
import argparse
import hashlib
import json
import os

from _common import pick_device
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from src import _compat  # noqa: F401
from src.datasets import build_dataset, collate
from src.losses.compression import bpp_from_info
from src.losses.task_losses import SegLoss
from src.models import build_model
from src.models.student import build_student
from src.utils import load_config
from src.utils.metrics import SegMeter


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--teacher-ckpt", required=True)
    ap.add_argument("--student-ckpt", required=True)
    ap.add_argument("--gfa-mode", required=True,
                    choices=["original", "group_softmax", "memory_softmax"])
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260717)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    cfg = load_config(args.config)
    cfg.model.backbone.args.pretrained = None
    cfg.model.backbone.args.gfa_mode = args.gfa_mode
    num_classes = int(cfg.get("num_classes", 150))
    ignore_index = int(cfg.get("ignore_index", 255))

    teacher = build_model(cfg.model).to(device).eval()
    teacher_state = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    teacher_load = teacher.load_state_dict(teacher_state["model"], strict=True)
    student = build_student(cfg, teacher).to(device).eval()
    student_state = torch.load(args.student_ckpt, map_location="cpu", weights_only=False)
    student_load = student.load_state_dict(student_state["student"], strict=True)

    dataset = build_dataset(cfg.data.val)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate, pin_memory=True)
    meter = SegMeter(num_classes=num_classes, ignore_index=ignore_index)
    loss_cfg = cfg.stage2.seg_loss
    loss_fn = SegLoss(
        num_classes=num_classes,
        ignore_index=ignore_index,
        w_ce=float(loss_cfg.get("w_ce", 1.0)),
        w_dice=float(loss_cfg.get("w_dice", 1.0)),
        w_lovasz=float(loss_cfg.get("w_lovasz", 0.0)),
    ).to(device)

    loss_sum = 0.0
    bpp_sum = 0.0
    samples = 0
    with torch.inference_mode():
        for step, batch in enumerate(loader):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["targets"]["seg"].to(device, non_blocking=True)
            output = student(image)
            logits = output["preds"]["seg"]
            batch_size, _, height, width = image.shape
            meter.update(logits.float().cpu(), batch["targets"]["seg"])
            loss_sum += float(loss_fn(logits, target)) * batch_size
            bpp_sum += float(bpp_from_info(
                output["info"], batch_size * height * width)) * batch_size
            samples += batch_size
            if step % 100 == 0:
                print(f"eval {step + 1}/{len(loader)} samples={samples}", flush=True)

    metrics = meter.compute()
    confusion = meter.conf
    true_positive = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - true_positive
    per_class_iou = [
        float(true_positive[i] / union[i]) if union[i] > 0 else None
        for i in range(num_classes)
    ]
    result = {
        "config": os.path.abspath(args.config),
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        "teacher_checkpoint": os.path.abspath(args.teacher_ckpt),
        "teacher_sha256": _sha256(args.teacher_ckpt),
        "student_checkpoint": os.path.abspath(args.student_ckpt),
        "student_sha256": _sha256(args.student_ckpt),
        "gfa_mode": args.gfa_mode,
        "dataset": {
            "name": type(dataset).__name__,
            "split": "validation",
            "samples": len(dataset),
            "preprocess": OmegaConf.to_container(cfg.data.val, resolve=True),
        },
        "evaluator": {
            "name": "SegMeter + Stage2 SegLoss",
            "batch_size": args.batch_size,
            "num_classes": num_classes,
            "ignore_index": ignore_index,
            "seed": args.seed,
        },
        "strict_load": {
            "teacher_missing": list(teacher_load.missing_keys),
            "teacher_unexpected": list(teacher_load.unexpected_keys),
            "student_missing": list(student_load.missing_keys),
            "student_unexpected": list(student_load.unexpected_keys),
        },
        "metrics": {
            **metrics,
            "loss": loss_sum / max(samples, 1),
            "bpp": bpp_sum / max(samples, 1),
            "per_class_iou": per_class_iou,
        },
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["metrics"], indent=2))
    print(f"saved {args.out_json}")


if __name__ == "__main__":
    main()
