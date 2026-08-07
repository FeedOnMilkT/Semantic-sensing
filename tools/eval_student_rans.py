#!/usr/bin/env python
"""Full-val FP32 evaluation with estimated bpp AND real rANS bitstream bpp.

Primary path: CPU compress()/decompress() -> receiver -> SegMeter.
Also records eval-mode forward metrics for comparison.
JSON shape includes metrics.per_class_iou for large_object_metric.py.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _common import pick_device  # noqa: F401
from src.datasets import build_dataset, collate
from src.losses.compression import bpp_from_info
from src.models import build_model
from src.models.student import build_student
from src.utils.config import load_config
from src.utils.metrics import SegMeter


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _per_class_iou(meter: SegMeter):
    conf = meter.conf
    tp = conf.diag()
    union = conf.sum(0) + conf.sum(1) - tp
    return [
        float(tp[i] / union[i]) if union[i] > 0 else None
        for i in range(meter.n)
    ]


def _load_student(ckpt: Path, device: torch.device, gfa_mode: str | None):
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = state["cfg"]
    cfg.model.backbone.args.pretrained = None
    if gfa_mode is not None:
        cfg.model.backbone.args.gfa_mode = gfa_mode
    teacher = build_model(cfg.model).eval()
    student = build_student(cfg, teacher).eval()
    student.load_state_dict(state["student"], strict=True)
    student.update(force=True)
    for p in student.parameters():
        p.requires_grad_(False)
    return student.to(device), cfg, state.get("beta")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student-ckpt", required=True)
    ap.add_argument("--config", default=None, help="optional override for data.val")
    ap.add_argument("--gfa-mode", default=None,
                    choices=["original", "group_softmax", "memory_softmax"])
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = Path(args.student_ckpt).resolve()
    student, cfg, beta = _load_student(ckpt, device, args.gfa_mode)
    data_cfg = load_config(args.config).data.val if args.config else cfg.data.val
    dataset = build_dataset(data_cfg)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate
    )

    num_classes = int(cfg.get("num_classes", 150))
    ignore_index = int(cfg.get("ignore_index", 255))
    meter_eval = SegMeter(num_classes, ignore_index)
    meter_rans = SegMeter(num_classes, ignore_index)

    est_bpp_sum = 0.0
    act_bpp_sum = 0.0
    n_img = 0
    cpu_bottleneck = copy.deepcopy(student.bottleneck).cpu().eval()
    cpu_bottleneck.update(force=True)

    student.eval()
    total = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    with torch.inference_mode():
        for idx, batch in enumerate(loader):
            if args.limit is not None and idx >= args.limit:
                break
            image = batch["image"].to(device)
            target = batch["targets"]["seg"]
            B, _, H, W = image.shape
            assert B == 1

            out = student(image)
            est_bpp = float(bpp_from_info(out["info"], B * H * W).item())
            meter_eval.update(out["preds"]["seg"].float().cpu(), target)

            feats, ctx = student._sender(image)
            feats_cpu = tuple(t.detach().cpu() for t in feats)
            payload = cpu_bottleneck.compress(feats_cpu)
            recon = cpu_bottleneck.decompress(payload)
            recon_dev = tuple(t.to(device) for t in recon)
            preds = student._receiver(recon_dev, ctx)
            meter_rans.update(preds["seg"].float().cpu(), target)

            actual_bpp = len(payload) * 8.0 / float(H * W)
            est_bpp_sum += est_bpp
            act_bpp_sum += actual_bpp
            n_img += 1
            if (n_img % 50) == 0 or n_img == total:
                print(
                    f"[{n_img}/{total}] est_bpp={est_bpp_sum/n_img:.4f} "
                    f"act_bpp={act_bpp_sum/n_img:.4f}",
                    flush=True,
                )

    m_eval = meter_eval.compute()
    m_rans = meter_rans.compute()
    per_class = _per_class_iou(meter_rans)
    result = {
        "student_checkpoint": str(ckpt),
        "student_sha256": _sha256(ckpt),
        "beta": beta,
        "gfa_mode": cfg.model.backbone.args.get("gfa_mode", "original"),
        "split": cfg.student.split,
        "bottleneck": cfg.bottleneck.name,
        "activation": cfg.bottleneck.args.get("act"),
        "num_images": n_img,
        "estimated_bpp": est_bpp_sum / max(1, n_img),
        "actual_bitstream_bpp": act_bpp_sum / max(1, n_img),
        "eval_path": {
            "mIoU": float(m_eval["mIoU"]),
            "pixel_acc": float(m_eval["pixel_acc"]),
            "note": "CompressAI eval-mode quantization (not rANS bytes)",
        },
        "rans_path": {
            "mIoU": float(m_rans["mIoU"]),
            "pixel_acc": float(m_rans["pixel_acc"]),
            "note": "real CPU compress()/decompress() rANS bitstream reconstruction",
        },
        # Compatible with large_object_metric.py (uses metrics.per_class_iou)
        "metrics": {
            "mIoU": float(m_rans["mIoU"]),
            "pixel_acc": float(m_rans["pixel_acc"]),
            "bpp": act_bpp_sum / max(1, n_img),
            "estimated_bpp": est_bpp_sum / max(1, n_img),
            "actual_bitstream_bpp": act_bpp_sum / max(1, n_img),
            "per_class_iou": per_class,
        },
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "mIoU_rans": result["metrics"]["mIoU"],
        "bpp_est": result["estimated_bpp"],
        "bpp_act": result["actual_bitstream_bpp"],
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
