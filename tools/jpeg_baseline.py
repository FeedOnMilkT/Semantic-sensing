#!/usr/bin/env python
"""External JPEG baseline (phase2 spec §6): encode the original image at a sweep of
quality factors -> decode -> run the FROZEN teacher -> (bpp, mIoU). The only external
arm the learned curves are compared against; the mid main line should sit ABOVE this at
equal bpp (spec §9). bpp = JPEG file bits / (H*W image pixels) -- SAME normalisation as
the learned arms, so the curves are comparable.

  CUDA_VISIBLE_DEVICES=5 python tools/jpeg_baseline.py configs/baseline_jpeg.yaml \
    --teacher-ckpt .../ade20k_rtformer_paperhead.pth --out .../_orch/jpeg.csv
"""
import argparse
import io

from _common import pick_device
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from src import _compat  # noqa: F401
from src.utils import load_config, get_logger
from src.models import build_model
from src.datasets import build_dataset, collate
from src.utils.metrics import build_meter

log = get_logger("jpeg")
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _denorm_uint8(img):                       # img (3,H,W) normalised -> HWC uint8
    x = (img.cpu() * _STD + _MEAN).clamp(0, 1) * 255.0
    return x.round().byte().permute(1, 2, 0).numpy()


def _jpeg(uint8_hwc, q):                       # -> (decoded HWC uint8, n_bytes)
    buf = io.BytesIO()
    Image.fromarray(uint8_hwc).save(buf, format="JPEG", quality=int(q))
    n = buf.tell()
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), np.uint8), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--teacher-ckpt", required=True)
    ap.add_argument("--val-limit", type=int, default=2000)
    ap.add_argument("--out", default=None, help="CSV to append [JPEG] points to")
    args = ap.parse_args()

    device = pick_device()
    cfg = load_config(args.config)
    nc = int(cfg.get("num_classes", 150)); ign = int(cfg.get("ignore_index", 255))
    qualities = list(cfg.get("jpeg", {}).get("qualities", [10, 20, 30, 40, 50, 70, 90]))

    cfg.model.backbone.args.pretrained = None
    teacher = build_model(cfg.model).to(device).eval()
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location="cpu")["model"])

    full = build_dataset(cfg.data.val)
    n = min(args.val_limit, len(full))
    loader = DataLoader(Subset(full, range(n)), batch_size=1, shuffle=False, collate_fn=collate)

    rows = []
    for q in qualities:
        meter = build_meter("seg", {"num_classes": nc, "ignore_index": ign})
        bits = px = 0
        with torch.no_grad():
            for b in loader:
                dec, nb = _jpeg(_denorm_uint8(b["image"][0]), q)
                H, W = dec.shape[:2]
                bits += nb * 8; px += H * W
                t = torch.from_numpy(dec.transpose(2, 0, 1)).float() / 255.0
                t = ((t - _MEAN) / _STD).unsqueeze(0).to(device)
                meter.update(teacher(t)["seg"].float().cpu(), b["targets"]["seg"])
        m = meter.compute(); bpp = bits / px
        log.info(f"[JPEG-POINT] Q={q} bpp={bpp:.4f} mIoU={m['mIoU']:.4f} pixel_acc={m['pixel_acc']:.4f}")
        rows.append((q, bpp, m["mIoU"], m["pixel_acc"]))

    if args.out:
        with open(args.out, "w") as f:
            f.write("method,q,bpp,mIoU,pixel_acc\n")
            for q, bpp, miou, pa in rows:
                f.write(f"jpeg,{q},{bpp:.5f},{miou:.5f},{pa:.5f}\n")
        log.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
