#!/usr/bin/env python
"""compress-then-analyze baseline: send the IMAGE via a standard codec (JPEG/WebP),
run the (frozen) teacher at the receiver. Sweeps codec quality -> a bpp-mIoU curve,
the reference our semantic-feature compression must beat (same mIoU at lower bpp).

bpp = real codec file bytes * 8 / image pixels -- the SAME units as the student's bpp
(bits per image pixel), so the curves overlay directly.

  CUDA_VISIBLE_DEVICES=0 python tools/baseline_image_codec.py configs/ade20k_rtformer.yaml \
      --teacher-ckpt /data/sidney/checkpoints/semantic-sensing/ade20k/ade20k_rtformer_bs32_lr4e4.pth \
      --codec jpeg --qualities 5 10 20 35 50 75
"""
import argparse
import io

from _common import pick_device
import numpy as np
from PIL import Image
import torch

from src.utils import load_config, get_logger
from src.models import build_model
from src.datasets import build_dataset, collate
from src.datasets.ade20k import _MEAN, _STD
from src.utils.metrics import build_meter
from torch.utils.data import DataLoader, Subset

log = get_logger("baseline")


def denorm_to_uint8(img):
    """(3,H,W) normalized tensor -> (H,W,3) uint8 RGB (exact inverse of the dataset)."""
    x = (img.cpu() * _STD + _MEAN).clamp(0, 1) * 255.0
    return x.round().byte().permute(1, 2, 0).numpy()


def recode(uint8_hwc, codec, quality):
    """Encode -> bytes (measured), decode back -> (H,W,3) uint8. Returns (decoded, nbytes)."""
    buf = io.BytesIO()
    fmt = {"jpeg": "JPEG", "webp": "WEBP"}[codec]
    Image.fromarray(uint8_hwc).save(buf, format=fmt, quality=int(quality))
    data = buf.getvalue()
    dec = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), np.uint8)
    return dec, len(data)


def to_model_input(uint8_hwc, device):
    t = torch.from_numpy(uint8_hwc.transpose(2, 0, 1).copy()).float() / 255.0
    return ((t - _MEAN) / _STD).to(device)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="teacher config (e.g. configs/ade20k_rtformer.yaml)")
    ap.add_argument("--teacher-ckpt", required=True)
    ap.add_argument("--codec", choices=["jpeg", "webp"], default="jpeg")
    ap.add_argument("--qualities", type=int, nargs="+", default=[5, 10, 20, 35, 50, 75])
    ap.add_argument("--val-limit", type=int, default=None)
    args = ap.parse_args()

    device = pick_device()
    cfg = load_config(args.config)
    cfg.model.backbone.args.pretrained = None
    nc = int(cfg.get("num_classes", 150))
    ignore = int(cfg.get("ignore_index", 255))

    teacher = build_model(cfg.model).to(device).eval()
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location="cpu")["model"])
    log.info(f"teacher loaded; codec={args.codec} qualities={args.qualities} device={device}")

    ds = build_dataset(cfg.data.val)
    if args.val_limit:
        ds = Subset(ds, range(args.val_limit))
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate)

    meters = {q: build_meter("seg", {"num_classes": nc, "ignore_index": ignore}) for q in args.qualities}
    bytes_q = {q: 0 for q in args.qualities}
    npix = 0
    for batch in loader:
        imgs, seg = batch["image"], batch["targets"]["seg"]
        uint8s = [denorm_to_uint8(imgs[b]) for b in range(imgs.shape[0])]
        npix += sum(u.shape[0] * u.shape[1] for u in uint8s)
        for q in args.qualities:
            decoded, xin = [], []
            for u in uint8s:
                dec, nb = recode(u, args.codec, q)
                bytes_q[q] += nb
                xin.append(to_model_input(dec, device))
            x = torch.stack(xin)
            logits = teacher(x, tasks=["seg"])["seg"].float().cpu()
            meters[q].update(logits, seg)

    print(f"\n== compress-then-analyze baseline ({args.codec}) ==")
    print(f"{'quality':>8} {'bpp':>8} {'mIoU':>8} {'pixel_acc':>10}")
    for q in args.qualities:
        bpp = bytes_q[q] * 8 / npix
        m = meters[q].compute()
        print(f"{q:>8} {bpp:>8.4f} {m['mIoU']:>8.4f} {m['pixel_acc']:>10.4f}")
    print("\n(reference: student late b=0.001 -> bpp 1.054, mIoU 0.3751 | teacher uncompressed 0.3823; raw image = 24 bpp)")


if __name__ == "__main__":
    main()
