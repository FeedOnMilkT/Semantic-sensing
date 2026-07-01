#!/usr/bin/env python
"""Aggregate the phase2 split-computing results into the bpp-mIoU report (spec §7).

Re-evaluates each saved student ckpt from disk (robust: the ckpt is the ground truth,
independent of any training log) and merges the JPEG baseline + teacher ceiling into a
table + CSV + plot. Covers the 3 splits (early/mid/late) x betas x {Stage1, Stage2}.

  CUDA_VISIBLE_DEVICES=0 python tools/eval_rd_curve.py \
    --teacher-ckpt .../ade20k_rtformer_paperhead.pth \
    --students-dir .../students --jpeg-csv .../_orch/jpeg.csv \
    --ceiling-file .../_orch/teacher_ceiling_fullval.txt --out .../_orch/report
"""
import argparse
import glob
import os

from _common import pick_device
import torch
from torch.utils.data import DataLoader, Subset

from src import _compat  # noqa: F401
from src.utils import load_config, get_logger
from src.models import build_model
from src.models.student import build_student
from src.datasets import build_dataset, collate
from src.engine import evaluate_student

log = get_logger("rd_curve")


_TEACHER = {"obj": None}                          # built once, shared across all ckpts


def _eval_ckpt(path, teacher_ckpt, val_limit, device):
    sd = torch.load(path, map_location="cpu")
    cfg = sd["cfg"]
    if _TEACHER["obj"] is None:                   # build the frozen teacher ONCE (all ckpts share it)
        cfg.model.backbone.args.pretrained = None
        t = build_model(cfg.model).to(device).eval()
        t.load_state_dict(torch.load(teacher_ckpt, map_location="cpu")["model"])
        _TEACHER["obj"] = t
    student = build_student(cfg, _TEACHER["obj"]).to(device)
    student.load_state_dict(sd["student"])
    full = build_dataset(cfg.data.val)
    n = min(val_limit, len(full))
    loader = DataLoader(Subset(full, range(n)), batch_size=4, shuffle=False, collate_fn=collate)
    m = evaluate_student(student, loader, int(cfg.get("num_classes", 150)), device=device,
                         ignore_index=int(cfg.get("ignore_index", 255)))
    return cfg.student.split, float(sd.get("beta", 0)), m["bpp"], m["mIoU"], m["pixel_acc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-ckpt", required=True)
    ap.add_argument("--students-dir", required=True)
    ap.add_argument("--jpeg-csv", default=None)
    ap.add_argument("--ceiling-file", default=None)
    ap.add_argument("--val-limit", type=int, default=2000)
    ap.add_argument("--stage", default="s2", choices=["s1", "s2", "both"])
    ap.add_argument("--out", default="report")
    args = ap.parse_args()
    device = pick_device()

    stages = ["s1", "s2"] if args.stage == "both" else [args.stage]
    rows = []
    for split in ["early", "mid", "late"]:
        for st in stages:
            for ck in sorted(glob.glob(os.path.join(args.students_dir, f"{split}_b*_{st}.pth"))):
                try:
                    sp, beta, bpp, miou, pa = _eval_ckpt(ck, args.teacher_ckpt, args.val_limit, device)
                    rows.append((sp, st, beta, bpp, miou, pa))
                    log.info(f"{sp:5s} {st} beta={beta:<6g} bpp={bpp:.4f} mIoU={miou:.4f} pacc={pa:.4f}")
                except Exception as e:
                    log.info(f"SKIP {ck}: {e}")

    ceiling = None
    if args.ceiling_file and os.path.exists(args.ceiling_file):
        txt = open(args.ceiling_file).read()
        for tok in txt.split():
            if tok.startswith("mIoU="):
                ceiling = float(tok.split("=")[1])

    jpeg = []
    if args.jpeg_csv and os.path.exists(args.jpeg_csv):
        for ln in open(args.jpeg_csv).read().splitlines()[1:]:
            p = ln.split(",")
            jpeg.append((float(p[2]), float(p[3])))     # bpp, mIoU

    # CSV
    with open(args.out + ".csv", "w") as f:
        f.write("split,stage,beta,bpp,mIoU,pixel_acc\n")
        for sp, st, beta, bpp, miou, pa in rows:
            f.write(f"{sp},{st},{beta:g},{bpp:.5f},{miou:.5f},{pa:.5f}\n")
        if ceiling is not None:
            f.write(f"teacher,-,-,-,{ceiling:.5f},-\n")
        for bpp, miou in jpeg:
            f.write(f"jpeg,-,-,{bpp:.5f},{miou:.5f},-\n")
    log.info(f"wrote {args.out}.csv  (teacher_ceiling={ceiling})")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for split, mk in [("early", "s--"), ("mid", "o-"), ("late", "^:")]:
            pts = sorted([(b, mi) for sp, st, be, b, mi, pa in rows if sp == split and st == "s2"])
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], mk, label=f"{split} (Stage2)")
        if jpeg:
            jp = sorted(jpeg)
            ax.plot([p[0] for p in jp], [p[1] for p in jp], "x-", color="gray", label="JPEG")
        if ceiling is not None:
            ax.axhline(ceiling, ls="--", color="k", alpha=0.6, label=f"teacher {ceiling:.3f}")
        ax.set_xlabel("bpp (bits / image pixel)"); ax.set_ylabel("mIoU")
        ax.set_title("ADE20K split-computing: bpp vs mIoU"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(args.out + ".png", dpi=130)
        log.info(f"wrote {args.out}.png")
    except Exception as e:
        log.info(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
