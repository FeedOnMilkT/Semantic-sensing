"""Streaming metric accumulators, one per task. Each .update(pred, gt) on raw
model outputs (logits / depth / normal) and .compute() -> dict of scalars.

- seg:    mIoU (+ pixel acc) via a confusion matrix, ignoring `ignore_index`.
- depth:  RMSE / AbsRel / delta<1.25^{1,2,3} on a valid mask.
- normal: mean/median angular error (deg) + % within 11.25/22.5/30 on a valid mask.
"""
import torch


class SegMeter:
    def __init__(self, num_classes, ignore_index=255, exclude_classes=()):
        self.n = num_classes
        self.ignore = ignore_index
        self.exclude = tuple(int(c) for c in exclude_classes)
        if any(c < 0 or c >= num_classes for c in self.exclude):
            raise ValueError("exclude_classes must be valid class ids")
        self.conf = torch.zeros(num_classes, num_classes, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits, gt):
        # logits: (B,K,H,W) at any resolution; gt: (B,H,W) long
        if logits.shape[-2:] != gt.shape[-2:]:
            logits = torch.nn.functional.interpolate(
                logits, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(1)
        valid = gt != self.ignore
        p, g = pred[valid].view(-1), gt[valid].view(-1)
        idx = g * self.n + p
        binc = torch.bincount(idx.cpu(), minlength=self.n ** 2).reshape(self.n, self.n)
        self.conf += binc.double()

    def compute(self):
        conf = self.conf
        tp = conf.diag()
        union = conf.sum(0) + conf.sum(1) - tp
        iou = tp / union.clamp(min=1)
        include = torch.ones(self.n, dtype=torch.bool)
        if self.exclude:
            include[list(self.exclude)] = False
        present = (union > 0) & include
        return {
            "mIoU": float(iou[present].mean()) if present.any() else 0.0,
            "pixel_acc": float(tp.sum() / conf.sum().clamp(min=1)),
        }


class DepthMeter:
    def __init__(self, min_depth=1e-3, max_depth=80.0):
        self.min, self.max = min_depth, max_depth
        self.sq_err = 0.0
        self.abs_rel = 0.0
        self.d1 = self.d2 = self.d3 = 0.0
        self.count = 0.0

    @torch.no_grad()
    def update(self, pred, gt):
        # pred,gt: (B,1,H,W) in meters
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = torch.nn.functional.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        valid = (gt > self.min) & (gt < self.max)
        p, g = pred[valid].clamp(self.min, self.max), gt[valid]
        if p.numel() == 0:
            return
        self.sq_err += float(((p - g) ** 2).sum())
        self.abs_rel += float((torch.abs(p - g) / g).sum())
        ratio = torch.maximum(p / g, g / p)
        self.d1 += float((ratio < 1.25).sum())
        self.d2 += float((ratio < 1.25 ** 2).sum())
        self.d3 += float((ratio < 1.25 ** 3).sum())
        self.count += p.numel()

    def compute(self):
        c = max(self.count, 1.0)
        return {
            "RMSE": (self.sq_err / c) ** 0.5,
            "AbsRel": self.abs_rel / c,
            "delta1": self.d1 / c,
            "delta2": self.d2 / c,
            "delta3": self.d3 / c,
        }


class NormalMeter:
    def __init__(self):
        self.errs = []

    @torch.no_grad()
    def update(self, pred, gt):
        # pred,gt: (B,3,H,W); pred assumed L2-normalized; gt valid where ||gt||>1e-3
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = torch.nn.functional.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        gt = torch.nn.functional.normalize(gt, dim=1)
        valid = gt.norm(dim=1) > 1e-3
        cos = (pred * gt).sum(1).clamp(-1, 1)
        ang = torch.rad2deg(torch.arccos(cos))[valid]
        if ang.numel():
            self.errs.append(ang.cpu())

    def compute(self):
        if not self.errs:
            return {"mean_ang": 0.0, "median_ang": 0.0, "within_11.25": 0.0, "within_22.5": 0.0, "within_30": 0.0}
        e = torch.cat(self.errs)
        return {
            "mean_ang": float(e.mean()),
            "median_ang": float(e.median()),
            "within_11.25": float((e < 11.25).float().mean()),
            "within_22.5": float((e < 22.5).float().mean()),
            "within_30": float((e < 30).float().mean()),
        }


def build_meter(task, cfg=None):
    cfg = cfg or {}
    if task == "seg":
        return SegMeter(num_classes=cfg.get("num_classes", 19),
                        ignore_index=cfg.get("ignore_index", 255),
                        exclude_classes=cfg.get("exclude_classes", ()))
    if task == "depth":
        return DepthMeter(min_depth=cfg.get("min_depth", 1e-3), max_depth=cfg.get("max_depth", 80.0))
    if task == "normal":
        return NormalMeter()
    raise KeyError(f"unknown task for meter: {task}")
