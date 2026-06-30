"""Per-task losses (CLAUDE.md §5 / instruction §5):
  seg    = CE + Dice + Lovasz-Softmax
  depth  = SiLog + 0.1 * gradient-matching   (valid mask)
  normal = (1 - cos)[valid].mean()
Each loss is a Module taking (pred, gt) -> scalar; preds are raw head outputs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Lovasz-Softmax (Berman et al. 2018, standard impl) ----------
def _lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - inter / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_softmax_flat(probs, labels, ignore):
    if probs.numel() == 0:
        return probs * 0.0
    C = probs.shape[1]
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    if not losses:
        return probs.sum() * 0.0
    return torch.stack(losses).mean()


def lovasz_softmax(logits, labels, ignore_index=255):
    probs = logits.softmax(1)
    B, C, H, W = probs.shape
    probs = probs.permute(0, 2, 3, 1).reshape(-1, C)
    labels = labels.reshape(-1)
    valid = labels != ignore_index
    return _lovasz_softmax_flat(probs[valid], labels[valid], ignore_index)


class SegLoss(nn.Module):
    def __init__(self, num_classes=19, ignore_index=255, w_ce=1.0, w_dice=1.0, w_lovasz=1.0):
        super().__init__()
        self.nc, self.ignore = num_classes, ignore_index
        self.w_ce, self.w_dice, self.w_lovasz = w_ce, w_dice, w_lovasz

    def _dice(self, logits, gt):
        probs = logits.softmax(1)
        valid = (gt != self.ignore)
        gt_c = gt.clone()
        gt_c[~valid] = 0
        oh = F.one_hot(gt_c, self.nc).permute(0, 3, 1, 2).float() * valid.unsqueeze(1)
        probs = probs * valid.unsqueeze(1)
        dims = (0, 2, 3)
        inter = (probs * oh).sum(dims)
        denom = probs.sum(dims) + oh.sum(dims)
        return 1.0 - (2 * inter + 1.0) / (denom + 1.0)

    def forward(self, logits, gt):
        if logits.shape[-2:] != gt.shape[-2:]:
            logits = F.interpolate(logits, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        ce = F.cross_entropy(logits, gt, ignore_index=self.ignore)
        dice = self._dice(logits, gt).mean()
        lov = lovasz_softmax(logits, gt, self.ignore)
        return self.w_ce * ce + self.w_dice * dice + self.w_lovasz * lov


class DepthLoss(nn.Module):
    """SiLog (Eigen 2014) + grad-matching; valid = gt in (min,max)."""
    def __init__(self, min_depth=1e-3, max_depth=80.0, variance_focus=0.85, w_grad=0.1):
        super().__init__()
        self.min, self.max = min_depth, max_depth
        self.vf, self.w_grad = variance_focus, w_grad

    @staticmethod
    def _grad_loss(pred, gt, valid):
        # first-order differences; match where both pixels valid
        dx_p, dx_g = pred[..., :, 1:] - pred[..., :, :-1], gt[..., :, 1:] - gt[..., :, :-1]
        dy_p, dy_g = pred[..., 1:, :] - pred[..., :-1, :], gt[..., 1:, :] - gt[..., :-1, :]
        vx = valid[..., :, 1:] & valid[..., :, :-1]
        vy = valid[..., 1:, :] & valid[..., :-1, :]
        lx = (dx_p - dx_g).abs()[vx].mean() if vx.any() else pred.sum() * 0.0
        ly = (dy_p - dy_g).abs()[vy].mean() if vy.any() else pred.sum() * 0.0
        return lx + ly

    def forward(self, pred, gt):
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        valid = (gt > self.min) & (gt < self.max)
        if not valid.any():
            return pred.sum() * 0.0
        p = pred.clamp(self.min, self.max)
        d = torch.log(p[valid]) - torch.log(gt[valid])
        silog = torch.sqrt((d ** 2).mean() - self.vf * (d.mean() ** 2) + 1e-7) * 10.0
        grad = self._grad_loss(torch.log(p.clamp(self.min)), torch.log(gt.clamp(self.min)), valid)
        return silog + self.w_grad * grad


class NormalLoss(nn.Module):
    """(1 - cos)[valid].mean(); valid = ||gt|| > 1e-3."""
    def forward(self, pred, gt):
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        pred = F.normalize(pred, dim=1)
        gtn = F.normalize(gt, dim=1)
        valid = gt.norm(dim=1) > 1e-3
        cos = (pred * gtn).sum(1)
        if not valid.any():
            return pred.sum() * 0.0
        return (1.0 - cos)[valid].mean()


def build_task_loss(task, cfg=None):
    cfg = dict(cfg or {})
    if task == "seg":
        return SegLoss(**cfg)
    if task == "depth":
        return DepthLoss(**cfg)
    if task == "normal":
        return NormalLoss(**cfg)
    raise KeyError(f"unknown task loss: {task}")
