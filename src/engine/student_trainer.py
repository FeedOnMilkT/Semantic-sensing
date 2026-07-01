"""Phase-2 student training: a rate-distortion loop for a SplitStudent.

    L = distortion + beta * rate
      distortion = feature_mse(target, recon)   # distill the split feature (Entropic-
                   Student main line); the frozen teacher tail turns `recon` into task
                   predictions, so preserving the feature preserves the task.
      (+ optional task loss on the predictions, weight `task_weight` -> ICM-style arm.)
      rate       = estimated bpp from the bottleneck's entropy model.

Two optimizers (standard CompressAI): the codec optimizer trains g_a/g_s/h_a/h_s +
entropy model; the aux optimizer trains the factorized CDF quantiles on aux_loss().
Only the bottleneck trains — the teacher is frozen inside the student.

`beta` is a single value per run; a bpp-mIoU curve = sweep beta across runs.
"""
import time

import torch
import torch.nn.functional as F

from ..losses.compression import feature_mse, bpp_from_info
from ..losses.task_losses import SegLoss
from ..utils.metrics import build_meter
from ..utils.logging import get_logger
from .trainer import poly_lr, _fmt_eta

log = get_logger("student")


def _distortion(target, recon):
    """Feature-MSE distillation distortion. Sums over branches for the dual
    (mid/late) payload where target/recon are (x_h, x_l) tuples; single tensor
    otherwise. Targets are detached (frozen teacher features)."""
    if isinstance(target, (tuple, list)):
        return sum(feature_mse(t.detach(), r) for t, r in zip(target, recon))
    return feature_mse(target.detach(), recon)


class StudentTrainer:
    def __init__(self, cfg, student, beta, device="cuda"):
        self.cfg = cfg
        self.student = student.to(device)
        self.beta = beta
        self.device = device
        opt = cfg.get("optimizer", {})
        self.codec_opt = torch.optim.Adam(student.codec_parameters(),
                                          lr=opt.get("lr", 1e-3),
                                          weight_decay=opt.get("weight_decay", 0.0))
        self.aux_opt = torch.optim.Adam(student.aux_parameters(), lr=cfg.get("aux_lr", 1e-3))
        self.base_lr = self.codec_opt.param_groups[0]["lr"]
        self.sched = cfg.get("scheduler", "poly")
        self.warmup = cfg.get("warmup", 0)
        self.grad_clip = cfg.get("grad_clip", 0.0)
        self.task_weight = cfg.get("task_weight", 0.0)
        self.ignore_index = cfg.get("ignore_index", 255)

    def _set_lr(self, step, total):
        lr = poly_lr(step, total, self.base_lr, warmup=self.warmup) if self.sched == "poly" else self.base_lr
        for g in self.codec_opt.param_groups:
            g["lr"] = lr
        return lr

    def fit(self, loader, epochs=1, max_steps=None, log_every=20):
        spe = len(loader)
        total = max_steps or epochs * spe
        step, t0 = 0, time.time()
        self.student.train()
        for ep in range(epochs):
            for batch in loader:
                if max_steps and step >= max_steps:
                    break
                lr = self._set_lr(step, total)
                x = batch["image"].to(self.device, non_blocking=True)
                B, _, H, W = x.shape
                num_pixels = B * H * W

                out = self.student(x)
                distortion = _distortion(out["target"], out["recon"])
                if "recon_aux" in out:                       # early + compress_s4: distill s4 too
                    distortion = distortion + _distortion(out["target_aux"], out["recon_aux"])
                bpp = bpp_from_info(out["info"], num_pixels)
                loss = distortion + self.beta * bpp
                if self.task_weight and "seg" in batch.get("targets", {}):
                    gt = batch["targets"]["seg"].to(self.device)
                    loss = loss + self.task_weight * F.cross_entropy(
                        out["preds"]["seg"], gt, ignore_index=self.ignore_index)

                self.codec_opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.student.codec_parameters(), self.grad_clip)
                self.codec_opt.step()

                aux = self.student.aux_loss()           # train the CDF quantiles separately
                self.aux_opt.zero_grad(set_to_none=True)
                aux.backward()
                self.aux_opt.step()

                if step % log_every == 0:
                    eta = (time.time() - t0) / max(1, step + 1) * (total - step - 1)
                    log.info(f"ep{ep} [{step + 1}/{total}] lr={lr:.2e} beta={self.beta:g} "
                             f"loss={float(loss):.4f} dist={float(distortion):.4f} "
                             f"bpp={float(bpp):.4f} aux={float(aux):.1f} eta={_fmt_eta(eta)}")
                step += 1
            if max_steps and step >= max_steps:
                break
        return self.student


class Stage2Trainer:
    """Stage 2 (spec §5.2): freeze the sender encoder + bottleneck (deployment side
    fixed, entropy model locked), fine-tune the RECEIVER TAIL with the task loss
    L = CE + Dice (+ Lovász switch). The student runs in eval() mode so the frozen
    bottleneck quantises like deployment (rounded latents) and tail BN stats stay
    fixed — only the tail conv/linear weights update.

    ignore_index / num_classes MUST match the teacher's ADE20K convention (spec §5.2)
    or mIoU is meaningless; both come from cfg."""

    def __init__(self, cfg, student, device="cuda"):
        self.cfg = cfg
        student.set_stage2_trainable()
        self.student = student.to(device)
        self.student.eval()                          # frozen BN; quantised (deploy-like) recon
        self.device = device
        nc = cfg.get("num_classes", 150)
        self.ignore_index = cfg.get("ignore_index", 255)
        sw = cfg.get("seg_loss", {})
        self.loss_fn = SegLoss(num_classes=nc, ignore_index=self.ignore_index,
                               w_ce=sw.get("w_ce", 1.0), w_dice=sw.get("w_dice", 1.0),
                               w_lovasz=sw.get("w_lovasz", 0.0)).to(device)   # Lovász off by default
        opt = cfg.get("optimizer", {})
        self.opt = torch.optim.AdamW(student.receiver_parameters(),
                                     lr=opt.get("lr", 1e-4), weight_decay=opt.get("weight_decay", 0.01))
        self.base_lr = self.opt.param_groups[0]["lr"]
        self.sched = cfg.get("scheduler", "poly")
        self.warmup = cfg.get("warmup", 0)
        self.grad_clip = cfg.get("grad_clip", 1.0)

    def _set_lr(self, step, total):
        lr = poly_lr(step, total, self.base_lr, warmup=self.warmup) if self.sched == "poly" else self.base_lr
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr

    def fit(self, loader, epochs=1, max_steps=None, log_every=20):
        spe = len(loader)
        total = max_steps or epochs * spe
        step, t0 = 0, time.time()
        self.student.eval()                          # keep eval throughout Stage 2
        for ep in range(epochs):
            for batch in loader:
                if max_steps and step >= max_steps:
                    break
                lr = self._set_lr(step, total)
                x = batch["image"].to(self.device, non_blocking=True)
                gt = batch["targets"]["seg"].to(self.device)
                out = self.student(x)                # frozen bottleneck -> trainable tail -> preds
                loss = self.loss_fn(out["preds"]["seg"], gt)

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.student.receiver_parameters(), self.grad_clip)
                self.opt.step()

                if step % log_every == 0:
                    eta = (time.time() - t0) / max(1, step + 1) * (total - step - 1)
                    log.info(f"[stage2] ep{ep} [{step + 1}/{total}] lr={lr:.2e} "
                             f"loss={float(loss):.4f} eta={_fmt_eta(eta)}")
                step += 1
            if max_steps and step >= max_steps:
                break
        return self.student


@torch.no_grad()
def evaluate_student(student, loader, num_classes, device="cuda", ignore_index=255):
    """One pass over val -> the bpp-mIoU point: seg metrics (mIoU/pixel_acc) from the
    frozen tail on reconstructed features, plus mean estimated bpp."""
    student.eval()
    meter = build_meter("seg", {"num_classes": num_classes, "ignore_index": ignore_index})
    bpp_sum, n = 0.0, 0
    for batch in loader:
        x = batch["image"].to(device)
        B, _, H, W = x.shape
        out = student(x)
        meter.update(out["preds"]["seg"].float().cpu(), batch["targets"]["seg"])
        bpp_sum += bpp_from_info(out["info"], B * H * W).item() * B
        n += B
    metrics = meter.compute()
    metrics["bpp"] = bpp_sum / max(1, n)
    return metrics
