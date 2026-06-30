"""Phase-2 supervised-compression loss: distortion + beta * rate.

  distortion = feature_mse : ||target_feat - f_hat||^2   (distillation / reconstruction —
               align the reconstructed split feature to the frozen teacher's feature)
             = task_loss   : the task loss on the student's predictions (ICM-style arm)
  rate       = EntropyBottleneck estimated bpp (learned entropy model; emits real bits)

The two distortion arms differ ONLY in the distortion term — rate is identical — so a
single switch draws the "distillation vs task-driven" rate-distortion curves. This is
the same machinery the deployment codec uses, kept dataset/task agnostic.
"""
import math

import torch
import torch.nn.functional as F


def feature_mse(target_feat, f_hat):
    return F.mse_loss(f_hat, target_feat)


def bpp_from_info(info, num_pixels):
    """Estimated bits-per-image-pixel from a bottleneck's `info` dict. Handles both
    a single likelihood tensor (factorized bottleneck) and a dict of them (hyperprior
    -> {"y", "z"}), so the loss is agnostic to the entropy model."""
    liks = info["likelihoods"]
    tensors = list(liks.values()) if isinstance(liks, dict) else [liks]
    bits = sum(torch.log(t).sum() / (-math.log(2.0)) for t in tensors)
    return bits / num_pixels


def rate_distortion_loss(f_hat, target_feat, info, beta, num_pixels):
    """L = feature_mse(target_feat, f_hat) + beta * bpp.

    Returns (total, parts) where parts holds detached scalars for logging."""
    distortion = feature_mse(target_feat, f_hat)
    bpp = bpp_from_info(info, num_pixels)
    total = distortion + beta * bpp
    parts = {"distortion": distortion.detach(), "bpp": bpp.detach(), "loss": total.detach()}
    return total, parts
