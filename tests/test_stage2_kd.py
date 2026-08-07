"""Stage-2 teacher logit distillation (Phase-2.5 A-arm, fused workflow §9.1):
  off-switch : kd.weight=0 (or no kd block) is bit-identical to the plain CE+Dice path.
  on-path    : kd.weight>0 adds a finite, non-negative forward-KL term (T² scale)
               that backpropagates into the receiver tail.
  masking    : ignore_index pixels are excluded; an all-ignore batch yields a
               graph-connected zero, never NaN.
  alignment  : logits at a different resolution than gt are aligned the same way
               SegLoss aligns for CE (bilinear to gt size).
"""
import copy

import pytest
import torch
from omegaconf import OmegaConf

from src.engine.student_trainer import Stage2Trainer, kd_loss
from src.models.bottlenecks import build_bottleneck
from src.models.student import SplitStudent, split_in_channels

IGNORE = 255


def _student(teacher):
    cfg = OmegaConf.create({"name": "rtformer_fused",
                            "args": {"latent_channels": 64, "hyper_channels": 64,
                                     "hidden": 64, "num_down": 1}})
    torch.manual_seed(0)
    bn = build_bottleneck(cfg, in_channels=split_in_channels(teacher, "mid"))
    return SplitStudent(teacher, bn, split="mid")


def _trainer(teacher, num_classes, kd=None):
    cfg = OmegaConf.create({"num_classes": num_classes, "ignore_index": IGNORE,
                            "optimizer": {"lr": 1e-4}})
    if kd is not None:
        cfg.kd = kd
    return Stage2Trainer(cfg, _student(teacher), device="cpu")


def _gt(num_classes, hw=(64, 64)):
    torch.manual_seed(2)
    gt = torch.randint(0, num_classes, (1, *hw))
    gt[:, :8, :8] = IGNORE                       # some ignored pixels in the mask path
    return gt


def test_kd_weight_zero_matches_plain_path(teacher, image, num_classes):
    gt = _gt(num_classes)
    t_plain = _trainer(teacher, num_classes)                                # no kd block
    t_zero = _trainer(teacher, num_classes, kd={"weight": 0.0, "temperature": 4.0})
    loss_plain = t_plain._loss(image, gt)
    loss_zero = t_zero._loss(image, gt)
    # both equal each other and the bare SegLoss on the same forward
    assert torch.equal(loss_plain, loss_zero)
    ref = t_plain.loss_fn(t_plain.student(image)["preds"]["seg"], gt)
    assert torch.equal(loss_plain, ref)


def test_kd_positive_finite_and_backprops(teacher, image, num_classes):
    gt = _gt(num_classes)
    t_kd = _trainer(teacher, num_classes, kd={"weight": 0.5, "temperature": 4.0})
    loss = t_kd._loss(image, gt)
    assert torch.isfinite(loss)
    # forward KL is non-negative, so the KD path can only add loss
    plain = t_kd.loss_fn(t_kd.student(image)["preds"]["seg"], gt)
    assert float(loss) >= float(plain)
    loss.backward()
    grads = [p.grad for p in t_kd.student.receiver_parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    # frozen send side stays untouched
    assert all(p.grad is None for p in t_kd.student.bottleneck.parameters())


def test_kd_loss_all_ignore_is_zero_not_nan(num_classes):
    torch.manual_seed(3)
    s = torch.randn(2, num_classes, 16, 16, requires_grad=True)
    t = torch.randn(2, num_classes, 16, 16)
    gt = torch.full((2, 16, 16), IGNORE, dtype=torch.long)
    kd = kd_loss(s, t, gt, temperature=4.0, ignore_index=IGNORE)
    assert torch.isfinite(kd) and float(kd) == 0.0
    assert kd.requires_grad                      # graph-connected zero, backward-safe
    kd.backward()
    assert torch.isfinite(s.grad).all()


def test_kd_loss_masks_ignore_and_aligns_resolution(num_classes):
    torch.manual_seed(4)
    s = torch.randn(1, num_classes, 8, 8)        # coarser than gt -> bilinear align
    t = torch.randn(1, num_classes, 8, 8)
    gt = torch.randint(0, num_classes, (1, 16, 16))
    full = kd_loss(s, t, gt, temperature=2.0, ignore_index=IGNORE)
    gt_ig = gt.clone()
    gt_ig[:, 8:, :] = IGNORE                     # masking changes the valid-pixel mean
    half = kd_loss(s, t, gt_ig, temperature=2.0, ignore_index=IGNORE)
    assert torch.isfinite(full) and torch.isfinite(half)
    assert not torch.equal(full, half)
    # teacher == student -> KL is exactly zero regardless of mask
    same = kd_loss(s, s.clone(), gt, temperature=2.0, ignore_index=IGNORE)
    assert pytest.approx(0.0, abs=1e-6) == float(same)


def test_reference_kd_is_frozen_and_uses_candidate_reconstruction(
        teacher, image, num_classes):
    candidate = _student(teacher)
    reference = _student(copy.deepcopy(teacher))
    seen = {}
    original_receiver = reference._receiver

    def recording_receiver(feat, ctx):
        seen["feat"] = feat
        return original_receiver(feat, ctx)

    reference._receiver = recording_receiver
    cfg = OmegaConf.create({
        "num_classes": num_classes,
        "ignore_index": IGNORE,
        "optimizer": {"lr": 1e-4},
        "reference_kd": {"weight": 0.05, "temperature": 4.0},
    })
    trainer = Stage2Trainer(cfg, candidate, device="cpu", reference_student=reference)
    loss = trainer._loss(image, _gt(num_classes))
    assert torch.isfinite(loss)
    assert isinstance(seen["feat"], tuple) and len(seen["feat"]) == 2
    loss.backward()
    assert any(p.grad is not None for p in candidate.receiver_parameters())
    assert all(p.grad is None and not p.requires_grad for p in reference.parameters())
