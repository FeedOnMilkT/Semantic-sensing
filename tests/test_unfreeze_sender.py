"""Stage-1 unfreeze-sender (workflow §9.5): sender + bottleneck train; receiver frozen;
distillation targets anchored to frozen init-time _sender_ref."""
import copy

import torch
from omegaconf import OmegaConf

from src.engine.student_trainer import StudentTrainer, _distortion
from src.models.bottlenecks import build_bottleneck
from src.models.student import SplitStudent, build_student, split_in_channels


def _fused_student(teacher, unfreeze_sender=False):
    cfg = OmegaConf.create({"name": "rtformer_fused",
                            "args": {"latent_channels": 64, "hyper_channels": 64,
                                     "hidden": 64, "num_down": 1}})
    bn = build_bottleneck(cfg, in_channels=split_in_channels(teacher, "mid"))
    s = SplitStudent(teacher, bn, split="mid", unfreeze_sender=unfreeze_sender)
    s.set_stage1_trainable(unfreeze_sender)
    return s


def test_unfreeze_grad_pattern(teacher, image):
    s = _fused_student(teacher, unfreeze_sender=True)
    s.train()
    x = image.repeat(2, 1, 1, 1)                   # BN in fused bottleneck needs batch>1
    out = s(x)
    loss = _distortion(out["target"], out["recon"])
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in s.sender_parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in s.codec_parameters())
    assert all(p.grad is None for p in s.receiver_parameters())


def test_target_from_frozen_ref_not_live_sender(teacher, image):
    s = _fused_student(teacher, unfreeze_sender=True)
    s.train()
    with torch.no_grad():
        target_before = s._teacher_split_target(image)
    # Perturb live sender; frozen ref (and thus target) must stay fixed.
    with torch.no_grad():
        for p in s.sender_parameters():
            p.add_(0.5)
    with torch.no_grad():
        target_after = s._teacher_split_target(image)
    live, _ = s._sender(image)
    assert isinstance(target_before, tuple) and isinstance(live, tuple)
    assert torch.allclose(target_before[0], target_after[0])
    assert torch.allclose(target_before[1], target_after[1])
    assert not torch.allclose(target_before[0], live[0].detach())


def test_build_student_unfreeze_wires_trainer(teacher):
    cfg = OmegaConf.create({
        "student": {"split": "mid", "unfreeze_sender": True, "sender_lr": 1e-4},
        "bottleneck": {"name": "rtformer_fused",
                       "args": {"latent_channels": 64, "hyper_channels": 64,
                                "hidden": 64, "num_down": 1}},
    })
    s = build_student(cfg, copy.deepcopy(teacher))
    tcfg = OmegaConf.create({"optimizer": {"lr": 1e-3}, "unfreeze_sender": True,
                             "sender_lr": 1e-4, "grad_clip": 1.0})
    tr = StudentTrainer(tcfg, s, beta=0.01, device="cpu")
    assert len(tr.codec_opt.param_groups) == 2
    assert tr.codec_opt.param_groups[0]["lr"] == 1e-3
    assert tr.codec_opt.param_groups[1]["lr"] == 1e-4
