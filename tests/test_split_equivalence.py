"""§9: with an IDENTITY bottleneck, encoder + tail must reproduce the frozen teacher
exactly — for ALL splits. This is the geometric foundation everything else stands on
(phase2 spec §8.1). Two levels:
  (a) backbone refactor:  forward() == forward_receiver_*(forward_sender_*())
  (b) full student wiring: SplitStudent(identity).preds == teacher.preds
"""
import torch

from src.models.backbones.rtformer import RTFormer
from src.models.bottlenecks import Identity
from src.models.student import SplitStudent

SPLITS = ["early", "mid", "late", "neck"]


def test_backbone_split_refactor_is_lossless():
    torch.manual_seed(0)
    bb = RTFormer().eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        ref = bb(x)
        # early (existing): forward_receiver(forward_sender)
        s4, high = bb.forward_sender(x)
        early = bb.forward_receiver(s4, high)
        # mid: forward_receiver_mid(forward_sender_mid)
        mid = bb.forward_receiver_mid(*bb.forward_sender_mid(x))
        # late: forward_receiver_late(forward_sender_late)
        late = bb.forward_receiver_late(*bb.forward_sender_late(x))
    for name, out in [("early", early), ("mid", mid), ("late", late)]:
        assert torch.allclose(out["s8"], ref["s8"], atol=1e-5), f"{name} s8 mismatch"
    # the scales each split can still produce must also match the teacher
    assert torch.allclose(mid["s16"], ref["s16"], atol=1e-5)
    assert torch.allclose(mid["s32"], ref["s32"], atol=1e-5)
    assert torch.allclose(late["s32"], ref["s32"], atol=1e-5)


def test_identity_student_matches_teacher(teacher, image):
    """Every split, wired through SplitStudent with a no-op bottleneck, reproduces the
    teacher's seg logits (the split routing is lossless)."""
    with torch.no_grad():
        ref = teacher(image)["seg"]
        for split in SPLITS:
            student = SplitStudent(teacher, Identity(), split=split).eval()
            pred = student(image)["preds"]["seg"]
            assert pred.shape == ref.shape, f"{split} shape {pred.shape} != {ref.shape}"
            assert torch.allclose(pred, ref, atol=1e-4), f"{split} split is not lossless"
