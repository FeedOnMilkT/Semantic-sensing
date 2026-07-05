"""Fused single-latent bottleneck (rtformer_fused, fused-latent workflow §4):
  contract    : forward((x_h,x_l)) -> ((x̂_h,x̂_l), info), likelihoods = {y,z} exactly
                (ONE joint latent, vs rtformer_distill's four streams); size guards
                hold on odd (non-/32-divisible) inputs; bpp_from_info consumes info.
  factory     : build_bottleneck("rtformer_fused") needs a (c_high, c_low) tuple.
  bitstream   : ONE bytes container (magic RTFF); compress -> decompress reproduces
                the eval-mode forward reconstruction; estimated bpp tracks actual.
  gate variant: use_attn=True (Cheng-style conv sigmoid gate) keeps the contract.
  integration : SplitStudent(mid) consumes the fused bottleneck unchanged.
"""
import math

import pytest
import torch
from omegaconf import OmegaConf

from src.losses.compression import bpp_from_info
from src.models.bottlenecks import build_bottleneck
from src.models.bottlenecks.rtformer_fused import FusedLatentBottleneck
from src.models.student import SplitStudent, split_in_channels

C_HIGH, C_LOW = 128, 256


def _bn(**kw):
    torch.manual_seed(0)
    args = dict(latent_channels=64, hyper_channels=64, hidden=64,
                num_down=1, hyper_down=1, act="convbn")
    args.update(kw)
    return FusedLatentBottleneck(C_HIGH, C_LOW, **args).eval()


def _feats(hw_h=(16, 16), hw_l=(8, 8)):
    torch.manual_seed(0)
    return torch.randn(1, C_HIGH, *hw_h), torch.randn(1, C_LOW, *hw_l)


@pytest.mark.parametrize("hw_h,hw_l", [((16, 16), (8, 8)),      # /32-divisible
                                       ((17, 19), (9, 10))])    # odd -> size guards
def test_fused_contract_shapes_and_likelihoods(hw_h, hw_l):
    bn = _bn()
    x_h, x_l = _feats(hw_h, hw_l)
    (f_h, f_l), info = bn((x_h, x_l))
    assert f_h.shape == x_h.shape and f_l.shape == x_l.shape
    assert set(info["likelihoods"]) == {"y", "z"}, "fused = ONE y + ONE z"
    bpp = bpp_from_info(info, num_pixels=1 * 64 * 64)
    assert torch.isfinite(bpp) and bpp.item() > 0
    # gradients reach the codec params
    loss = f_h.mean() + f_l.mean()
    loss.backward()
    assert any(p.grad is not None for p in bn.codec_parameters())


def test_fused_factory():
    cfg = OmegaConf.create({"name": "rtformer_fused",
                            "args": {"latent_channels": 64, "hyper_channels": 64,
                                     "hidden": 64, "num_down": 1}})
    bn = build_bottleneck(cfg, in_channels=(C_HIGH, C_LOW))
    assert bn.in_channels == (C_HIGH, C_LOW)
    with pytest.raises(ValueError):
        build_bottleneck(cfg, in_channels=C_HIGH)        # int -> dual payload required


def test_fused_roundtrip_matches_eval_forward(tmp_path):
    bn = _bn()
    bn.update(force=True)
    feats = _feats()
    data = bn.compress(feats)
    assert isinstance(data, (bytes, bytearray)) and data[:4] == b"RTFF"
    f = tmp_path / "stream.bin"
    f.write_bytes(data)
    r_h, r_l = bn.decompress(f.read_bytes())
    (fwd_h, fwd_l), _ = bn(feats)
    assert r_h.shape == fwd_h.shape and r_l.shape == fwd_l.shape
    assert torch.allclose(r_h, fwd_h, atol=1e-4), "high recon roundtrip != eval forward"
    assert torch.allclose(r_l, fwd_l, atol=1e-4), "low recon roundtrip != eval forward"


def test_fused_estimated_bpp_tracks_actual_filesize():
    torch.manual_seed(0)
    bn = FusedLatentBottleneck(C_HIGH, C_LOW, latent_channels=96, hyper_channels=96,
                               hidden=96, num_down=1, act="convbn").eval()
    bn.update(force=True)
    # sizeable feature so the fixed header is negligible (deployment regime)
    feats = (torch.randn(1, C_HIGH, 64, 64), torch.randn(1, C_LOW, 32, 32))
    actual_bits = len(bn.compress(feats)) * 8
    _, info = bn(feats)
    est_bits = sum(float(torch.log(t).sum() / -math.log(2.0))
                   for t in info["likelihoods"].values())
    rel = abs(est_bits - actual_bits) / actual_bits
    assert rel < 0.20, f"estimated {est_bits:.0f} vs actual {actual_bits} bits ({rel:.1%} off)"


def test_fused_attn_variant_keeps_contract():
    bn = _bn(use_attn=True)
    x_h, x_l = _feats()
    (f_h, f_l), info = bn((x_h, x_l))
    assert f_h.shape == x_h.shape and f_l.shape == x_l.shape
    assert set(info["likelihoods"]) == {"y", "z"}


def test_fused_split_student_integration(teacher, image):
    cfg = OmegaConf.create({"name": "rtformer_fused",
                            "args": {"latent_channels": 64, "hyper_channels": 64,
                                     "hidden": 64, "num_down": 1}})
    bn = build_bottleneck(cfg, in_channels=split_in_channels(teacher, "mid"))
    student = SplitStudent(teacher, bn, split="mid").eval()
    out = student(image)
    assert {"preds", "target", "recon", "info"} <= set(out)
    assert isinstance(out["recon"], tuple) and len(out["recon"]) == 2
    for t, r in zip(out["target"], out["recon"]):
        assert r.shape == t.shape
    assert "seg" in out["preds"]
    assert set(out["info"]["likelihoods"]) == {"y", "z"}
