"""§9: the estimated bpp (entropy-model -log2 likelihood) must track the ACTUAL
container bit count (phase2 spec §8.5: deviation < 10–20%). Validated on a sizeable
feature so the fixed container header (magic + per-branch shapes + per-string lengths)
is negligible vs the payload — exactly the deployment regime.
"""
import math

import torch

from src.models.bottlenecks.rtformer_distill import DualLatentBottleneck

C_HIGH, C_LOW = 128, 256


def _est_bits(info):
    liks = info["likelihoods"]
    return sum(float(torch.log(t).sum() / -math.log(2.0)) for t in liks.values())


def test_estimated_bpp_tracks_actual_filesize():
    torch.manual_seed(0)
    bn = DualLatentBottleneck(C_HIGH, C_LOW, latent_channels=96, hyper_channels=96,
                              hidden=96, num_down=1, act="convbn").eval()
    bn.update(force=True)
    # sizeable feature so header overhead is negligible (deployment regime)
    x_h = torch.randn(1, C_HIGH, 64, 64)
    x_l = torch.randn(1, C_LOW, 32, 32)

    data = bn.compress((x_h, x_l))
    actual_bits = len(data) * 8
    _, info = bn((x_h, x_l))
    est_bits = _est_bits(info)

    rel = abs(est_bits - actual_bits) / actual_bits
    assert rel < 0.20, f"estimated {est_bits:.0f} vs actual {actual_bits} bits ({rel:.1%} off)"
