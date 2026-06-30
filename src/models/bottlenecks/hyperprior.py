"""Mean-scale hyperprior communication bottleneck — the full (optimal-deployable)
Phase-2 bottleneck.

    f --g_a--> y --h_a--> z --EntropyBottleneck--> z_hat --h_s--> (means, scales)
                                                                       |
         y --GaussianConditional(y | means, scales)--> y_hat --g_s--> f_hat

Two latents cross the channel: the hyper-latent z (factorized prior) carries the
per-element (means, scales) side info; the main latent y is then coded with the
predicted Gaussian -> far better rate-distortion than a factorized prior on y alone
(Minnen et al., NeurIPS 2018).

Design choices for THIS project:
- All transforms (g_a/g_s/h_a/h_s) are conv + BN + ReLU (no GDN / LayerNorm) -> fold
  for Hailo; g_s reconstructs to in_channels so the downstream tail wires unchanged.
- NO autoregressive context model (it is not Hailo-deployable) -> this is the optimal
  *deployable* entropy model, not the absolute LIC SOTA.

Contract matches the bottleneck slot:  forward(f) -> (f_hat, info), info carries the
y and z likelihoods. compress/decompress give the real two-stream bitstream.
"""
import math

from ... import _compat  # noqa: F401  applies the add_safe_globals shim before CompressAI

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional


def _cbr(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def _scale_table(smin=0.11, smax=256.0, levels=64):
    """Log-spaced Gaussian scale table for the conditional coder (CompressAI default)."""
    return torch.exp(torch.linspace(math.log(smin), math.log(smax), levels))


class _Analysis(nn.Module):
    """Conv stack reducing channels with `num_down` stride-2 steps -> latent (no
    final activation: latents are signed)."""
    def __init__(self, c_in, c_out, hidden, num_down):
        super().__init__()
        layers, c = [], c_in
        for _ in range(num_down):
            layers.append(_cbr(c, hidden, stride=2))
            c = hidden
        if num_down == 0:
            layers.append(_cbr(c, hidden, 1))
            c = hidden
        layers.append(nn.Conv2d(c, c_out, 3, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _Synthesis(nn.Module):
    """Conv stack mirroring `num_up` bilinear upsamples -> output (no final activation)."""
    def __init__(self, c_in, c_out, hidden, num_up):
        super().__init__()
        self.head = _cbr(c_in, hidden, 1)
        self.ups = nn.ModuleList(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                _cbr(hidden, hidden, 1))
            for _ in range(num_up))
        self.proj = nn.Conv2d(hidden, c_out, 3, 1, 1)

    def forward(self, x):
        x = self.head(x)
        for u in self.ups:
            x = u(x)
        return self.proj(x)


class HyperpriorBottleneck(nn.Module):
    def __init__(self, in_channels, latent_channels=128, hyper_channels=128,
                 hidden=128, num_down=1, hyper_down=1):
        super().__init__()
        self.in_channels = in_channels
        self.num_down = num_down
        self.g_a = _Analysis(in_channels, latent_channels, hidden, num_down)
        self.g_s = _Synthesis(latent_channels, in_channels, hidden, num_down)
        self.h_a = _Analysis(latent_channels, hyper_channels, hidden, hyper_down)
        self.h_s = _Synthesis(hyper_channels, latent_channels * 2, hidden, hyper_down)  # -> means, scales
        self.entropy_bottleneck = EntropyBottleneck(hyper_channels)   # factorized prior on z
        self.gaussian_conditional = GaussianConditional(None)         # conditional coder for y

    def forward(self, f):
        y = self.g_a(f)
        z = self.h_a(y)
        z_hat, z_lik = self.entropy_bottleneck(z)
        means, scales = self.h_s(z_hat).chunk(2, dim=1)
        y_hat, y_lik = self.gaussian_conditional(y, scales, means=means)
        f_hat = self.g_s(y_hat)
        if f_hat.shape[-2:] != f.shape[-2:]:                # guard odd sizes
            f_hat = F.interpolate(f_hat, size=f.shape[-2:], mode="bilinear", align_corners=False)
        return f_hat, {"likelihoods": {"y": y_lik, "z": z_lik}, "latent": y}

    # ---- deployment path (real two-stream bitstream) ----
    def compress(self, f):
        y = self.g_a(f)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.shape[-2:])
        means, scales = self.h_s(z_hat).chunk(2, dim=1)
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means)
        return {"strings": [y_strings, z_strings], "z_shape": tuple(z.shape[-2:]),
                "feat_hw": tuple(f.shape[-2:])}

    def decompress(self, payload):
        y_strings, z_strings = payload["strings"]
        z_hat = self.entropy_bottleneck.decompress(z_strings, payload["z_shape"])
        means, scales = self.h_s(z_hat).chunk(2, dim=1)
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_hat = self.gaussian_conditional.decompress(y_strings, indexes, means=means)
        f_hat = self.g_s(y_hat)
        hw = payload.get("feat_hw")
        if hw is not None and f_hat.shape[-2:] != tuple(hw):
            f_hat = F.interpolate(f_hat, size=tuple(hw), mode="bilinear", align_corners=False)
        return f_hat

    # ---- optimiser plumbing (CompressAI main + aux split) ----
    def update(self, force=True):
        """Build CDF tables. The conditional coder needs its scale table set first."""
        updated = self.gaussian_conditional.update_scale_table(_scale_table(), force=force)
        updated |= self.entropy_bottleneck.update(force=force)
        return updated

    def aux_loss(self):
        return self.entropy_bottleneck.loss()

    def codec_parameters(self):
        """g_a/g_s/h_a/h_s + entropy model means/scales (everything except the
        factorized CDF quantiles)."""
        return [p for n, p in self.named_parameters() if not n.endswith("quantiles")]

    def aux_parameters(self):
        return [p for n, p in self.named_parameters() if n.endswith("quantiles")]
