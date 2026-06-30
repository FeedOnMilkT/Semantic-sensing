"""Communication bottleneck for split inference — the Phase-2 core.

Inserts a learned compression transform into the neck->decoder slot (the
RTFormer-stage3 split-point analogue):

    f --g_a (analysis)--> y --EntropyBottleneck (quantise + learned rate)--> y_hat
      --g_s (synthesis)--> f_hat

The quantised latent y_hat is what crosses the channel; g_a is the send-side
transform, g_s the receive-side. Deploy-friendly by construction — conv + BN + ReLU
only (no GDN / LayerNorm) — so both halves fold for Hailo (project constraint: the
receiver decoder must be DFC-compilable). g_s reconstructs to the SAME channel count
as f, so the downstream decoder/heads wire against it unchanged.

Contract (matches the bottleneck slot):
    forward(f) -> (f_hat, info)     info = {"likelihoods", "latent"}
Deployment:
    compress(f) -> payload(dict)    decompress(payload) -> f_hat

Optimiser plumbing follows CompressAI: codec_parameters() (g_a/g_s + entropy model)
trained on distortion + beta*rate; aux_parameters() (the CDF quantiles) trained
separately on aux_loss().
"""
import torch.nn as nn
import torch.nn.functional as F

from .entropy_bottleneck import EntropyBottleneckModule


def _cbr(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class AnalysisTransform(nn.Module):
    """g_a: feature (C_in) -> latent (D), with `num_down` stride-2 reductions."""
    def __init__(self, c_in, c_latent, hidden, num_down):
        super().__init__()
        layers, c = [], c_in
        for _ in range(num_down):
            layers.append(_cbr(c, hidden, stride=2))
            c = hidden
        if num_down == 0:                    # still mix channels when not downsampling
            layers.append(_cbr(c, hidden, stride=1))
            c = hidden
        layers.append(nn.Conv2d(c, c_latent, 3, 1, 1))   # -> latent (signed, no activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SynthesisTransform(nn.Module):
    """g_s: latent (D) -> feature (C_out), mirroring `num_up` bilinear upsamples."""
    def __init__(self, c_latent, c_out, hidden, num_up):
        super().__init__()
        self.head = _cbr(c_latent, hidden, stride=1)
        self.ups = nn.ModuleList(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                _cbr(hidden, hidden, stride=1))
            for _ in range(num_up))
        self.proj = nn.Conv2d(hidden, c_out, 3, 1, 1)    # -> feature channels (no activation)

    def forward(self, y):
        y = self.head(y)
        for u in self.ups:
            y = u(y)
        return self.proj(y)


class CommBottleneck(nn.Module):
    """Learned analysis/synthesis transforms wrapped around a CompressAI
    EntropyBottleneck. Compresses C_in channels -> D-channel latent (optionally
    spatially downsampled `num_down` times) and reconstructs back to C_in."""

    def __init__(self, in_channels, latent_channels=64, hidden=128, num_down=1):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.num_down = num_down
        self.g_a = AnalysisTransform(in_channels, latent_channels, hidden, num_down)
        self.entropy = EntropyBottleneckModule(latent_channels)
        self.g_s = SynthesisTransform(latent_channels, in_channels, hidden, num_down)

    def forward(self, f):
        y = self.g_a(f)
        y_hat, likelihoods = self.entropy(y)
        f_hat = self.g_s(y_hat)
        if f_hat.shape[-2:] != f.shape[-2:]:             # guard odd input sizes
            f_hat = F.interpolate(f_hat, size=f.shape[-2:], mode="bilinear", align_corners=False)
        return f_hat, {"likelihoods": likelihoods, "latent": y}

    @staticmethod
    def estimated_bpp(info, num_pixels):
        return EntropyBottleneckModule.estimated_bpp(info["likelihoods"], num_pixels)

    # ---- deployment path (real bitstream) ----
    def compress(self, f):
        """Send side: g_a + range-code. Returns a self-describing payload dict."""
        y = self.g_a(f)
        payload = self.entropy.compress(y)               # {strings, shape}
        payload["feat_hw"] = tuple(f.shape[-2:])
        return payload

    def decompress(self, payload):
        """Receive side: range-decode + g_s -> f_hat at the original feature size."""
        y_hat = self.entropy.decompress(payload["strings"], payload["shape"])
        f_hat = self.g_s(y_hat)
        hw = payload.get("feat_hw")
        if hw is not None and f_hat.shape[-2:] != tuple(hw):
            f_hat = F.interpolate(f_hat, size=tuple(hw), mode="bilinear", align_corners=False)
        return f_hat

    # ---- optimiser plumbing (CompressAI main + aux split) ----
    def update(self, force=True):
        return self.entropy.update(force=force)

    def aux_loss(self):
        return self.entropy.aux_loss()

    def codec_parameters(self):
        """Main params: g_a + g_s + entropy model (everything except CDF quantiles)."""
        return [p for n, p in self.named_parameters() if "quantiles" not in n]

    def aux_parameters(self):
        """CDF quantile params — trained by a separate aux optimizer (CompressAI)."""
        return [p for n, p in self.named_parameters() if "quantiles" in n]
