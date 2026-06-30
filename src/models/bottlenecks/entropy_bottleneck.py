"""EntropyBottleneck module — the rate machinery (CompressAI), reused for Phase 2.

Thin wrapper over CompressAI's EntropyBottleneck, operating on a single feature
tensor (B, C, H, W) and deliberately task/dataset agnostic:

    training:   forward(y) -> (y_hat, likelihoods)   # uniform noise + learned entropy model
    deployment: compress(y) -> {strings, shape}      # round + rANS range coder -> bitstream
                decompress(strings, shape) -> y_hat

Rate is the learned entropy model's estimate; ``estimated_bpp`` normalises bits by the
number of *image* pixels (passed in), matching the bpp convention used across the
project so curves are comparable regardless of latent spatial size.

We do NOT use a pure-VIB rate: KL is a loose upper bound on bpp and emits no bits.
EntropyBottleneck is simultaneously the rate term and the actual coder.
"""
from ... import _compat  # noqa: F401  applies the add_safe_globals shim before CompressAI

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck as _CompressAIEntropyBottleneck


class EntropyBottleneckModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.entropy_bottleneck = _CompressAIEntropyBottleneck(channels)

    def forward(self, y):
        """Training/eval forward. Returns (y_hat, likelihoods), both (B, C, H, W).

        Train: y_hat = y + U(-0.5, 0.5); eval: y_hat = round(y) (dequantised)."""
        y_hat, likelihoods = self.entropy_bottleneck(y)
        return y_hat, likelihoods

    @staticmethod
    def estimated_bpp(likelihoods, num_pixels):
        """Entropy-model bit estimate per image pixel: -sum log2(p) / num_pixels."""
        bits = torch.log(likelihoods).sum() / (-torch.log(torch.tensor(2.0)))
        return bits / num_pixels

    def aux_loss(self):
        """Quantile loss for the learned CDF — optimised by a separate aux optimizer."""
        return self.entropy_bottleneck.loss()

    def update(self, force=True):
        """Build the CDF tables required before compress/decompress."""
        return self.entropy_bottleneck.update(force=force)

    # ---- deployment path (real bitstream) ----
    def compress(self, y):
        """Quantise + range-code. Returns {'strings': [...], 'shape': (H, W)}."""
        strings = self.entropy_bottleneck.compress(y)
        return {"strings": strings, "shape": y.shape[-2:]}

    def decompress(self, strings, shape):
        """Inverse of compress -> y_hat (B, C, H, W)."""
        return self.entropy_bottleneck.decompress(strings, shape)
