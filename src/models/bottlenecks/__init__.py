"""Bottleneck factory (no registry) — the neck->decoder slot.

Uniform contract:  forward(f) -> (f_hat, info)
  - identity     : (f, None)                       no-op placeholder (seg / Phase 1).
  - compression  : (f_hat, {"likelihoods", ...})   Phase-2 communication bottleneck
                   (learned analysis/synthesis transforms + CompressAI EntropyBottleneck;
                   emits a real bitstream via compress/decompress and a learned rate term).

Adding a bottleneck = one builder fn + one dict entry (grep-able, no magic). The
`compression` builder pulls in CompressAI, so it is imported LAZILY inside the builder
— plain seg training (identity) never touches CompressAI.
"""
import torch.nn as nn


class Identity(nn.Module):
    """No-op slot: returns the feature unchanged + None info, matching the tuple
    contract so MultiTaskNet / trainers handle every bottleneck uniformly. Phase 2
    swaps this for `compression` (same forward(f) -> (f, info) signature)."""
    def __init__(self, **_):
        super().__init__()

    def forward(self, f):
        return f, None


def _build_identity(cfg, in_channels):
    return Identity(**cfg.get("args", {}))


def _build_compression(cfg, in_channels):
    from .compression import CommBottleneck   # lazy: imports CompressAI only on demand
    if in_channels is None:
        raise ValueError("compression bottleneck needs in_channels (split feature channels)")
    return CommBottleneck(in_channels=in_channels, **cfg.get("args", {}))


def _build_hyperprior(cfg, in_channels):
    from .hyperprior import HyperpriorBottleneck   # lazy: imports CompressAI only on demand
    if in_channels is None:
        raise ValueError("hyperprior bottleneck needs in_channels (split feature channels)")
    return HyperpriorBottleneck(in_channels=in_channels, **cfg.get("args", {}))


def _build_rtformer_distill(cfg, in_channels):
    """Dual-latent / single-container bottleneck (phase2 MID/LATE main line).
    `in_channels` must be a (c_high, c_low) tuple (set by build_student from the
    split's two-branch channels)."""
    from .rtformer_distill import DualLatentBottleneck   # lazy: imports CompressAI on demand
    if not (isinstance(in_channels, (tuple, list)) and len(in_channels) == 2):
        raise ValueError("rtformer_distill needs in_channels=(c_high, c_low) for the dual payload")
    c_high, c_low = in_channels
    return DualLatentBottleneck(in_channels_high=c_high, in_channels_low=c_low, **cfg.get("args", {}))


_BOTTLENECKS = {
    "identity": _build_identity,                  # no-op (seg / Phase 1)
    "compression": _build_compression,            # factorized EntropyBottleneck (de-risk baseline)
    "hyperprior": _build_hyperprior,              # mean-scale hyperprior (full / optimal-deployable)
    "rtformer_distill": _build_rtformer_distill,  # dual-latent / single-container (MID/LATE main line)
}


def build_bottleneck(cfg, in_channels=None):
    return _BOTTLENECKS[cfg.name](cfg, in_channels)
