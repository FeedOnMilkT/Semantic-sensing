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


_BOTTLENECKS = {
    "identity": _build_identity,            # no-op (seg / Phase 1)
    "compression": _build_compression,      # factorized EntropyBottleneck (de-risk baseline)
    "hyperprior": _build_hyperprior,        # mean-scale hyperprior (full / optimal-deployable)
}


def build_bottleneck(cfg, in_channels=None):
    return _BOTTLENECKS[cfg.name](cfg, in_channels)
