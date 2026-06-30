"""Bottleneck factory (no registry).

Demo 1 ships ONLY `identity` (a no-op placeholder), per CLAUDE.md constraint #4:
the slot sits between neck and decoder so Phase 2 can drop a CompressAI
EntropyBottleneck in here WITHOUT restructuring the framework. The contract is
just feature-in -> feature-out (same shape); a future compressing bottleneck may
additionally return a rate term, which the trainer can pick up then.
"""
import torch.nn as nn


class Identity(nn.Module):
    """No-op slot. Phase 2 replaces this with EntropyBottleneck (quantize +
    learned entropy model + rate), keeping the same forward(f) -> f signature."""
    def __init__(self, **_):
        super().__init__()

    def forward(self, f):
        return f


_BOTTLENECKS = {
    "identity": Identity,
}


def build_bottleneck(cfg):
    return _BOTTLENECKS[cfg.name](**cfg.get("args", {}))
