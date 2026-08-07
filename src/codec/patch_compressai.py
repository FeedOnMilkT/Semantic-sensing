"""Patch CompressAI ``EntropyModel.compress/decompress`` to use semsense fast rANS.

Importing this module (or calling :func:`apply`) replaces the stock methods that
materialise huge Python lists via ``.tolist()`` on every call. RTFF/RTFD container
layout is unchanged — only the entropy string generation path is swapped.
"""
from __future__ import annotations

_APPLIED = False


def apply() -> bool:
    """Monkey-patch CompressAI EntropyModel. Returns True if patched."""
    global _APPLIED
    if _APPLIED:
        return True
    try:
        from compressai.entropy_models.entropy_models import EntropyModel
    except ImportError:
        return False

    from .entropy_coder import compress_tensor, decompress_tensor

    def compress(self, inputs, indexes, means=None):
        return compress_tensor(self, inputs, indexes, means)

    def decompress(self, strings, indexes, dtype=None, means=None):
        import torch

        if dtype is None:
            dtype = torch.float
        return decompress_tensor(self, strings, indexes, dtype=dtype, means=means)

    EntropyModel.compress = compress
    EntropyModel.decompress = decompress
    _APPLIED = True
    return True
