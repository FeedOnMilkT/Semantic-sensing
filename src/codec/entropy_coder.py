"""Select and invoke the CPU entropy coder for latent → bitstream.

Backends
--------
- ``fast``: contiguous int32 buffers → C++ rANS (``rans_fast`` extension)
- ``compressai``: stock CompressAI ``encode_with_indexes`` (Python lists)
- ``auto``: prefer ``fast`` when the extension is importable

Controlled by ``SEMSENSE_ENTROPY_CODER`` or :func:`set_coder`.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch

_CODER: Optional[str] = None  # None → read env / auto


def _detect_fast() -> bool:
    try:
        from .rans_fast import available

        return available()
    except Exception:
        return False


def available_coder() -> str:
    return "fast" if _detect_fast() else "compressai"


def get_coder_name() -> str:
    name = _CODER or os.environ.get("SEMSENSE_ENTROPY_CODER", "auto").lower()
    if name == "auto":
        return available_coder()
    if name not in ("fast", "compressai"):
        raise ValueError(f"unknown entropy coder {name!r}")
    if name == "fast" and not _detect_fast():
        raise RuntimeError(
            "SEMSENSE_ENTROPY_CODER=fast but rans_fast extension is not built"
        )
    return name


def set_coder(name: Optional[str]) -> None:
    """Set coder for this process. ``None`` restores env/auto selection."""
    global _CODER
    if name is not None and name not in ("fast", "compressai", "auto"):
        raise ValueError(f"unknown entropy coder {name!r}")
    _CODER = name


def _as_i32_1d(t: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(t.detach().cpu().reshape(-1).to(torch.int32).numpy())


def _cdf_tables_np(entropy_model):
    """Cache flat CDF numpy tables on the EntropyModel after update()."""
    cache = getattr(entropy_model, "_semsense_cdf_np", None)
    q = entropy_model._quantized_cdf
    lengths = entropy_model._cdf_length
    offsets = entropy_model._offset
    key = (id(q), id(lengths), id(offsets), tuple(q.shape), int(q.sum().item()) if q.numel() else 0)
    if cache is not None and cache[0] == key:
        return cache[1]
    cdfs = np.ascontiguousarray(q.detach().cpu().to(torch.int32).numpy())
    cdfs_sizes = np.ascontiguousarray(lengths.detach().cpu().reshape(-1).to(torch.int32).numpy())
    offs = np.ascontiguousarray(offsets.detach().cpu().reshape(-1).to(torch.int32).numpy())
    tables = (cdfs, cdfs_sizes, offs)
    entropy_model._semsense_cdf_np = (key, tables)
    return tables


def _cdf_tables_list(entropy_model):
    cache = getattr(entropy_model, "_semsense_cdf_list", None)
    q = entropy_model._quantized_cdf
    lengths = entropy_model._cdf_length
    offsets = entropy_model._offset
    key = (id(q), id(lengths), id(offsets), tuple(q.shape))
    if cache is not None and cache[0] == key:
        return cache[1]
    tables = (
        q.tolist(),
        lengths.reshape(-1).int().tolist(),
        offsets.reshape(-1).int().tolist(),
    )
    entropy_model._semsense_cdf_list = (key, tables)
    return tables


def encode_symbols(entropy_model, symbols: torch.Tensor, indexes: torch.Tensor) -> bytes:
    """Encode one batch item (already flattened or N-D) to a single bytes string."""
    coder = get_coder_name()
    if coder == "fast":
        from .rans_fast import encode_with_indexes

        cdfs, cdfs_sizes, offsets = _cdf_tables_np(entropy_model)
        return encode_with_indexes(
            _as_i32_1d(symbols),
            _as_i32_1d(indexes),
            cdfs,
            cdfs_sizes,
            offsets,
        )
    # compressai path (cached CDF lists still avoid re-tolist of the big table)
    cdfs, cdfs_sizes, offsets = _cdf_tables_list(entropy_model)
    return entropy_model.entropy_coder.encode_with_indexes(
        symbols.reshape(-1).int().tolist(),
        indexes.reshape(-1).int().tolist(),
        cdfs,
        cdfs_sizes,
        offsets,
    )


def decode_symbols(
    entropy_model,
    payload: bytes,
    indexes: torch.Tensor,
) -> torch.Tensor:
    """Decode one bytes string to a flat int32 CPU tensor matching ``indexes`` numel."""
    coder = get_coder_name()
    if coder == "fast":
        from .rans_fast import decode_with_indexes

        cdfs, cdfs_sizes, offsets = _cdf_tables_np(entropy_model)
        arr = decode_with_indexes(
            payload if isinstance(payload, (bytes, bytearray)) else bytes(payload),
            _as_i32_1d(indexes),
            cdfs,
            cdfs_sizes,
            offsets,
        )
        return torch.from_numpy(np.ascontiguousarray(arr))
    cdfs, cdfs_sizes, offsets = _cdf_tables_list(entropy_model)
    values = entropy_model.entropy_coder.decode_with_indexes(
        payload,
        indexes.reshape(-1).int().tolist(),
        cdfs,
        cdfs_sizes,
        offsets,
    )
    return torch.tensor(values, dtype=torch.int32)


def compress_tensor(
    entropy_model,
    inputs: torch.Tensor,
    indexes: torch.Tensor,
    means: Optional[torch.Tensor] = None,
) -> List[bytes]:
    """Drop-in replacement for ``EntropyModel.compress`` body (per-batch strings)."""
    symbols = entropy_model.quantize(inputs, "symbols", means)
    if inputs.size() != indexes.size():
        raise ValueError("`inputs` and `indexes` should have the same size.")
    entropy_model._check_cdf_size()
    entropy_model._check_cdf_length()
    entropy_model._check_offsets_size()
    strings = []
    for i in range(symbols.size(0)):
        strings.append(encode_symbols(entropy_model, symbols[i], indexes[i]))
    return strings


def decompress_tensor(
    entropy_model,
    strings: List[bytes],
    indexes: torch.Tensor,
    dtype: torch.dtype = torch.float,
    means: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Drop-in replacement for ``EntropyModel.decompress`` body."""
    if not isinstance(strings, (tuple, list)):
        raise ValueError("Invalid `strings` parameter type.")
    if not len(strings) == indexes.size(0):
        raise ValueError("Invalid strings or indexes parameters")
    entropy_model._check_cdf_size()
    entropy_model._check_cdf_length()
    entropy_model._check_offsets_size()

    outputs = indexes.new_empty(indexes.size())
    for i, s in enumerate(strings):
        values = decode_symbols(entropy_model, s, indexes[i])
        outputs[i] = values.to(device=outputs.device, dtype=outputs.dtype).reshape(
            outputs[i].size()
        )
    return entropy_model.dequantize(outputs, means, dtype)
