#!/usr/bin/env python3
"""Pi microbench: buffer rANS vs list-tax path (no torch)."""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "codec" / "rans_fast"))
import _rans_fast as rans  # noqa: E402


def make_tables(n_cdfs=64, alphabet=64, seed=0):
    rng = np.random.default_rng(seed)
    cdfs = np.zeros((n_cdfs, alphabet + 1), dtype=np.int32)
    sizes = np.full(n_cdfs, alphabet + 1, dtype=np.int32)
    offsets = np.full(n_cdfs, alphabet // 2, dtype=np.int32)
    for i in range(n_cdfs):
        freqs = rng.integers(1, 64, size=alphabet).astype(np.int64)
        freqs = (freqs * 65536) // freqs.sum()
        freqs[-1] += 65536 - freqs.sum()
        cdfs[i, 0] = 0
        cdfs[i, 1:] = np.cumsum(freqs)
        assert cdfs[i, -1] == 65536
    return cdfs, sizes, offsets


def make_symbols(n, n_cdfs, alphabet, offsets, seed=1):
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, n_cdfs, size=n, dtype=np.int32)
    max_v = alphabet - 2
    values = rng.integers(0, max_v, size=n, dtype=np.int32)
    symbols = values + offsets[indexes]
    return symbols.astype(np.int32), indexes


def time_fn(fn, warmup=10, iters=40):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def main():
    cdfs, sizes, offsets = make_tables()
    n = 192 * 16 * 16
    symbols, indexes = make_symbols(n, cdfs.shape[0], cdfs.shape[1] - 1, offsets)
    print(f"host=pi n_symbols={n} n_cdfs={cdfs.shape[0]} alphabet={cdfs.shape[1]-1}")
    payload = {"b": None}

    def enc_fast():
        payload["b"] = rans.encode_with_indexes(symbols, indexes, cdfs, sizes, offsets)

    def dec_fast():
        rans.decode_with_indexes(payload["b"], indexes, cdfs, sizes, offsets)

    enc_fast()
    out = rans.decode_with_indexes(payload["b"], indexes, cdfs, sizes, offsets)
    assert np.array_equal(out, symbols), "roundtrip mismatch"
    nbytes = len(payload["b"])
    ef = time_fn(enc_fast)
    df = time_fn(dec_fast)

    def enc_list():
        s = np.asarray(symbols.tolist(), dtype=np.int32)
        i = np.asarray(indexes.tolist(), dtype=np.int32)
        c = np.asarray(cdfs.tolist(), dtype=np.int32)
        z = np.asarray(sizes.tolist(), dtype=np.int32)
        o = np.asarray(offsets.tolist(), dtype=np.int32)
        payload["b"] = rans.encode_with_indexes(s, i, c, z, o)

    def dec_list():
        i = np.asarray(indexes.tolist(), dtype=np.int32)
        c = np.asarray(cdfs.tolist(), dtype=np.int32)
        z = np.asarray(sizes.tolist(), dtype=np.int32)
        o = np.asarray(offsets.tolist(), dtype=np.int32)
        rans.decode_with_indexes(payload["b"], i, c, z, o)

    el = time_fn(enc_list)
    dl = time_fn(dec_list)
    print(f"bytes={nbytes}")
    print(f"fast      enc={ef:.2f} ms  dec={df:.2f} ms")
    print(f"list_tax  enc={el:.2f} ms  dec={dl:.2f} ms")
    print(f"speedup   enc={el/ef:.2f}x  dec={dl/df:.2f}x")


if __name__ == "__main__":
    main()
