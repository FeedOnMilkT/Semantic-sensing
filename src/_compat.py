"""Environment compatibility shims — import this BEFORE anything touches CompressAI.

This NGC container ships torch 2.4.0a alongside a torch_geometric that calls
``torch.serialization.add_safe_globals`` at import time (an API that only appears in
newer torch). CompressAI pulls torch_geometric in via its point-cloud dataset
registry, so a bare ``import compressai`` explodes. We use only CompressAI's
EntropyBottleneck (no point-cloud datasets, no untrusted checkpoint loading), so a
no-op shim for that one symbol is safe and unblocks the import chain.

Also caps CPU threads: this host has 100+ cores shared with the team; torch's default
OpenMP pool busy-spins one thread per core and thrashes the shared CPU. Override with
SEMSENSE_NUM_THREADS.
"""
import os

import torch
import torch.serialization as _s

if not hasattr(_s, "add_safe_globals"):
    _s.add_safe_globals = lambda *args, **kwargs: None

_THREADS = int(os.environ.get("SEMSENSE_NUM_THREADS", "8"))
torch.set_num_threads(_THREADS)
os.environ.setdefault("OMP_NUM_THREADS", str(_THREADS))

# Patch CompressAI EntropyModel.compress/decompress to the fast buffer rANS path
# (latent→bitstream only). Soft-fail if compressai is not installed yet.
try:
    from .codec.patch_compressai import apply as _apply_fast_rans

    _apply_fast_rans()
except Exception:
    pass
