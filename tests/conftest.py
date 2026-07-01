"""pytest fixtures for the phase2 split-computing tests.

All tests run on CPU with RANDOM weights (hermetic — no real checkpoint needed):
the split-equivalence / contract / bitstream properties are about WIRING and the
codec, not trained values. The rANS range coder is CPU-only (and segfaults mixed
with a CUDA context on this NGC build), so compress/decompress stay on CPU.
"""
import os
import sys

# Single-threaded for the test process: this host shares 100+ cores with the team and
# CompressAI's CDF-table update (torch.erfc over the scale table) intermittently
# segfaults under the default multi-thread OpenMP pool here. Set BEFORE importing torch.
os.environ["SEMSENSE_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pytest
import torch
from omegaconf import OmegaConf

torch.set_num_threads(1)

# make `src` importable without installing the package
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src import _compat  # noqa: F401,E402  applies the CompressAI import shim first
from src.models import build_model  # noqa: E402


def _teacher_cfg(num_classes=8):
    """Tiny paper-faithful RTFormer teacher (passthrough(s8) + identity decoder +
    RTFormer-paper seg head) — the wiring the phase2 MID/LATE main line cuts."""
    return OmegaConf.create({
        "backbone": {"name": "rtformer", "args": {}},
        "neck": {"name": "passthrough", "args": {"key": "s8"}},
        "bottleneck": {"name": "identity", "args": {}},
        "decoder": {"name": "identity", "args": {}},
        "heads": {"seg": {"name": "seg",
                          "args": {"num_classes": num_classes, "upsample": 8, "mid": True}}},
    })


@pytest.fixture(scope="session")
def teacher():
    torch.manual_seed(0)
    return build_model(_teacher_cfg()).eval()


@pytest.fixture(scope="session")
def num_classes():
    return 8


@pytest.fixture
def image():
    torch.manual_seed(1)
    return torch.randn(1, 3, 64, 64)
