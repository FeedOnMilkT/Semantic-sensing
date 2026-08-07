"""Fast buffer rANS vs CompressAI: bit-identical strings + RTFF roundtrip.

Scope: latent → bitstream only (Phase A). Does not touch fuse/Hailo graphs.
"""
import os

import pytest
import torch

os.environ.setdefault("SEMSENSE_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from src import _compat  # noqa: F401
from src.codec.entropy_coder import set_coder
from src.codec.patch_compressai import apply
from src.codec.rans_fast import available
from src.models.bottlenecks.rtformer_fused import FusedLatentBottleneck

pytestmark = pytest.mark.skipif(not available(), reason="rans_fast extension not built")


@pytest.fixture(autouse=True)
def _patch():
    apply()
    yield
    set_coder(None)


def _fused_bn():
    torch.manual_seed(0)
    bn = FusedLatentBottleneck(
        128, 256, latent_channels=64, hyper_channels=64, hidden=64,
        num_down=1, act="convbn",
    ).eval()
    torch.set_num_threads(1)
    bn.update(force=True)
    return bn


def _feats():
    torch.manual_seed(0)
    return torch.randn(1, 128, 16, 16), torch.randn(1, 256, 8, 8)


def test_fast_vs_compressai_bytes_identical():
    bn = _fused_bn()
    feats = _feats()
    set_coder("compressai")
    ref = bn.compress(feats)
    set_coder("fast")
    fast = bn.compress(feats)
    assert ref == fast
    assert ref[:4] == b"RTFF"


def test_fast_stream_decodes_with_compressai():
    """Bit-compat gate: old CompressAI decoder must accept fast-path RTFF."""
    bn = _fused_bn()
    feats = _feats()
    set_coder("fast")
    data = bn.compress(feats)
    set_coder("compressai")
    r_h, r_l = bn.decompress(data)
    (fwd_h, fwd_l), _ = bn(feats)
    assert torch.allclose(r_h, fwd_h, atol=1e-4)
    assert torch.allclose(r_l, fwd_l, atol=1e-4)


def test_entropy_bottleneck_bytes_identical():
    from compressai.entropy_models import EntropyBottleneck

    torch.manual_seed(0)
    eb = EntropyBottleneck(16).eval()
    torch.set_num_threads(1)
    eb.update(force=True)
    x = torch.randn(1, 16, 8, 8)
    set_coder("compressai")
    ref = eb.compress(x)
    set_coder("fast")
    fast = eb.compress(x)
    assert ref == fast
    set_coder("compressai")
    assert torch.allclose(
        eb.decompress(fast, x.shape[-2:]), eb.decompress(ref, x.shape[-2:])
    )
