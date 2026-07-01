"""§9: the dual-latent single-container bitstream (phase2 spec §4.4).
  compress((x_h,x_l)) -> ONE bytes container -> write to disk -> read -> decompress
  reproduces the eval-mode forward reconstruction (rounded latents), exactly.
The container packs both branches' z/y CompressAI strings + shapes; decompress is the
deterministic inverse. rANS coding is CPU-only.
"""
import torch

from src.models.bottlenecks.rtformer_distill import DualLatentBottleneck, _pack, _unpack

C_HIGH, C_LOW = 128, 256


def _bn_and_feats():
    torch.manual_seed(0)
    bn = DualLatentBottleneck(C_HIGH, C_LOW, latent_channels=64, hyper_channels=64,
                              hidden=64, num_down=1, act="convbn").eval()
    bn.update(force=True)                       # build CDF tables before coding
    x_h = torch.randn(1, C_HIGH, 16, 16)
    x_l = torch.randn(1, C_LOW, 8, 8)
    return bn, (x_h, x_l)


def test_container_pack_unpack_inverse():
    bn, feats = _bn_and_feats()
    p_h = bn.high.compress(feats[0])
    p_l = bn.low.compress(feats[1])
    data = _pack([p_h, p_l])
    assert data[:4] == b"RTFD"
    back = _unpack(data)
    assert len(back) == 2
    for orig, got in [(p_h, back[0]), (p_l, back[1])]:
        assert got["z_shape"] == orig["z_shape"] and got["feat_hw"] == orig["feat_hw"]
        assert got["strings"] == orig["strings"]


def test_roundtrip_matches_eval_forward(tmp_path):
    bn, feats = _bn_and_feats()
    # deployment: compress -> disk -> read -> decompress
    data = bn.compress(feats)
    assert isinstance(data, (bytes, bytearray))
    f = tmp_path / "stream.bin"
    f.write_bytes(data)
    r_h, r_l = bn.decompress(f.read_bytes())
    # eval-mode forward (quantised) reconstruction = what deployment must reproduce
    (fwd_h, fwd_l), _ = bn(feats)
    assert torch.allclose(r_h, fwd_h, atol=1e-4), "high branch roundtrip != eval forward"
    assert torch.allclose(r_l, fwd_l, atol=1e-4), "low branch roundtrip != eval forward"
