"""§9: the bottleneck slot contract (phase2 spec §4.3).
  forward(feats) -> (feats_hat, info), same structure in/out.
  - identity        : (f, None)
  - hyperprior      : single tensor -> (f_hat, {"likelihoods":{y,z}, "latent"})
  - rtformer_distill: (x_h, x_l)   -> ((x_h_hat, x_l_hat), {"likelihoods":{y_h,z_h,y_l,z_l}})
feats_hat reconstructs to the SAME channels/shape so the downstream tail wires unchanged.
"""
import torch

from src.models.bottlenecks import Identity
from src.models.bottlenecks.hyperprior import HyperpriorBottleneck
from src.models.bottlenecks.rtformer_distill import DualLatentBottleneck

C_HIGH, C_LOW = 128, 256


def _feats():
    torch.manual_seed(0)
    return torch.randn(1, C_HIGH, 16, 16), torch.randn(1, C_LOW, 8, 8)


def test_identity_contract():
    f = torch.randn(1, 32, 8, 8)
    f_hat, info = Identity()(f)
    assert f_hat.shape == f.shape and info is None


def test_hyperprior_single_branch_contract():
    x_h, _ = _feats()
    bn = HyperpriorBottleneck(in_channels=C_HIGH, latent_channels=64, hyper_channels=64,
                              hidden=64, num_down=1).eval()
    f_hat, info = bn(x_h)
    assert f_hat.shape == x_h.shape
    assert set(info["likelihoods"]) == {"y", "z"}


def test_dual_latent_contract_and_act_tiers():
    x_h, x_l = _feats()
    for act in ("convbn", "gdn"):
        bn = DualLatentBottleneck(C_HIGH, C_LOW, latent_channels=64, hyper_channels=64,
                                  hidden=64, num_down=1, act=act).eval()
        (f_h, f_l), info = bn((x_h, x_l))
        assert f_h.shape == x_h.shape, f"{act}: high recon shape"
        assert f_l.shape == x_l.shape, f"{act}: low recon shape"
        assert set(info["likelihoods"]) == {"y_h", "z_h", "y_l", "z_l"}, f"{act}: likelihood keys"
        # gradients reach the codec params (rate-distortion is trainable)
        (f_h2, f_l2), info2 = bn((x_h, x_l))
        loss = f_h2.mean() + f_l2.mean()
        loss.backward()
        assert any(p.grad is not None for p in bn.codec_parameters()), f"{act}: no codec grad"
