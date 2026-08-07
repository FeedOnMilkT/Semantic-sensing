"""Fused single-latent communication bottleneck — the phase2 MID split rate-side fix
(fused-latent workflow §4).

The dual-latent `rtformer_distill` codes x_h and x_l with two INDEPENDENT hyperpriors,
so the (large) mutual information between the two same-source branches is paid twice
(workflow §2). This bottleneck is the two-scale instance of L-MSFC's interleaved
fuse-and-encode (Kim et al., TCSVT 2023, FENet/DRNet): encode the LARGE scale first
(stride-2 align block on x_h), concat with x_l, and jointly analyse into ONE latent y
with ONE mean-scale hyperprior — the joint code pays for the mutual information once.
Fully convolutional, no token attention, no autoregressive context model (= the
original's "w/o CM" tier; Hailo-deployable). The Entropic Student two-stage training
recipe is unchanged.

  x_h (c_high @ 1/8) --align_h: _block(stride=2)--> c_high @ 1/16       # "encode large scale first"
  concat(·, x_l c_low @ 1/16) --g_a--> y (latent @ 1/32) --h_a/h_s/entropy--> y_hat
  y_hat --g_s trunk--> hidden @ 1/16 --head_l--> x̂_l (c_low @ 1/16)     # low res out first
       concat(trunk, x̂_l) --head_h: up + _block + conv--> x̂_h (c_high @ 1/8)
                                                          # DRNet bottom-up mixing analogue

Contract matches DualLatentBottleneck so SplitStudent / StudentTrainer / bpp_from_info
are untouched: forward((x_h, x_l)) -> ((x̂_h, x̂_l), info), info["likelihoods"]={"y","z"}.
compress/decompress serialise ONE bytes container (magic RTFF, single branch: both
feat_hw go in the header for the decoder-side size guards).

Optional `use_attn` (default OFF): Cheng et al. CVPR 2020 simplified attention — a
purely convolutional trunk-and-mask sigmoid gate on both sides of the latent (spatial
rate allocation), NOT token attention, so it does not affect Hailo deployability.
Kept as the capacity upgrade for workflow §7 risk 2 (joint g_a underfitting).
"""
import struct

from ... import _compat  # noqa: F401  applies the add_safe_globals shim before CompressAI
try:
    from ...codec.patch_compressai import apply as _apply_fast_rans
    _apply_fast_rans()
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN

from .hyperprior import _Analysis, _Synthesis, _block, _scale_table

_MAGIC = b"RTFF"


class _BoundedIGDN(GDN):
    """IGDN with a bounded divisive-norm multiplier (NaN fix, workflow §7 risk 5).

    Stock IGDN computes x * sqrt(beta + sum_j gamma[j,i] x_j^2), which is QUADRATIC in
    large activations. The fused x̂_h synthesis path chains THREE IGDNs (g_s_trunk head +
    up block + head_h) vs the dual-branch reference's two, so outlier activations get
    amplified with exponent 2^3: measured on real Stage-1 training, one batch at near-init
    already mapped teacher-range features (|x|≈5) to recon |x̂_h|≈91, and mid-training
    spikes reached dist≈1e13 / grad-norm≈7e14; one such spike overflowing fp32 turns
    clip_grad_norm_'s total_norm to inf, its clip coefficient to 0, and 0*inf grads write
    NaN into every codec parameter (the observed all-NaN collapse at step ~2700).

    The guard clamps the multiplier sqrt(norm) at MAX_MULT, making the layer LINEAR in x
    past the bound and the 3-stage chain boundedly amplifying. Healthy-regime multipliers
    measured over 3k real steps peak at 5.2 (encoder GDNs stay ~1.0 and keep the stock
    layer), so 8.0 is inactive in normal operation: RD behaviour and the fused-vs-distill
    comparison are untouched; only the pathological transients are cut. Parameter names /
    shapes match GDN exactly -> checkpoints trade both ways.
    """
    MAX_MULT = 8.0

    def __init__(self, ch):
        super().__init__(ch, inverse=True)

    def forward(self, x):
        _, C, _, _ = x.size()
        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma).reshape(C, C, 1, 1)
        norm = F.conv2d(x ** 2, gamma, beta)
        return x * norm.sqrt().clamp(max=self.MAX_MULT)


def _syn_block(cin, cout, act):
    """Synthesis-side _block twin (same Sequential layout / state-dict keys as
    hyperprior._block(inverse=True)) with the bounded IGDN. convbn tier unchanged."""
    if act == "gdn":
        return nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1, bias=True), _BoundedIGDN(cout))
    return _block(cin, cout, 1, act=act, inverse=True)


class _ResBlock(nn.Module):
    """Bottlenecked residual unit (Cheng et al. CVPR 2020 attention building block):
    1x1 (C->C/2) -> ReLU -> 3x3 -> ReLU -> 1x1 (C/2->C), plus identity."""
    def __init__(self, c):
        super().__init__()
        h = c // 2
        self.body = nn.Sequential(
            nn.Conv2d(c, h, 1), nn.ReLU(inplace=True),
            nn.Conv2d(h, h, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(h, c, 1))

    def forward(self, x):
        return x + self.body(x)


class _SpatialGate(nn.Module):
    """Cheng et al. simplified attention: out = x + trunk(x) * sigmoid(mask(x)).
    Convolutional sigmoid gating for spatial bit allocation — no softmax/tokens."""
    def __init__(self, c):
        super().__init__()
        self.trunk = nn.Sequential(*[_ResBlock(c) for _ in range(3)])
        self.mask = nn.Sequential(*[_ResBlock(c) for _ in range(3)],
                                  nn.Conv2d(c, c, 1), nn.Sigmoid())

    def forward(self, x):
        return x + self.trunk(x) * self.mask(x)


class FusedLatentBottleneck(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, latent_channels=192,
                 hyper_channels=128, hidden=128, num_down=1, hyper_down=1,
                 act="convbn", use_attn=False):
        super().__init__()
        self.c_high, self.c_low = in_channels_high, in_channels_low
        # encoder: align x_h to x_l's grid, concat, joint analysis -> single y
        self.align_h = _block(in_channels_high, in_channels_high, stride=2, act=act)
        self.g_a = _Analysis(in_channels_high + in_channels_low, latent_channels,
                             hidden, num_down, act=act)
        # hyper transforms stay conv+ReLU (same tier policy as HyperpriorBottleneck)
        self.h_a = _Analysis(latent_channels, hyper_channels, hidden, hyper_down)
        self.h_s = _Synthesis(hyper_channels, latent_channels * 2, hidden, hyper_down)
        self.entropy_bottleneck = EntropyBottleneck(hyper_channels)
        self.gaussian_conditional = GaussianConditional(None)
        # decoder: shared trunk up to 1/16, low head first, high head mixes x̂_l back in.
        # All synthesis IGDNs are the bounded variant (_BoundedIGDN) — the x̂_h path
        # chains three of them, which is what blew up unbounded (see _BoundedIGDN).
        ups = [nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                             _syn_block(hidden, hidden, act=act))
               for _ in range(num_down)]
        self.g_s_trunk = nn.Sequential(_syn_block(latent_channels, hidden, act=act), *ups)
        self.head_l = nn.Conv2d(hidden, in_channels_low, 3, 1, 1)
        self.head_h = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _syn_block(hidden + in_channels_low, hidden, act=act),
            nn.Conv2d(hidden, in_channels_high, 3, 1, 1))
        self.attn_a = _SpatialGate(latent_channels) if use_attn else None
        self.attn_s = _SpatialGate(latent_channels) if use_attn else None

    @property
    def in_channels(self):
        return (self.c_high, self.c_low)

    # ---- shared transform paths ----
    def _analyze(self, feats):
        x_h, x_l = feats
        a = self.align_h(x_h)
        if a.shape[-2:] != x_l.shape[-2:]:               # odd-size stride rounding
            a = F.interpolate(a, size=x_l.shape[-2:], mode="bilinear", align_corners=False)
        y = self.g_a(torch.cat([a, x_l], dim=1))
        if self.attn_a is not None:
            y = self.attn_a(y)
        return y

    def _synthesize(self, y_hat, hw_h, hw_l):
        if self.attn_s is not None:
            y_hat = self.attn_s(y_hat)
        t = self.g_s_trunk(y_hat)
        x_l = self.head_l(t)
        x_h = self.head_h(torch.cat([t, x_l], dim=1))    # bottom-up mixing: x̂_l feeds x̂_h
        if x_h.shape[-2:] != tuple(hw_h):                # guard odd sizes (as hyperprior.py)
            x_h = F.interpolate(x_h, size=tuple(hw_h), mode="bilinear", align_corners=False)
        if x_l.shape[-2:] != tuple(hw_l):
            x_l = F.interpolate(x_l, size=tuple(hw_l), mode="bilinear", align_corners=False)
        return x_h, x_l

    def _hyper_params(self, z_hat, y_hw):
        """(means, scales) snapped to y's spatial size — same desync guard as
        HyperpriorBottleneck._hyper_params (y_hw travels in the container)."""
        y_hw = tuple(y_hw)
        means, scales = self.h_s(z_hat).chunk(2, dim=1)
        if means.shape[-2:] != y_hw:
            means = F.interpolate(means, size=y_hw, mode="bilinear", align_corners=False)
            scales = F.interpolate(scales, size=y_hw, mode="bilinear", align_corners=False)
        return means, scales

    def forward(self, feats):
        x_h, x_l = feats
        y = self._analyze(feats)
        z = self.h_a(y)
        z_hat, z_lik = self.entropy_bottleneck(z)
        means, scales = self._hyper_params(z_hat, y.shape[-2:])
        y_hat, y_lik = self.gaussian_conditional(y, scales, means=means)
        f_h, f_l = self._synthesize(y_hat, x_h.shape[-2:], x_l.shape[-2:])
        return (f_h, f_l), {"likelihoods": {"y": y_lik, "z": z_lik}, "latent": y}

    # ---- deployment path (real single-container bitstream, magic RTFF) ----
    def compress(self, feats):
        """-> ONE bytes container. Layout (big-endian): magic | u16 z_h,z_w | u16 y_h,y_w
        | u16 x_h feat_hw | u16 x_l feat_hw | u8 B | z_strings | y_strings, each string
        as u32 len + bytes. Both feat_hw are needed for the decoder-side size guards."""
        x_h, x_l = feats
        y = self._analyze(feats)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.shape[-2:])
        means, scales = self._hyper_params(z_hat, y.shape[-2:])
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means)
        B = len(y_strings)
        assert len(z_strings) == B, "y/z batch mismatch"
        buf = bytearray(_MAGIC)
        buf += struct.pack(">HHHHHHHHB", *z.shape[-2:], *y.shape[-2:],
                           *x_h.shape[-2:], *x_l.shape[-2:], B)
        for strings in (z_strings, y_strings):           # z first, then y
            for s in strings:
                buf += struct.pack(">I", len(s)) + s
        return bytes(buf)

    def decompress(self, data):
        assert data[:4] == _MAGIC, "bad container magic"
        off = 4
        z_h, z_w, y_h, y_w, fh_h, fh_w, fl_h, fl_w, B = struct.unpack_from(">HHHHHHHHB", data, off)
        off += 17
        streams = []
        for _stream in range(2):                         # z, then y
            ss = []
            for _b in range(B):
                (ln,) = struct.unpack_from(">I", data, off); off += 4
                ss.append(data[off:off + ln]); off += ln
            streams.append(ss)
        z_strings, y_strings = streams
        z_hat = self.entropy_bottleneck.decompress(z_strings, (z_h, z_w))
        means, scales = self._hyper_params(z_hat, (y_h, y_w))
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_hat = self.gaussian_conditional.decompress(y_strings, indexes, means=means)
        return self._synthesize(y_hat, (fh_h, fh_w), (fl_h, fl_w))

    # ---- optimiser plumbing (same split as HyperpriorBottleneck) ----
    def update(self, force=True):
        updated = self.gaussian_conditional.update_scale_table(_scale_table(), force=force)
        updated |= self.entropy_bottleneck.update(force=force)
        return updated

    def aux_loss(self):
        return self.entropy_bottleneck.loss()

    def codec_parameters(self):
        return [p for n, p in self.named_parameters() if not n.endswith("quantiles")]

    def aux_parameters(self):
        return [p for n, p in self.named_parameters() if n.endswith("quantiles")]
