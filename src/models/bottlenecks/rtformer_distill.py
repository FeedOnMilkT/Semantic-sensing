"""Dual-latent / single-container communication bottleneck — the phase2 MID/LATE
split bottleneck (spec §4).

The RTFormer conv->attention cut emits a DUAL payload: a high-res branch (x_h, 1/8)
and a low-res branch (x_l, 1/16 at mid, 1/32 at late). Each branch is compressed by
its OWN independent mean-scale hyperprior (analysis/synthesis/entropy model, NO
coupling between branches — reconstruction is "split the container, decode each,
synthesise each", so there is no fuse-then-mis-split problem, spec §4.1). The two
branches' bitstreams are serialised into ONE container (spec §4.4) = the single
interface handed to the teammate's channel coder.

  (x_h, x_l) --[g_a^h | g_a^l]--> (y_h, y_l) --hyperpriors--> (ŷ_h, ŷ_l)
             --[g_s^h | g_s^l]--> (x̂_h, x̂_l)
  compress(): pack([magic | n | (len,payload)* for each branch's z/y strings]) -> ONE bytes

Contract matches the bottleneck slot (spec §4.3): forward(feats) -> (feats_hat, info),
feats = (x_h, x_l) tuple. info["likelihoods"] sums ALL four streams (y_h,z_h,y_l,z_l)
so bpp (normalised by the IMAGE H×W upstream) counts every transmitted bit. Optimiser
plumbing (codec/aux params, aux_loss, update) delegates to both branches.
"""
import struct

import torch.nn as nn

from .hyperprior import HyperpriorBottleneck

_MAGIC = b"RTFD"


def _pack(branch_payloads):
    """Serialise per-branch HyperpriorBottleneck.compress() dicts into ONE bytes.
    Layout (big-endian): magic | u8 n_branches | per branch:
      u16 z_h, u16 z_w, u16 f_h, u16 f_w, u8 B,
      then z_strings then y_strings, each as B × (u32 len + bytes)  (spec §4.4)."""
    buf = bytearray(_MAGIC)
    buf += struct.pack(">B", len(branch_payloads))
    for p in branch_payloads:
        y_strings, z_strings = p["strings"]
        z_h, z_w = p["z_shape"]
        y_h, y_w = p["y_shape"]
        f_h, f_w = p["feat_hw"]
        B = len(y_strings)
        assert len(z_strings) == B, "y/z batch mismatch in a branch"
        buf += struct.pack(">HHHHHHB", z_h, z_w, y_h, y_w, f_h, f_w, B)
        for strings in (z_strings, y_strings):        # z first, then y (spec §4.1 order)
            for s in strings:
                buf += struct.pack(">I", len(s)) + s
    return bytes(buf)


def _unpack(data):
    """Inverse of `_pack` -> list of per-branch dicts (compat HyperpriorBottleneck.decompress)."""
    assert data[:4] == _MAGIC, "bad container magic"
    off = 4
    (n_branches,) = struct.unpack_from(">B", data, off); off += 1
    payloads = []
    for _ in range(n_branches):
        z_h, z_w, y_h, y_w, f_h, f_w, B = struct.unpack_from(">HHHHHHB", data, off); off += 13
        streams = []
        for _stream in range(2):                       # z, then y
            ss = []
            for _b in range(B):
                (ln,) = struct.unpack_from(">I", data, off); off += 4
                ss.append(data[off:off + ln]); off += ln
            streams.append(ss)
        z_strings, y_strings = streams
        payloads.append({"strings": [y_strings, z_strings], "z_shape": (z_h, z_w),
                         "y_shape": (y_h, y_w), "feat_hw": (f_h, f_w)})
    return payloads


class DualLatentBottleneck(nn.Module):
    """Two independent HyperpriorBottleneck branches (high / low) + a single-container
    bitstream. Branch order is fixed (high, low) for deterministic (de)serialisation."""

    def __init__(self, in_channels_high, in_channels_low, **hyper_kwargs):
        super().__init__()
        self.high = HyperpriorBottleneck(in_channels=in_channels_high, **hyper_kwargs)
        self.low = HyperpriorBottleneck(in_channels=in_channels_low, **hyper_kwargs)

    @property
    def in_channels(self):
        return (self.high.in_channels, self.low.in_channels)

    def forward(self, feats):
        x_h, x_l = feats
        f_h, info_h = self.high(x_h)
        f_l, info_l = self.low(x_l)
        liks = {"y_h": info_h["likelihoods"]["y"], "z_h": info_h["likelihoods"]["z"],
                "y_l": info_l["likelihoods"]["y"], "z_l": info_l["likelihoods"]["z"]}
        info = {"likelihoods": liks, "latent": {"y_h": info_h["latent"], "y_l": info_l["latent"]}}
        return (f_h, f_l), info

    # ---- deployment path (real single-container bitstream) ----
    def compress(self, feats):
        x_h, x_l = feats
        return _pack([self.high.compress(x_h), self.low.compress(x_l)])

    def decompress(self, data):
        p_h, p_l = _unpack(data)
        return self.high.decompress(p_h), self.low.decompress(p_l)

    # ---- optimiser plumbing (delegate to both branches) ----
    def _branches(self):
        return (self.high, self.low)

    def codec_parameters(self):
        return [p for b in self._branches() for p in b.codec_parameters()]

    def aux_parameters(self):
        return [p for b in self._branches() for p in b.aux_parameters()]

    def aux_loss(self):
        return sum(b.aux_loss() for b in self._branches())

    def update(self, force=True):
        return all(b.update(force=force) for b in self._branches())
