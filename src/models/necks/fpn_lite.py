"""Lightweight multi-scale fusion necks. Each consumes the backbone's 4-scale
dict {"s4","s8","s16","s32"} and returns a SINGLE fused feature at stride 4.

`in_channels` (a dict scale->C) is injected by MultiTaskNet from the chosen
backbone, so the neck stays backbone-agnostic and the swap is a one-line YAML
change. Both necks expose `.out_channels` (int) for the decoder to wire against.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNLite(nn.Module):
    """Top-down FPN: 1x1 laterals -> common dim, add upsampled coarser level,
    then sum all levels at stride 4."""
    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        self.scales = ["s4", "s8", "s16", "s32"]
        self.lateral = nn.ModuleDict(
            {s: nn.Conv2d(in_channels[s], out_channels, 1) for s in self.scales})
        self.smooth = nn.ModuleDict(
            {s: nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                              nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
             for s in self.scales})
        self.out_channels = out_channels

    def forward(self, feats):
        lat = {s: self.lateral[s](feats[s]) for s in self.scales}
        # top-down accumulation: s16<-s32, s8<-s16, s4<-s8
        for i in range(len(self.scales) - 2, -1, -1):
            hi, lo = self.scales[i], self.scales[i + 1]
            lat[hi] = lat[hi] + F.interpolate(lat[lo], size=lat[hi].shape[-2:], mode="bilinear", align_corners=False)
        out = self.smooth["s4"](lat["s4"])
        return out


class Passthrough(nn.Module):
    """No-fusion neck: forwards a single chosen backbone scale (default the
    stride-8 feature). Used by the paper-faithful RTFormer, whose backbone already
    aggregates context (DAPPM) into its s8 output — so no extra FPN fusion is
    wanted. `key` selects which scale; out_channels mirrors it."""
    def __init__(self, in_channels, key="s8", **_):   # **_ absorbs inherited args (e.g. out_channels)
        super().__init__()
        self.key = key
        self.out_channels = in_channels[key]

    def forward(self, feats):
        return feats[self.key]


class MLPFusion(nn.Module):
    """SegFormer-style: 1x1 each scale to common dim, upsample all to stride 4,
    concat, fuse with 1x1."""
    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        self.scales = ["s4", "s8", "s16", "s32"]
        self.proj = nn.ModuleDict(
            {s: nn.Conv2d(in_channels[s], out_channels, 1) for s in self.scales})
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.out_channels = out_channels

    def forward(self, feats):
        ref = feats["s4"].shape[-2:]
        ups = [F.interpolate(self.proj[s](feats[s]), size=ref, mode="bilinear", align_corners=False)
               for s in self.scales]
        return self.fuse(torch.cat(ups, dim=1))
