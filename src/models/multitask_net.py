"""Top-level assembly (the blueprint core, CLAUDE.md §"顶层装配").

build_model(cfg) / MultiTaskNet(cfg) assembles every component purely from cfg.
Channel counts flow downstream automatically: backbone.out_channels -> neck ->
decoder -> heads, so swapping the backbone (one YAML line) needs no other edit.

forward(x, tasks=None) runs the backbone/neck/bottleneck/decoder once, then only
the requested heads (default: all). The trainer passes the batch's active tasks
so absent heads never run.
"""
import torch.nn as nn

from .backbones import build_backbone
from .necks import build_neck
from .bottlenecks import build_bottleneck
from .decoders import build_decoder
from .heads import build_head


class MultiTaskNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg.backbone)
        self.neck = build_neck(cfg.neck, in_channels=self.backbone.out_channels)
        self.bottleneck = build_bottleneck(cfg.bottleneck)            # demo1: identity
        self.decoder = build_decoder(cfg.decoder, in_channels=self.neck.out_channels)
        self.heads = nn.ModuleDict(
            {t: build_head(h, in_channels=self.decoder.out_channels) for t, h in cfg.heads.items()})

    def forward(self, x, tasks=None):
        feats = self.backbone(x)          # dict {"s4","s8","s16","s32"}
        f = self.neck(feats)
        f = self.bottleneck(f)            # demo1 no-op; phase2 compresses here
        d = self.decoder(f)
        return {t: self.heads[t](d) for t in (tasks or self.heads.keys()) if t in self.heads}


def build_model(cfg):
    return MultiTaskNet(cfg)
