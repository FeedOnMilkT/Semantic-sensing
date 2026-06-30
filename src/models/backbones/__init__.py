"""Backbone factory — NO registry / decorators. Adding a backbone = one import
line + one dict entry. Grep-able, no magic (see CLAUDE.md hard constraint #1).

`pretrained` is handled here (not per-backbone) so swapping backbone stays a true
one-line `name:` change: it's popped from args and applied via the backbone's
`load_pretrained` method if it has one; backbones without pretrained support
(rtformer/topformer) just warn and train from scratch.
"""
from .mit_b0 import MiTB0
from .rtformer import RTFormer
from .topformer import TopFormer

_BACKBONES = {
    "mit_b0": MiTB0,
    "rtformer": RTFormer,
    "topformer": TopFormer,
}


def build_backbone(cfg):
    args = dict(cfg.get("args", {}))
    pretrained = args.pop("pretrained", None)
    bb = _BACKBONES[cfg.name](**args)
    if pretrained:
        if hasattr(bb, "load_pretrained"):
            bb.load_pretrained(pretrained)
        else:
            print(f"[build_backbone] WARN: '{cfg.name}' has no pretrained support; "
                  f"ignoring pretrained={pretrained} (training from scratch)")
    return bb
