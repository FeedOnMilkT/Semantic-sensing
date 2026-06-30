"""Head factory (no registry). `in_channels` injected from the decoder.

A head cfg is `{name, args}`. The task NAME (the key in cfg.heads) is decoupled
from the head TYPE (cfg.heads[task].name) so you could, e.g., attach two seg
heads — but typically task name == head type.
"""
from .heads import SegHead, DepthHead, NormalHead

_HEADS = {
    "seg": SegHead,
    "depth": DepthHead,
    "normal": NormalHead,
}


def build_head(cfg, in_channels):
    return _HEADS[cfg.name](in_channels=in_channels, **cfg.get("args", {}))
