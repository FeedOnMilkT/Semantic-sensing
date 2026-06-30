"""Neck factory (no registry). `in_channels` is injected by MultiTaskNet."""
from .fpn_lite import FPNLite, MLPFusion, Passthrough

_NECKS = {
    "fpn_lite": FPNLite,
    "mlp_fusion": MLPFusion,
    "passthrough": Passthrough,
}


def build_neck(cfg, in_channels):
    return _NECKS[cfg.name](in_channels=in_channels, **cfg.get("args", {}))
