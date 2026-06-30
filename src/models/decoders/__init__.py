"""Decoder factory (no registry). `in_channels` injected from the neck."""
from .conv_decoder import ConvDecoder, IdentityDecoder

_DECODERS = {
    "conv_decoder": ConvDecoder,
    "identity": IdentityDecoder,
}


def build_decoder(cfg, in_channels):
    return _DECODERS[cfg.name](in_channels=in_channels, **cfg.get("args", {}))
