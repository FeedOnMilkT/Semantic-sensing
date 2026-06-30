"""Shared convolutional decoder body (3-4 conv layers).

Per CLAUDE.md: a shared decoder main body produces one feature map that the
task heads fork off. A `transformer_hook` slot is RESERVED but left as a no-op
in Demo 1 — Phase 2 drops a deploy-friendly RTFormer/GFA block in here (the
attention enhancement that was split off the sender), gated by cfg, with no
refactor needed.
"""
import torch.nn as nn


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, 1, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class IdentityDecoder(nn.Module):
    """No-op decoder: passes the neck feature straight to the heads. Used by the
    paper-faithful RTFormer, where the seg-head's own 3x3 conv (in the head) plays
    the role of the decoder, so the shared trunk ends at the backbone's stride-8
    feature. Keeps the framework's decoder slot present (out_channels unchanged)."""
    def __init__(self, in_channels, **_):   # **_ absorbs inherited args (e.g. out_channels/num_layers)
        super().__init__()
        self.out_channels = in_channels

    def forward(self, f):
        return f


class ConvDecoder(nn.Module):
    def __init__(self, in_channels, out_channels=128, num_layers=3, transformer_enhance=False):
        super().__init__()
        assert 2 <= num_layers <= 4, "shared decoder body is 3-4 layers (2 allowed for tiny smoke runs)"
        chans = [in_channels] + [out_channels] * num_layers
        self.body = nn.Sequential(*[conv_block(chans[i], chans[i + 1]) for i in range(num_layers)])
        # RESERVED Phase-2 hook: deploy-friendly RTFormer/GFA attention block goes
        # here to rebuild stage4/5-level global context. No-op in Demo 1.
        if transformer_enhance:
            raise NotImplementedError("transformer enhancement is a Phase-2 hook; not implemented in Demo 1")
        self.transformer_hook = nn.Identity()
        self.out_channels = out_channels

    def forward(self, f):
        f = self.body(f)
        f = self.transformer_hook(f)   # Phase-2: attention enhancement injects here
        return f
