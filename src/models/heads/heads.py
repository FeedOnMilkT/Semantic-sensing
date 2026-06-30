"""Task heads (CLAUDE.md §4). The shared decoder feeds each head; heads are the
task fork. Seg is light (classification); depth/normal are heavier and use a
depthwise 3x3 to preserve boundaries (continuous regression: blurry edges wreck
contours). Each head upsamples by a static `upsample` factor to input resolution
(scale_factor keeps ONNX export static).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegHead(nn.Module):
    """Seg classification head. Default minimal (1x1 -> logits). With `mid=True`
    it is the RTFormer-paper head: a 3x3 conv (hidden=in_channels) + BN + ReLU
    before the 1x1 classifier (§3.2). Then bilinear upsample to input resolution."""
    def __init__(self, in_channels, num_classes=19, upsample=4, mid=False):
        super().__init__()
        self.pre = None
        if mid:
            self.pre = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))
        self.cls = nn.Conv2d(in_channels, num_classes, 1)
        self.up = upsample

    def forward(self, x):
        if self.pre is not None:
            x = self.pre(x)
        x = self.cls(x)
        return F.interpolate(x, scale_factor=self.up, mode="bilinear", align_corners=False)


class _RegHead(nn.Module):
    """Shared body for depth/normal: 1x1 expand -> depthwise 3x3 (boundary) ->
    1x1 to out_ch -> bilinear upsample. Subclasses add the final activation."""
    def __init__(self, in_channels, out_ch, hidden=None, upsample=4):
        super().__init__()
        hidden = hidden or in_channels
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_ch, 1),
        )
        self.up = upsample

    def _upsample(self, x):
        return F.interpolate(x, scale_factor=self.up, mode="bilinear", align_corners=False)


class DepthHead(_RegHead):
    """...-> 1 channel -> upsample -> sigmoid * max_depth (metric depth)."""
    def __init__(self, in_channels, max_depth=80.0, hidden=None, upsample=4):
        super().__init__(in_channels, out_ch=1, hidden=hidden, upsample=upsample)
        self.max_depth = max_depth

    def forward(self, x):
        x = self._upsample(self.body(x))
        return torch.sigmoid(x) * self.max_depth


class NormalHead(_RegHead):
    """...-> 3 channels -> upsample -> L2-normalize along channel (unit vector)."""
    def __init__(self, in_channels, hidden=None, upsample=4):
        super().__init__(in_channels, out_ch=3, hidden=hidden, upsample=upsample)

    def forward(self, x):
        x = self._upsample(self.body(x))
        return F.normalize(x, dim=1)
