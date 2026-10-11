"""PyTorch RTFormer-Base, adapted from PaddleSeg release/2.8.

Architecture source: ``paddleseg/models/rtformer.py`` at PaddleSeg commit
``a485168da281e942af7d1643aa726d44df8c9951`` (Apache-2.0; see the source
header and https://github.com/PaddlePaddle/PaddleSeg/blob/a485168da281e942af7d1643aa726d44df8c9951/paddleseg/models/rtformer.py).
The segmentation-head wiring follows that implementation.  The public split API
keeps this project’s stage-3 payload: high [B,128,H/8,W/8], low
[B,256,H/16,W/16].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d(cin, cout, kernel_size, stride=1, padding=0, bias=False):
    return nn.Conv2d(cin, cout, kernel_size, stride, padding, bias=bias)


def bn2d(channels, eps=1e-5):
    # Paddle BatchNorm momentum=0.1 is the PyTorch update momentum=0.1.
    return nn.BatchNorm2d(channels, eps=eps, momentum=0.1)


def _init_kaiming(module):
    """PaddleSeg RTFormer's Conv/BN initialization for residual and DAPPM paths."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _init_truncated(module, std=0.02):
    """PaddleSeg's truncated-normal Conv initialization used by EABlock paths."""
    if isinstance(module, nn.Conv2d):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 no_relu=False):
        super().__init__()
        self.conv1 = conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = bn2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = bn2d(out_channels)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        residual = x if self.downsample is None else self.downsample(x)
        y = y + residual
        return y if self.no_relu else self.relu(y)


def _make_layer(in_channels, out_channels, blocks, stride=1):
    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1, stride), bn2d(out_channels))
    layers = [BasicBlock(in_channels, out_channels, stride, downsample)]
    for i in range(1, blocks):
        layers.append(BasicBlock(out_channels, out_channels,
                                 no_relu=(i == blocks - 1)))
    return nn.Sequential(*layers)


class DropPath(nn.Module):
    """Per-sample stochastic depth used by the official EABlock."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class MLP(nn.Module):
    """Official pre-BN, 3x3-conv/GELU/dropout MLP (no channel expansion)."""
    def __init__(self, channels, drop_rate=0.0):
        super().__init__()
        self.norm = bn2d(channels, eps=1e-6)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.drop = nn.Dropout(drop_rate)
        self.apply(_init_kaiming)

    def forward(self, x):
        y = self.drop(self.act(self.conv1(self.norm(x))))
        return self.drop(self.conv2(y))


class ExternalAttention(nn.Module):
    """RTFormer grouped double-normalized external attention."""
    def __init__(self, in_channels, out_channels, inter_channels,
                 num_heads=8, use_cross_kv=False):
        super().__init__()
        if out_channels % num_heads:
            raise ValueError("out_channels must be divisible by num_heads")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.use_cross_kv = use_cross_kv
        self.same_in_out_chs = in_channels == out_channels
        self.norm = bn2d(in_channels)
        if use_cross_kv:
            if not self.same_in_out_chs:
                raise ValueError("cross K/V attention requires equal input/output channels")
        else:
            self.k = nn.Parameter(torch.empty(inter_channels, in_channels, 1, 1))
            self.v = nn.Parameter(torch.empty(out_channels, inter_channels, 1, 1))
            nn.init.normal_(self.k, std=0.001)
            nn.init.normal_(self.v, std=0.001)
        self.apply(lambda module: _init_truncated(module, std=0.001))

    def _double_normalize(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, self.num_heads, c // self.num_heads, h * w)
        x = x.softmax(dim=-1)
        x = x / (x.sum(dim=2, keepdim=True) + 1e-6)
        return x.reshape(b, c, h, w)

    def forward(self, x, cross_k=None, cross_v=None):
        x = self.norm(x)
        if not self.use_cross_kv:
            x = F.conv2d(x, self.k, stride=2 if not self.same_in_out_chs else 1)
            x = self._double_normalize(x)
            return F.conv2d(x, self.v)
        if cross_k is None or cross_v is None:
            raise ValueError("cross_k and cross_v are required for cross attention")
        b, c, h, w = x.shape
        x = x.reshape(1, b * c, h, w)
        x = F.conv2d(x, cross_k, groups=b)
        # Softmax over the 12x12 memory positions; scale matches PaddleSeg.
        x = (x.reshape(b, self.inter_channels, h, w) * self.inter_channels ** -0.5)
        x = x.softmax(dim=1).reshape(1, b * self.inter_channels, h, w)
        x = F.conv2d(x, cross_v, groups=b)
        return x.reshape(b, c, h, w)


class EABlock(nn.Module):
    """One stepped dual-resolution Efficient Attention block."""
    def __init__(self, in_channels, out_channels, num_heads=8, drop_rate=0.0,
                 drop_path_rate=0.0, use_injection=True, cross_size=12):
        super().__init__()
        in_h, in_l = in_channels
        out_h, out_l = out_channels
        if in_h != out_h:
            raise ValueError("RTFormer EABlock preserves high-branch channels")
        self.out_channels_h = out_h
        self.proj_flag = in_l != out_l
        self.use_injection = use_injection
        self.cross_size = cross_size
        if self.proj_flag:
            self.attn_shortcut_l = nn.Sequential(
                bn2d(in_l), conv2d(in_l, out_l, 1, 2))
            self.attn_shortcut_l.apply(_init_kaiming)
        self.attn_l = ExternalAttention(in_l, out_l, out_l, num_heads)
        self.mlp_l = MLP(out_l, drop_rate)
        self.drop_path = DropPath(drop_path_rate)
        self.compression = nn.Sequential(
            bn2d(out_l), nn.ReLU(inplace=False), conv2d(out_l, out_h, 1))
        self.compression.apply(_init_kaiming)
        self.attn_h = ExternalAttention(
            in_h, in_h, cross_size * cross_size, num_heads, use_cross_kv=True)
        self.mlp_h = MLP(out_h, drop_rate)
        self.cross_kv = nn.Sequential(
            bn2d(out_l), nn.AdaptiveMaxPool2d((cross_size, cross_size)),
            conv2d(out_l, 2 * out_h, 1))
        self.cross_kv.apply(_init_truncated)
        if use_injection:
            self.down = nn.Sequential(
                bn2d(out_h), nn.ReLU(inplace=False),
                conv2d(out_h, out_l // 2, 3, 2, 1),
                bn2d(out_l // 2), nn.ReLU(inplace=False),
                conv2d(out_l // 2, out_l, 3, 2, 1))
            self.down.apply(_init_kaiming)

    def forward(self, inputs):
        x_h, x_l = inputs
        x_l_res = self.attn_shortcut_l(x_l) if self.proj_flag else x_l
        x_l = x_l_res + self.drop_path(self.attn_l(x_l))
        x_l = x_l + self.drop_path(self.mlp_l(x_l))

        x_h = x_h + F.interpolate(
            self.compression(x_l), size=x_h.shape[-2:], mode="bilinear")
        kv = self.cross_kv(x_l)
        cross_k, cross_v = kv.chunk(2, dim=1)
        b, c, _, _ = cross_k.shape
        cross_k = cross_k.permute(0, 2, 3, 1).reshape(-1, c, 1, 1)
        cross_v = cross_v.reshape(-1, self.cross_size * self.cross_size, 1, 1)
        x_h = x_h + self.drop_path(self.attn_h(x_h, cross_k, cross_v))
        x_h = x_h + self.drop_path(self.mlp_h(x_h))
        if self.use_injection:
            x_l = x_l + self.down(x_h)
        return x_h, x_l


class DAPPM(nn.Module):
    """Official DAPPM: progressive pooled context and additive fusion."""
    def __init__(self, in_channels, inter_channels, out_channels):
        super().__init__()
        self.scale0 = nn.Sequential(bn2d(in_channels), nn.ReLU(inplace=False),
                                    conv2d(in_channels, inter_channels, 1))
        self.scale1 = nn.Sequential(nn.AvgPool2d(5, 2, 2, count_include_pad=True),
                                    bn2d(in_channels), nn.ReLU(inplace=False),
                                    conv2d(in_channels, inter_channels, 1))
        self.scale2 = nn.Sequential(nn.AvgPool2d(9, 4, 4, count_include_pad=True),
                                    bn2d(in_channels), nn.ReLU(inplace=False),
                                    conv2d(in_channels, inter_channels, 1))
        self.scale3 = nn.Sequential(nn.AvgPool2d(17, 8, 8, count_include_pad=True),
                                    bn2d(in_channels), nn.ReLU(inplace=False),
                                    conv2d(in_channels, inter_channels, 1))
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), bn2d(in_channels),
                                    nn.ReLU(inplace=False),
                                    conv2d(in_channels, inter_channels, 1))
        self.process1 = self._process(inter_channels)
        self.process2 = self._process(inter_channels)
        self.process3 = self._process(inter_channels)
        self.process4 = self._process(inter_channels)
        self.compression = nn.Sequential(
            bn2d(inter_channels * 5), nn.ReLU(inplace=False),
            conv2d(inter_channels * 5, out_channels, 1))
        self.shortcut = nn.Sequential(
            bn2d(in_channels), nn.ReLU(inplace=False),
            conv2d(in_channels, out_channels, 1))

    @staticmethod
    def _process(channels):
        return nn.Sequential(bn2d(channels), nn.ReLU(inplace=False),
                             conv2d(channels, channels, 3, 1, 1))

    def forward(self, x):
        size = x.shape[-2:]
        up = lambda y: F.interpolate(y, size=size, mode="bilinear")
        x0 = self.scale0(x)
        x1 = self.process1(up(self.scale1(x)) + x0)
        x2 = self.process2(up(self.scale2(x)) + x1)
        x3 = self.process3(up(self.scale3(x)) + x2)
        x4 = self.process4(up(self.scale4(x)) + x3)
        return self.compression(torch.cat((x0, x1, x2, x3, x4), dim=1)) + self.shortcut(x)


class RTFormer(nn.Module):
    """RTFormer-Base trunk plus DAPPM, with the paper's stage-3 split API."""
    def __init__(self, in_ch=3, channels=(64, 128, 256, 512), mem=128,
                 groups=8, cross_size=12, drop_rate=0.0, drop_path_rate=0.1,
                 use_injection=(True, False), spp_channels=128):
        super().__init__()
        c1, c2, c3, c4 = channels
        if mem != 128 or groups != 8:
            # Official ExternalAttention uses num_heads=8 and fixed cross_size;
            # legacy arguments remain accepted only at their source defaults.
            raise ValueError("official RTFormer-Base uses mem=128/groups=8")
        self.conv1 = nn.Sequential(
            conv2d(in_ch, c1, 3, 2, 1), bn2d(c1), nn.ReLU(inplace=False),
            conv2d(c1, c1, 3, 2, 1), bn2d(c1), nn.ReLU(inplace=False))
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = _make_layer(c1, c1, 2)
        self.layer2 = _make_layer(c1, c2, 2, stride=2)
        self.layer3 = _make_layer(c2, c3, 2, stride=2)
        self.layer3_ = _make_layer(c2, c2, 1)
        self.compression3 = nn.Sequential(
            bn2d(c3), nn.ReLU(inplace=False), conv2d(c3, c2, 1))
        self.layer4 = EABlock((c2, c3), (c2, c4), 8, drop_rate,
                              drop_path_rate, use_injection[0], cross_size)
        self.layer5 = EABlock((c2, c4), (c2, c4), 8, drop_rate,
                              drop_path_rate, use_injection[1], cross_size)
        self.dappm = DAPPM(c4, spp_channels, c2)
        self.conv1.apply(_init_kaiming)
        self.layer1.apply(_init_kaiming)
        self.layer2.apply(_init_kaiming)
        self.layer3.apply(_init_kaiming)
        self.layer3_.apply(_init_kaiming)
        self.compression3.apply(_init_kaiming)
        self.dappm.apply(_init_kaiming)
        self.out_channels = {"s4": c1, "s8": 2 * c2, "s16": c3, "s32": c4}
        self.split_channels = {"early": c2, "mid": (c2, c3), "late": (c2, c4)}

    def load_pretrained(self, path):
        """Load a tensor-only local ImageNet export with strict topology checks."""
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "backbone" in sd:
            sd = sd["backbone"]
        elif isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        if not isinstance(sd, dict) or not all(torch.is_tensor(v) for v in sd.values()):
            raise TypeError("expected a tensor-only RTFormer backbone state dict")
        if any(k.startswith("backbone.") for k in sd):
            sd = {k.removeprefix("backbone."): v for k, v in sd.items()}
        expected = {k: v for k, v in self.state_dict().items() if not k.startswith("dappm.")}
        got = {k: v for k, v in sd.items() if k in expected}
        bad_shapes = {k: (tuple(got[k].shape), tuple(expected[k].shape))
                      for k in got if got[k].shape != expected[k].shape}
        missing = sorted(set(expected) - set(got))
        unexpected = sorted(set(sd) - set(expected))
        if bad_shapes or missing or unexpected:
            raise RuntimeError(
                "incompatible RTFormer checkpoint; retrain/export with this topology "
                f"(missing={missing[:8]}, unexpected={unexpected[:8]}, "
                f"shape_mismatches={list(bad_shapes.items())[:8]})")
        self.load_state_dict(got, strict=False)
        print(f"[rtformer] loaded {len(got)} ImageNet tensors; DAPPM remains uninitialized")
        return [], []

    def _forward_stem(self, x):
        x1 = self.layer1(self.conv1(x))
        return x1, self.layer2(self.relu(x1))

    def _stage3(self, x2):
        x3 = self.layer3(self.relu(x2))
        x3_ = x2 + F.interpolate(self.compression3(x3), size=x2.shape[-2:],
                                 mode="bilinear")
        x3_ = self.layer3_(self.relu(x3_))
        return x3_, x3

    def _stage45(self, x_h, x_l):
        x_h, x_l = self.layer4((self.relu(x_h), self.relu(x_l)))
        return self.layer5((self.relu(x_h), self.relu(x_l)))

    def _head_feature(self, x_h, x_l):
        context = self.dappm(x_l)
        context = F.interpolate(context, size=x_h.shape[-2:], mode="bilinear")
        return torch.cat((x_h, context), dim=1)

    def forward_sender(self, x):
        s4, x2 = self._forward_stem(x)
        return s4, x2

    def forward_receiver(self, s4, x2):
        """Receiver tail for the older early split after stage 2."""
        x_h, x_l = self._stage3(x2)
        s16 = x_l
        x_h, x_l = self._stage45(x_h, x_l)
        return {"s4": s4, "s8": self._head_feature(x_h, x_l),
                "s16": s16, "s32": x_l}

    def forward_sender_mid(self, x):
        _, x2 = self._forward_stem(x)
        return self._stage3(x2)

    def forward_receiver_mid(self, x_h, x_l):
        s16 = x_l
        x_h, x_l = self._stage45(x_h, x_l)
        return {"s4": None, "s8": self._head_feature(x_h, x_l),
                "s16": s16, "s32": x_l}

    def forward_sender_late(self, x):
        return self._stage45(*self.forward_sender_mid(x))

    def forward_receiver_late(self, x_h, x_l):
        return {"s4": None, "s8": self._head_feature(x_h, x_l),
                "s16": None, "s32": x_l}

    def forward(self, x):
        s4, x2 = self._forward_stem(x)
        x_h, x_l = self._stage3(x2)
        s16 = x_l
        x_h, x_l = self._stage45(x_h, x_l)
        return {"s4": s4, "s8": self._head_feature(x_h, x_l),
                "s16": s16, "s32": x_l}

    def forward_trunk(self, x):
        """Return both stage-5 streams for two-branch ImageNet classification."""
        return self._stage45(*self.forward_sender_mid(x))
