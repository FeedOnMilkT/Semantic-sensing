"""TopFormer backbone — faithful to the paper (Zhang et al., CVPR 2022,
"TopFormer: Token Pyramid Transformer for Mobile Semantic Segmentation").

Three parts per Fig 2 / §3:
  1. Token Pyramid Module : stacked MobileNetV2 inverted-residual blocks ->
     local tokens at 4 scales {s4,s8,s16,s32}.
  2. Scale-aware Semantics Extractor (SSE): average-pool EVERY scale to a common
     (small) size, CONCAT along channels, run L Transformer blocks (LeViT-style
     attention with reduced K/Q dim + FFN of two 1x1 conv with a 3x3 depthwise
     between). All BN + ReLU6 + 1x1-conv (no LayerNorm/GELU) -> deploy/Hailo
     friendly, foldable. Output = scale-aware global semantics.
  3. Semantics Injection Module (SIM): inject the global semantics back into each
     scale's local tokens: out = local_1x1 * sigmoid(sem_weight) + sem_bias
     (Hadamard + add). Applied to the last three scales (§3.4); s4 passes through.

Returns the {"s4","s8","s16","s32"} contract (s8/s16/s32 -> M channels after SIM,
s4 left as raw local tokens). Defaults approximate TopFormer-Base (M=256, L=4, 8 heads).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_act(cin, cout, k=1, s=1, act=True):
    layers = [nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout)]
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class InvertedResidual(nn.Module):
    """MobileNetV2 block: pointwise-expand -> depthwise -> pointwise-project."""
    def __init__(self, cin, cout, stride, expand=4):
        super().__init__()
        hidden = cin * expand
        self.use_res = stride == 1 and cin == cout
        self.conv = nn.Sequential(
            conv_bn_act(cin, hidden, 1),
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            conv_bn_act(hidden, cout, 1, act=False),
        )

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class Attention(nn.Module):
    """LeViT/TopFormer attention: reduced K/Q dim (key_dim per head), V has
    attn_ratio*key_dim per head, 1x1-conv projections + BN, ReLU6. (§3.2)"""
    def __init__(self, dim, key_dim=16, heads=8, attn_ratio=2):
        super().__init__()
        self.heads, self.key_dim = heads, key_dim
        self.scale = key_dim ** -0.5
        self.d = int(attn_ratio * key_dim)            # value dim per head
        self.nh_kd = key_dim * heads
        self.dh = self.d * heads
        self.to_q = conv_bn_act(dim, self.nh_kd, 1, act=False)
        self.to_k = conv_bn_act(dim, self.nh_kd, 1, act=False)
        self.to_v = conv_bn_act(dim, self.dh, 1, act=False)
        self.proj = nn.Sequential(nn.ReLU6(inplace=True), conv_bn_act(self.dh, dim, 1, act=False))

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        q = self.to_q(x).reshape(B, self.heads, self.key_dim, N).permute(0, 1, 3, 2)  # B,h,N,kd
        k = self.to_k(x).reshape(B, self.heads, self.key_dim, N)                       # B,h,kd,N
        v = self.to_v(x).reshape(B, self.heads, self.d, N).permute(0, 1, 3, 2)         # B,h,N,d
        attn = (q @ k * self.scale).softmax(dim=-1)                                    # B,h,N,N
        out = (attn @ v).permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        return self.proj(out)


class FFN(nn.Module):
    """TopFormer FFN (§3.2): 1x1 -> 3x3 depthwise -> 1x1, expansion factor 2, ReLU6."""
    def __init__(self, dim, exp=2):
        super().__init__()
        h = dim * exp
        self.fc1 = conv_bn_act(dim, h, 1)
        self.dw = nn.Sequential(nn.Conv2d(h, h, 3, 1, 1, groups=h, bias=False),
                                nn.BatchNorm2d(h), nn.ReLU6(inplace=True))
        self.fc2 = conv_bn_act(h, dim, 1, act=False)

    def forward(self, x):
        return self.fc2(self.dw(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim, key_dim, heads, attn_ratio, mlp_exp):
        super().__init__()
        self.attn = Attention(dim, key_dim, heads, attn_ratio)
        self.ffn = FFN(dim, mlp_exp)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


class SemanticsExtractor(nn.Module):
    """SSE: pool all scales to a common size, concat channels, run L Transformer
    blocks. Produces scale-aware global semantics of `sum(in_chs)` channels."""
    def __init__(self, in_chs, depth=4, key_dim=16, heads=8, attn_ratio=2, mlp_exp=2):
        super().__init__()
        self.total = sum(in_chs)
        self.blocks = nn.ModuleList(
            [TransformerBlock(self.total, key_dim, heads, attn_ratio, mlp_exp) for _ in range(depth)])

    def forward(self, feats, target_hw):
        t = torch.cat([F.adaptive_avg_pool2d(f, target_hw) for f in feats], dim=1)  # B, total, h, w
        for blk in self.blocks:
            t = blk(t)
        return t


class SIM(nn.Module):
    """Semantics Injection Module (§3.3): out = (1x1 local) * sigmoid(1x1 global)
    + (1x1 global). Global semantics upsampled to the local scale first."""
    def __init__(self, local_ch, global_ch, out_ch):
        super().__init__()
        self.local = conv_bn_act(local_ch, out_ch, 1, act=False)
        self.weight = conv_bn_act(global_ch, out_ch, 1, act=False)
        self.bias = conv_bn_act(global_ch, out_ch, 1, act=False)

    def forward(self, local, glob):
        g = F.interpolate(glob, size=local.shape[-2:], mode="bilinear", align_corners=False)
        return self.local(local) * torch.sigmoid(self.weight(g)) + self.bias(g)


class TopFormer(nn.Module):
    def __init__(self, in_ch=3, channels=(32, 64, 128, 160), M=256,
                 sse_depth=4, heads=8, key_dim=16, attn_ratio=2, mlp_exp=2):
        super().__init__()
        c0, c1, c2, c3 = channels
        # --- Token Pyramid Module (MobileNetV2 blocks) ---
        self.stem = conv_bn_act(in_ch, c0, 3, 2)                                          # 1/2
        self.t4 = nn.Sequential(InvertedResidual(c0, c0, 2), InvertedResidual(c0, c0, 1))  # 1/4
        self.t8 = nn.Sequential(InvertedResidual(c0, c1, 2), InvertedResidual(c1, c1, 1))  # 1/8
        self.t16 = nn.Sequential(InvertedResidual(c1, c2, 2), InvertedResidual(c2, c2, 1)) # 1/16
        self.t32 = nn.Sequential(InvertedResidual(c2, c3, 2), InvertedResidual(c3, c3, 1)) # 1/32
        # --- Scale-aware Semantics Extractor (over all 4 scales) ---
        self.sse = SemanticsExtractor([c0, c1, c2, c3], depth=sse_depth, key_dim=key_dim,
                                      heads=heads, attn_ratio=attn_ratio, mlp_exp=mlp_exp)
        total = c0 + c1 + c2 + c3
        # --- Semantics Injection into the last three scales (§3.4); s4 passes through ---
        self.sim8 = SIM(c1, total, M)
        self.sim16 = SIM(c2, total, M)
        self.sim32 = SIM(c3, total, M)
        self.out_channels = {"s4": c0, "s8": M, "s16": M, "s32": M}

    def forward(self, x):
        x = self.stem(x)
        s4 = self.t4(x)
        s8 = self.t8(s4)
        s16 = self.t16(s8)
        s32 = self.t32(s16)
        sem = self.sse([s4, s8, s16, s32], s32.shape[-2:])   # global semantics @ 1/32
        return {
            "s4": s4,                              # raw local tokens (passthrough)
            "s8": self.sim8(s8, sem),
            "s16": self.sim16(s16, sem),
            "s32": self.sim32(s32, sem),
        }
