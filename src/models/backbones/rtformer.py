"""RTFormer backbone — faithful to the paper (Wang et al., NeurIPS 2022).

Built from Table 1 (RTFormer-Base) + Fig 2/3/4 + §3.1/3.2:

  stem (2x 3x3, s2)             -> 1/4
  stage1: 2 basic blocks, 64    -> 1/4
  stage2: 2 basic blocks, 128   -> 1/8   (high-res base, kept at 1/8 hereafter)
  stage3: dual, BASIC + DDRNet bilateral fusion   high 128 (1/8) / low 256 (1/16)
  stage4: dual, RTFormer block (stepped)          high 128 (1/8) / low 512 (1/32)
  stage5: dual, RTFormer block (stepped)          high 128 (1/8) / low 512 (1/32)
  DAPPM on low (1/32) -> fuse into high (1/8) -> stride-8 output feature

RTFormer block (Fig 2):
  - low branch  : GFA (single-head, Grouped Double Normalization) + FFN
  - high branch : Cross-resolution Attention (Q=high, K/V pooled from low) + FFN
  - stepped: low is computed first, its output feeds the high branch's CA
  - FFN = two 3x3 conv, NO dimension expansion (§3.1)

Returns {"s4","s8","s16","s32"}; s8 is the DAPPM-fused stride-8 feature that the
(passthrough) neck forwards to the heads. Width defaults to RTFormer-Base.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(cin, cout, k=3, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def conv_bn(cin, cout, k=1, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
                         nn.BatchNorm2d(cout))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, relu_out=True):
        super().__init__()
        self.conv1 = conv_bn_relu(cin, cout, 3, stride)
        self.conv2 = conv_bn(cout, cout, 3, 1)
        self.down = conv_bn(cin, cout, 1, stride) if (stride != 1 or cin != cout) else None
        self.relu_out = relu_out

    def forward(self, x):
        idn = x if self.down is None else self.down(x)
        y = self.conv2(self.conv1(x)) + idn
        return F.relu(y, inplace=True) if self.relu_out else y


class FFN(nn.Module):
    """RTFormer FFN (§3.1): two 3x3 conv layers, NO dimension expansion. Residual."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(conv_bn_relu(dim, dim, 3), conv_bn(dim, dim, 3))

    def forward(self, x):
        return F.relu(x + self.net(x), inplace=True)


class GFA(nn.Module):
    """GPU-Friendly Attention (eq.3): GDN(X·Kg^T)·Vg.

    Single-head, learnable external memory Kg,Vg in R^(Mg x d), Mg = M*H.
    Grouped Double Normalization: softmax over the N tokens, then L1 over memory
    split into H groups. Linear in N (no multi-head matmul split). Residual."""
    def __init__(self, dim, mem=128, groups=8):
        super().__init__()
        assert mem % groups == 0, "mem (Mg) must divide groups (H)"
        self.dim, self.mem, self.groups = dim, mem, groups
        self.per = mem // groups                       # M per group
        self.q = conv_bn(dim, dim, 1)                  # token projection
        self.kg = nn.Parameter(torch.randn(mem, dim) * dim ** -0.5)
        self.vg = nn.Parameter(torch.randn(mem, dim) * mem ** -0.5)
        self.proj = conv_bn(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        n = self.q(x).flatten(2).transpose(1, 2)        # B,N,C
        attn = n @ self.kg.t()                          # B,N,Mg
        attn = attn.softmax(dim=1)                       # 1st norm: softmax over tokens N
        attn = attn.view(B, H * W, self.groups, self.per)
        attn = attn / (attn.sum(dim=3, keepdim=True) + 1e-6)   # 2nd norm: grouped L1 over memory
        attn = attn.view(B, H * W, self.mem)
        out = (attn @ self.vg).transpose(1, 2).reshape(B, C, H, W)
        return F.relu(x + self.proj(out), inplace=True)


class CrossResAttention(nn.Module):
    """Cross-resolution Attention (eq.4): high-res queries attend to K/V pooled
    from the low-res branch. K_c,V_c = phi(X_l) (adaptive-pool to cross_size +
    1x1 conv to high dim). Single softmax, no multi-head. Residual on high."""
    def __init__(self, dim_h, dim_l, cross_size=12):
        super().__init__()
        self.scale = dim_h ** -0.5
        self.cross_size = cross_size
        self.q = conv_bn(dim_h, dim_h, 1)
        self.k = conv_bn(dim_l, dim_h, 1)
        self.v = conv_bn(dim_l, dim_h, 1)
        self.proj = conv_bn(dim_h, dim_h, 1)

    def forward(self, x_h, x_l):
        B, C, H, W = x_h.shape
        # fixed cross-feature grid (paper: 12x12). interpolate (not adaptive_avg_pool)
        # so it exports to ONNX/Hailo as a static Resize at any input resolution.
        lp = F.interpolate(x_l, size=(self.cross_size, self.cross_size),
                           mode="bilinear", align_corners=False)     # B,Cl,s,s
        q = self.q(x_h).flatten(2).transpose(1, 2)                  # B,Nh,C
        k = self.k(lp).flatten(2).transpose(1, 2)                   # B,S,C
        v = self.v(lp).flatten(2).transpose(1, 2)                   # B,S,C
        attn = (q @ k.transpose(1, 2)) * self.scale                # B,Nh,S
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, C, H, W)       # B,C,H,W
        return F.relu(x_h + self.proj(out), inplace=True)


class RTFormerStage(nn.Module):
    """Deep dual-resolution stage (stage 4/5): one RTFormer block, stepped layout.
    low: (optional strided-conv transition) + GFA + FFN ; high: stays 1/8,
    CrossResAttention(K/V from low) + FFN. stride=2 only at the stage4 transition
    (1/16->1/32); stage5 keeps 1/32 (stride=1, no channel change -> identity)."""
    def __init__(self, c_high, c_low_in, c_low_out, stride, mem, groups, cross_size):
        super().__init__()
        self.low_trans = (conv_bn_relu(c_low_in, c_low_out, 3, stride)
                          if (stride != 1 or c_low_in != c_low_out) else nn.Identity())
        self.gfa = GFA(c_low_out, mem=mem, groups=groups)
        self.ffn_low = FFN(c_low_out)
        self.ca = CrossResAttention(c_high, c_low_out, cross_size=cross_size)
        self.ffn_high = FFN(c_high)

    def forward(self, high, low):
        low = self.ffn_low(self.gfa(self.low_trans(low)))   # low branch first (stepped)
        high = self.ffn_high(self.ca(high, low))            # high enhanced by low via CA
        return high, low


class DAPPM(nn.Module):
    """Deep Aggregation Pyramid Pooling Module (DDRNet), used by RTFormer's seg
    head for multi-scale context. cout chosen to match the high branch for fusion."""
    def __init__(self, cin, cmid, cout):
        super().__init__()
        self.s0 = conv_bn_relu(cin, cmid, 1)
        self.s1 = nn.Sequential(nn.AvgPool2d(5, 2, 2), conv_bn_relu(cin, cmid, 1))
        self.s2 = nn.Sequential(nn.AvgPool2d(9, 4, 4), conv_bn_relu(cin, cmid, 1))
        self.s3 = nn.Sequential(nn.AvgPool2d(17, 8, 8), conv_bn_relu(cin, cmid, 1))
        self.s4 = nn.Sequential(nn.AdaptiveAvgPool2d(1), conv_bn_relu(cin, cmid, 1))
        self.p1 = conv_bn_relu(cmid, cmid, 3)
        self.p2 = conv_bn_relu(cmid, cmid, 3)
        self.p3 = conv_bn_relu(cmid, cmid, 3)
        self.p4 = conv_bn_relu(cmid, cmid, 3)
        self.compress = conv_bn_relu(cmid * 5, cout, 1)
        self.short = conv_bn(cin, cout, 1)

    def forward(self, x):
        sz = x.shape[-2:]
        up = lambda y: F.interpolate(y, size=sz, mode="bilinear", align_corners=False)
        x0 = self.s0(x)
        x1 = self.p1(up(self.s1(x)) + x0)
        x2 = self.p2(up(self.s2(x)) + x1)
        x3 = self.p3(up(self.s3(x)) + x2)
        x4 = self.p4(up(self.s4(x)) + x3)
        return self.compress(torch.cat([x0, x1, x2, x3, x4], 1)) + self.short(x)


class RTFormer(nn.Module):
    def __init__(self, in_ch=3, channels=(64, 128, 256, 512), mem=128, groups=8, cross_size=12):
        super().__init__()
        c1, c2, c3, c4 = channels        # stage1=64, stage2/high=128, low3=256, low45=512
        # stem -> 1/4, c1
        self.stem = nn.Sequential(conv_bn_relu(in_ch, c1, 3, 2), conv_bn_relu(c1, c1, 3, 2))
        self.stage1 = nn.Sequential(BasicBlock(c1, c1), BasicBlock(c1, c1))            # 1/4, 64
        self.stage2 = nn.Sequential(BasicBlock(c1, c2, stride=2), BasicBlock(c2, c2))  # 1/8, 128 (high base)

        # stage3: dual-resolution, BASIC blocks + DDRNet bilateral fusion
        self.s3_high = BasicBlock(c2, c2)                          # 1/8, 128 (1 block)
        self.s3_low = nn.Sequential(BasicBlock(c2, c3, stride=2), BasicBlock(c3, c3))  # 1/16, 256 (2 blocks)
        self.s3_l2h = conv_bn(c3, c2, 1)                           # low->high (upsample-add)
        self.s3_h2l = conv_bn(c2, c3, 3, 2)                        # high->low (strided-add)

        # stage4 / stage5: RTFormer blocks (stepped). stage4 low 1/16->1/32; stage5 stays 1/32
        self.stage4 = RTFormerStage(c2, c3, c4, 2, mem, groups, cross_size)   # low 256->512 @1/32
        self.stage5 = RTFormerStage(c2, c4, c4, 1, mem, groups, cross_size)   # low 512->512 @1/32

        # seg-head context: DAPPM on low(1/32) -> fuse into high(1/8)
        self.dappm = DAPPM(c4, c4 // 4, c2)                        # cout=c2 to add to high
        self.out_channels = {"s4": c1, "s8": c2, "s16": c3, "s32": c4}

    def load_pretrained(self, path):
        """Load an ImageNet-pretrained backbone state_dict (from tools/pretrain_imagenet.py).
        Accepts a raw RTFormer state_dict or a {'backbone': ...} wrapper. strict=False
        so a classifier head / untrained DAPPM don't block the load."""
        # Pretraining exports contain only tensors, so keep this safe under
        # PyTorch versions whose default torch.load policy is weights_only=True.
        sd = torch.load(path, map_location="cpu", weights_only=True)
        sd = sd.get("backbone", sd.get("model", sd))
        sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v for k, v in sd.items()}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"[rtformer] loaded pretrained from {path}; "
              f"missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"[rtformer] WARN missing (e.g.): {missing[:4]}")
        return missing, unexpected

    def _up(self, y, ref):
        return F.interpolate(y, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward_sender(self, x):
        """Send-side of the EARLY split: pure-conv stem + stage1 + stage2 (no
        attention, Hailo-foldable). Returns (s4, high):
          s4   (1/4, c1) -- high-res scale the head later needs (it is produced
                 before stage2, so it must reach the receiver -> 'high-res bypass').
          high (1/8, c2) -- the stage2 feature = the early split point, just BEFORE
                 the dual-resolution fork. This is the tensor to compress."""
        s4 = self.stage1(self.stem(x))                # 1/4, c1
        high = self.stage2(s4)                        # 1/8, c2  (early split feature)
        return s4, high

    def _stage3(self, high):
        """Stage3 dual-resolution BASIC blocks + DDRNet bilateral fusion. Input = the
        stage2 high feature (1/8, c2); returns the fused dual pair
          (x_h 1/8 c2, x_l 1/16 c3)
        = the conv->attention boundary (the MID split point, paper-faithful: this is
        exactly the (x3_, x3) payload of phase2 spec §3.1, no s4)."""
        h3 = self.s3_high(high)
        l3 = self.s3_low(high)
        x_h = F.relu(h3 + self._up(self.s3_l2h(l3), h3), inplace=True)   # 1/8, c2
        x_l = F.relu(l3 + self.s3_h2l(h3), inplace=True)                 # 1/16, c3
        return x_h, x_l

    def _stage45(self, x_h, x_l):
        """Stage4/5 RTFormer blocks (attention) from the stage3 pair. Returns
        (high, low) = (1/8 c2, 1/32 c4), BEFORE the DAPPM seg-head context (= the
        LATE split point payload (x5_, x5))."""
        high, low = self.stage4(x_h, x_l)             # low -> 1/32, c4
        high, low = self.stage5(high, low)            # low stays 1/32, c4
        return high, low

    def _receiver_trunk(self, s4, high):
        """Receive-side trunk from the stage2 feature: stage3 (dual + bilateral
        fusion) + stage4/5. Returns (s4, high_pre_dappm, s16, s32). The old in-line
        trunk now lives here so forward()/forward_trunk()/forward_receiver() share it
        verbatim (behaviour-identical)."""
        x_h, x_l = self._stage3(high)
        s16 = x_l                                     # 1/16, c3
        high, low = self._stage45(x_h, x_l)
        return s4, high, s16, low

    def _features(self, x):
        """Trunk (stem -> stage5), shared by forward() and forward_trunk().
        Composes forward_sender + _receiver_trunk. Returns (s4, high, s16, s32)."""
        return self._receiver_trunk(*self.forward_sender(x))

    def forward_receiver(self, s4, high):
        """Receive-side of the EARLY split: run the rest of the backbone from the
        (reconstructed) stage2 `high` feature + the bypassed `s4`. DAPPM context
        fused into the stride-8 feature. Returns the {s4,s8,s16,s32} contract."""
        s4, high, s16, s32 = self._receiver_trunk(s4, high)
        high = F.relu(high + self._up(self.dappm(s32), high), inplace=True)
        return {"s4": s4, "s8": high, "s16": s16, "s32": s32}

    # ---- MID split (phase2 main line): cut after stage3, payload (x_h, x_l) ----
    def forward_sender_mid(self, x):
        """Send-side of the MID split: pure-conv stem+stage1+stage2+stage3
        (Hailo-foldable, NO attention). Returns the dual payload that crosses the
        channel: (x_h 1/8 c2, x_l 1/16 c3) = phase2 spec §3.1 (x3_, x3). NO s4 needs
        to cross — the receiver tail is the paper head (passthrough(s8)+DAPPM)."""
        high = self.forward_sender(x)[1]              # pure conv up to stage2 (s4 not needed at mid)
        return self._stage3(high)                     # (x_h, x_l)

    def forward_receiver_mid(self, x_h, x_l):
        """Receive-side of the MID split: stage4/5 (attention) + DAPPM context fused
        into the stride-8 feature, from the (reconstructed) stage3 pair. s4 is None
        (produced pre-stage2 on the sender; the paper passthrough(s8) head ignores it)."""
        high, low = self._stage45(x_h, x_l)
        high = F.relu(high + self._up(self.dappm(low), high), inplace=True)
        return {"s4": None, "s8": high, "s16": x_l, "s32": low}

    # ---- LATE split (deep ablation): cut after stage5, payload (x5_, x5) ----
    def forward_sender_late(self, x):
        """Send-side of the LATE split: the whole backbone trunk through stage5
        (so the attention blocks run on the SENDER — NOT deployable, ablation only).
        Returns (high 1/8 c2, low 1/32 c4) = phase2 spec §3.1 (x5_, x5), pre-DAPPM."""
        x_h, x_l = self.forward_sender_mid(x)
        return self._stage45(x_h, x_l)

    def forward_receiver_late(self, x_h, x_l):
        """Receive-side of the LATE split: only the DAPPM seg-head context + fuse.
        s4/s16 are None (the passthrough(s8) head ignores them)."""
        high = F.relu(x_h + self._up(self.dappm(x_l), x_h), inplace=True)
        return {"s4": None, "s8": high, "s16": None, "s32": x_l}

    def forward(self, x):
        s4, high, s16, s32 = self._features(x)
        # DAPPM context fused into the high (stride-8) feature -> seg-head input
        high = F.relu(high + self._up(self.dappm(s32), high), inplace=True)
        return {"s4": s4, "s8": high, "s16": s16, "s32": s32}

    def forward_trunk(self, x):
        """For ImageNet pretraining: returns the s32 trunk feature (1/32, c4),
        WITHOUT the DAPPM seg-head module (left untrained, fine-tuned during seg)."""
        return self._features(x)[3]
