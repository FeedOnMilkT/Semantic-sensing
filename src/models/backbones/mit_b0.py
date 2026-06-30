"""MiclearT-B0 (SegFormer hierarchical encoder) — the safe lower-bound baseline.

Faithful to the SegFormer MiT design: overlap patch embedding, efficient
self-attention with spatial reduction (SR), and Mix-FFN (3x3 depthwise conv
replacing positional encoding). 4 stages -> 4 scales.

forward(img) -> {"s4","s8","s16","s32"} per the fixed backbone contract.
Note: trained from scratch here (no ImageNet weights available offline); for
hitting the mIoU bar you'd load official MiT-B0 pretrain into this module.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch, stride, in_ch, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch, stride, padding=patch // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        H, W = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        return self.norm(x), H, W


class EfficientAttention(nn.Module):
    def __init__(self, dim, heads, sr_ratio):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).reshape(B, C, H, W)
            x_ = self.sr(x_).flatten(2).transpose(1, 2)
            x_ = self.norm(x_)
        else:
            x_ = x
        kv = self.kv(x_).reshape(B, -1, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class MixFFN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def forward(self, x, H, W):
        x = self.fc1(x)
        B, N, C = x.shape
        x = self.dwconv(x.transpose(1, 2).view(B, C, H, W)).flatten(2).transpose(1, 2)
        x = self.act(x)
        return self.fc2(x)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, sr_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, dim * mlp_ratio)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x


def _remap_hf_segformer(hf):
    """Map an HF `nvidia/mit-b0` encoder state_dict onto this module's names.
    HF uses separate key/value Linears + `segformer.encoder.*` naming; we use a
    merged kv Linear + embeds/stages/norms. Drops the ImageNet classifier."""
    import torch
    out, kv = {}, {}
    for k, w in hf.items():
        if not k.startswith("segformer.encoder."):
            continue                                   # skip classifier.*
        k2 = k[len("segformer.encoder."):]
        if k2.startswith("patch_embeddings."):
            i, sub = k2[len("patch_embeddings."):].split(".", 1)
            out[f"embeds.{i}.{sub.replace('layer_norm', 'norm')}"] = w
        elif k2.startswith("layer_norm."):             # per-stage final norms
            i, sub = k2[len("layer_norm."):].split(".", 1)
            out[f"norms.{i}.{sub}"] = w
        elif k2.startswith("block."):
            i, j, sub = k2[len("block."):].split(".", 2)
            p, last = f"stages.{i}.{j}.", sub.rsplit(".", 1)[1]
            if sub.startswith("layer_norm_1"):   out[p + "norm1." + last] = w
            elif sub.startswith("layer_norm_2"): out[p + "norm2." + last] = w
            elif sub.startswith("attention.self.query"):     out[p + "attn.q." + last] = w
            elif sub.startswith("attention.self.key"):       kv[(i, j, "k", last)] = w
            elif sub.startswith("attention.self.value"):     kv[(i, j, "v", last)] = w
            elif sub.startswith("attention.self.sr"):        out[p + "attn.sr." + last] = w
            elif sub.startswith("attention.self.layer_norm"): out[p + "attn.norm." + last] = w
            elif sub.startswith("attention.output.dense"):   out[p + "attn.proj." + last] = w
            elif sub.startswith("mlp.dense1"):    out[p + "mlp.fc1." + last] = w
            elif sub.startswith("mlp.dwconv.dwconv"): out[p + "mlp.dwconv." + last] = w
            elif sub.startswith("mlp.dense2"):    out[p + "mlp.fc2." + last] = w
    for (i, j) in {(a, b) for (a, b, _, _) in kv}:
        for param in ("weight", "bias"):
            out[f"stages.{i}.{j}.attn.kv.{param}"] = torch.cat(
                [kv[(i, j, "k", param)], kv[(i, j, "v", param)]], dim=0)
    return out


class MiTB0(nn.Module):
    out_channels = {"s4": 32, "s8": 64, "s16": 160, "s32": 256}

    def __init__(self, in_ch=3, embed_dims=(32, 64, 160, 256), heads=(1, 2, 5, 8),
                 mlp_ratios=(4, 4, 4, 4), depths=(2, 2, 2, 2), sr_ratios=(8, 4, 2, 1)):
        super().__init__()
        patches = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        in_dims = [in_ch] + list(embed_dims[:-1])
        self.stages = nn.ModuleList()
        self.embeds = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(4):
            self.embeds.append(OverlapPatchEmbed(patches[i], strides[i], in_dims[i], embed_dims[i]))
            self.stages.append(nn.ModuleList(
                [Block(embed_dims[i], heads[i], mlp_ratios[i], sr_ratios[i]) for _ in range(depths[i])]))
            self.norms.append(nn.LayerNorm(embed_dims[i]))
        self.out_channels = {f"s{4 * 2 ** i}": embed_dims[i] for i in range(4)}

    def load_pretrained(self, path):
        import torch
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("state_dict", sd)
        remapped = _remap_hf_segformer(sd) if any(k.startswith("segformer.encoder") for k in sd) else sd
        missing, unexpected = self.load_state_dict(remapped, strict=False)
        print(f"[mit_b0] loaded {len(remapped)} tensors from {path}; "
              f"missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"[mit_b0] WARN missing (e.g.): {missing[:4]}")
        return missing, unexpected

    def forward(self, x):
        feats = {}
        for i in range(4):
            x, H, W = self.embeds[i](x)
            for blk in self.stages[i]:
                x = blk(x, H, W)
            x = self.norms[i](x)
            x = x.transpose(1, 2).reshape(x.shape[0], -1, H, W).contiguous()
            feats[f"s{4 * 2 ** i}"] = x
        return feats
