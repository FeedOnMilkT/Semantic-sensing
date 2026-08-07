"""Split-inference student: a frozen teacher (MultiTaskNet) with a trainable
communication bottleneck spliced in at a configurable split point.

Four splits (config `student.split`), reusing the SAME teacher + bottleneck recipe —
only the tap point and the receiver modules differ (a controlled split-point ablation,
phase2 spec §3.1). EARLY is single-branch; MID/LATE are dual-branch (the conv->attention
cut emits two feature streams), so they use the dual-latent / single-container bottleneck.

  - "early": stage2 feature (pre dual-resolution fork; sender stem+stage1+stage2 is pure
             conv -> Hailo-foldable). SINGLE latent. The high-res s4 scale also crosses:
               * compress_s4: true  -> a SECOND bottleneck compresses s4, bits counted.
               * compress_s4: false -> s4 bypasses uncompressed (bpp understated, flagged).
  - "mid"  : ★ MAIN LINE. cut AFTER stage3 (conv->attention boundary). DUAL payload
             (x_h 1/8 c2, x_l 1/16 c3) = spec (x3_, x3). NO s4 (paper passthrough(s8) tail).
  - "late" : cut AFTER stage5 (attention on the SENDER -> NOT deployable, ablation only).
             DUAL payload (x_h 1/8 c2, x_l 1/32 c4) = spec (x5_, x5).
  - "neck" : legacy de-risk split at the neck output (single fused tensor); kept for the
             older split smoke. Not a phase2 main/ablation arm.

The teacher is FROZEN: it supplies the distillation target feature(s) at the split and
the frozen tail that turns the reconstruction into task predictions. Stage 1 trains only
the bottleneck. Stage 2 (spec §5.2) freezes the bottleneck and fine-tunes the RECEIVER
TAIL (the teacher modules downstream of the cut) with the task loss — see
`set_stage2_trainable()` / `receiver_parameters()`.

forward(x) -> {"preds","target","recon","info"[, "target_aux","recon_aux"]}:
  target/recon : the split feature(s) + reconstruction(s). For mid/late these are
                 (x_h, x_l) tuples; the trainer's feature_mse sums over branches.
  info         : rate info; info["likelihoods"] sums ALL transmitted tensors.
"""
import copy

import torch.nn as nn
import torch.nn.functional as F

from .bottlenecks import build_bottleneck

# backbone submodules on the SENDER side of each split (everything else + neck/decoder/
# heads is the receiver tail that Stage 2 fine-tunes). Order matches rtformer.py forward.
_ALL_BB = ("stem", "stage1", "stage2", "s3_high", "s3_low", "s3_l2h", "s3_h2l",
           "stage4", "stage5", "dappm")
_SENDER_BB = {
    "early": ("stem", "stage1", "stage2"),
    "mid":   ("stem", "stage1", "stage2", "s3_high", "s3_low", "s3_l2h", "s3_h2l"),
    "late":  ("stem", "stage1", "stage2", "s3_high", "s3_low", "s3_l2h", "s3_h2l",
              "stage4", "stage5"),
}


class SplitStudent(nn.Module):
    def __init__(self, teacher, bottleneck, split, s4_bottleneck=None, unfreeze_sender=False):
        super().__init__()
        assert split in ("early", "mid", "late", "neck"), f"unknown split '{split}'"
        self.split = split
        self.teacher = teacher
        self.bottleneck = bottleneck
        self.s4_bottleneck = s4_bottleneck          # early + compress_s4: compresses the s4 bypass
        self.unfreeze_sender = unfreeze_sender
        self._sender_ref = None                       # frozen teacher snapshot for distillation targets
        if unfreeze_sender:
            self._build_sender_ref()
        self._freeze_teacher()

    def _freeze_teacher(self):
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def _build_sender_ref(self):
        """Shallow-copy the sender backbone submodules at init; frozen forever so Stage-1
        feature-MSE targets stay anchored to the original teacher even as the live sender
        drifts (Entropic-Student-style unfreeze without a moving target)."""
        bb = self.teacher.backbone
        names = _SENDER_BB.get(self.split, ())
        ref = nn.ModuleDict({n: copy.deepcopy(getattr(bb, n)) for n in names})
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        self._sender_ref = ref

    def train(self, mode=True):
        super().train(mode)
        if not self.unfreeze_sender:
            self.teacher.eval()                      # Stage 1 default: only bottleneck trains
            return self
        for m in self.receiver_modules():
            m.eval()                                 # receiver tail stays frozen BN stats
        for m in self.sender_modules():
            m.train(mode)
        for bn in self._bns():
            bn.train(mode)
        return self

    # ---- split routing ----
    def _sender(self, x):
        bb = self.teacher.backbone
        if self.split == "neck":
            return self.teacher.neck(bb(x)), {}
        if self.split == "early":
            s4, high = bb.forward_sender(x)          # pure-conv sender
            return high, {"s4": s4}
        if self.split == "mid":
            return bb.forward_sender_mid(x), {}      # (x_h, x_l)
        return bb.forward_sender_late(x), {}         # late: (x_h, x_l)

    def _sender_ref_forward(self, x):
        """Frozen teacher sender path (uses self._sender_ref, never the trainable copy)."""
        ref = self._sender_ref
        if self.split == "neck":
            return self.teacher.neck(self.teacher.backbone(x))
        s4 = ref["stage1"](ref["stem"](x))
        high = ref["stage2"](s4)
        if self.split == "early":
            return high
        h3 = ref["s3_high"](high)
        l3 = ref["s3_low"](high)
        x_h = F.relu(h3 + F.interpolate(ref["s3_l2h"](l3), size=h3.shape[-2:],
                                        mode="bilinear", align_corners=False), inplace=True)
        x_l = F.relu(l3 + ref["s3_h2l"](h3), inplace=True)
        if self.split == "mid":
            return x_h, x_l
        high, low = ref["stage4"](x_h, x_l)
        return ref["stage5"](high, low)

    def _teacher_split_target(self, x):
        """Distillation target at the split. When the live sender is trainable, targets
        must NOT come from it — use the frozen init-time reference instead."""
        if self.unfreeze_sender:
            return self._sender_ref_forward(x)
        feat, _ = self._sender(x)
        return feat

    def _receiver(self, feat, ctx):
        t = self.teacher
        if self.split == "neck":
            d = t.decoder(feat)
        elif self.split == "early":
            feats = t.backbone.forward_receiver(ctx["s4"], feat)
            d = t.decoder(t.neck(feats))
        elif self.split == "mid":
            feats = t.backbone.forward_receiver_mid(*feat)   # feat = (x_h_hat, x_l_hat)
            d = t.decoder(t.neck(feats))
        else:  # late
            feats = t.backbone.forward_receiver_late(*feat)
            d = t.decoder(t.neck(feats))
        return {k: t.heads[k](d) for k in t.heads}

    def forward(self, x, tasks=None):
        split_feat, ctx = self._sender(x)
        recon, info = self.bottleneck(split_feat)
        out = {"target": self._teacher_split_target(x), "recon": recon}
        if self.s4_bottleneck is not None and self.split == "early":
            s4_hat, info_s4 = self.s4_bottleneck(ctx["s4"])
            out["target_aux"], out["recon_aux"] = ctx["s4"], s4_hat
            liks = dict(info["likelihoods"])                      # merge so bpp counts s4 too
            for k, v in info_s4["likelihoods"].items():
                liks[f"{k}_s4"] = v
            info = {"likelihoods": liks}
            ctx = {"s4": s4_hat}                                  # receiver uses the reconstructed s4
        out["info"] = info
        out["preds"] = self._receiver(recon, ctx)
        return out

    # ---- deployment path (real bitstream; mid/late = single dual-latent container) ----
    def compress(self, x):
        split_feat, ctx = self._sender(x)
        payload = {"main": self.bottleneck.compress(split_feat)}
        if self.s4_bottleneck is not None and self.split == "early":
            payload["s4"] = self.s4_bottleneck.compress(ctx["s4"])
            return payload, {}
        return payload, ctx

    def decompress(self, payload, ctx):
        recon = self.bottleneck.decompress(payload["main"])
        if "s4" in payload:
            ctx = {"s4": self.s4_bottleneck.decompress(payload["s4"])}
        return self._receiver(recon, ctx)

    # ---- Stage 1 optimiser plumbing (delegate to bottleneck(s)) ----
    def _bns(self):
        return [self.bottleneck] + ([self.s4_bottleneck] if self.s4_bottleneck is not None else [])

    def codec_parameters(self):
        return [p for bn in self._bns() for p in bn.codec_parameters()]

    def aux_parameters(self):
        return [p for bn in self._bns() for p in bn.aux_parameters()]

    def aux_loss(self):
        return sum(bn.aux_loss() for bn in self._bns())

    def update(self, force=True):
        return all(bn.update(force=force) for bn in self._bns())

    # ---- Stage 1 sender (optional unfreeze, workflow §9.5) ----
    def sender_modules(self):
        """Teacher modules on the SEND side of the cut (= the encoder Stage 1 may unfreeze)."""
        if self.split == "neck":
            return [self.teacher.backbone, self.teacher.neck]
        return [getattr(self.teacher.backbone, n) for n in _SENDER_BB[self.split]]

    def sender_parameters(self):
        seen, out = set(), []
        for m in self.sender_modules():
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)
        return out

    def stage1_trainable_parameters(self):
        params = list(self.codec_parameters())
        if self.unfreeze_sender:
            params.extend(self.sender_parameters())
        return params

    def set_stage1_trainable(self, unfreeze_sender=None):
        """Stage 1: train bottleneck (+ optional sender); receiver tail stays frozen."""
        if unfreeze_sender is not None:
            self.unfreeze_sender = unfreeze_sender
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        for bn in self._bns():
            for p in bn.parameters():
                p.requires_grad_(True)
        if self.unfreeze_sender:
            for p in self.sender_parameters():
                p.requires_grad_(True)

    # ---- Stage 2: fine-tune the receiver tail (spec §5.2) ----
    def receiver_modules(self):
        """The teacher modules DOWNSTREAM of the cut (= the receiver tail). Stage 2
        unfreezes and fine-tunes exactly these; the sender encoder + bottleneck stay
        frozen so the deployed send-side (and its entropy model) is fixed."""
        t = self.teacher
        if self.split == "neck":
            mods = [t.decoder]
        else:
            sender = _SENDER_BB[self.split]
            mods = [getattr(t.backbone, n) for n in _ALL_BB if n not in sender]
            mods += [t.neck, t.decoder]
        mods += list(t.heads.values())
        return mods

    def receiver_parameters(self):
        seen, out = set(), []
        for m in self.receiver_modules():
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)
        return out

    def set_stage2_trainable(self):
        """Freeze bottleneck(s) + sender; unfreeze the receiver tail. Run the Stage-2
        loop with the student in eval() mode (frozen BN running stats; only the tail
        conv/linear weights update)."""
        for bn in self._bns():
            for p in bn.parameters():
                p.requires_grad_(False)
        for p in self.receiver_parameters():
            p.requires_grad_(True)


def split_in_channels(teacher, split):
    """Channels of the feature(s) that cross the split (drives the bottleneck width).
    Single int for early/neck; (c_high, c_low) tuple for the dual mid/late payload."""
    oc = teacher.backbone.out_channels               # {s4:c1, s8:c2, s16:c3, s32:c4}
    if split == "neck":
        return teacher.neck.out_channels
    if split == "early":
        return oc["s8"]                              # stage2 feature channels (c2)
    if split == "mid":
        return (oc["s8"], oc["s16"])                 # (x_h c2=128, x_l c3=256)
    return (oc["s8"], oc["s32"])                     # late: (x_h c2=128, x_l c4=512)


def build_student(cfg, teacher):
    split = cfg.student.split
    unfreeze_sender = bool(cfg.student.get("unfreeze_sender", False))
    bottleneck = build_bottleneck(cfg.bottleneck, in_channels=split_in_channels(teacher, split))
    s4_bottleneck = None
    if split == "early" and cfg.student.get("compress_s4", False):
        # same bottleneck recipe, sized for the s4 scale (c1 channels, 1/4 res)
        s4_bottleneck = build_bottleneck(cfg.bottleneck, in_channels=teacher.backbone.out_channels["s4"])
    student = SplitStudent(teacher, bottleneck, split=split, s4_bottleneck=s4_bottleneck,
                           unfreeze_sender=unfreeze_sender)
    student.set_stage1_trainable(unfreeze_sender)
    return student
