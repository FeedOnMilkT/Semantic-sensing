"""Split-inference student: a frozen teacher (MultiTaskNet) with a trainable
communication bottleneck spliced in at a configurable split point.

Two splits (config `student.split`), reusing the SAME teacher and bottleneck code —
only the tap point and the receiver modules differ (a controlled split-point ablation):

  - "late"  : split at the neck output (semantic waist, single fused tensor).
              sender = backbone + neck ; receiver = decoder + heads.
  - "early" : split at the backbone's stage2 feature (pre dual-resolution fork; the
              sender stem+stage1+stage2 is pure conv -> Hailo-foldable). receiver =
              stage3..5 + DAPPM + neck + decoder + heads. The high-res s4 scale (needed
              by the head but produced before stage2) BYPASSES the bottleneck — only
              the semantic stage2 feature is compressed (instruction open-Q#1 "高分旁路").
              NOTE: the s4 bypass is not yet rate-counted; compressing it too (multi-scale
              bottleneck) is a later refinement, flagged so the early bpp isn't understated
              silently.

The teacher is FROZEN: it supplies both the distillation target feature at the split
and the frozen tail that turns the reconstructed feature into task predictions. Only the
bottleneck trains, isolating the split point as the single variable.

forward(x) -> {"preds", "target", "recon", "info"}:
  preds  : task predictions via the frozen receiver on the reconstructed feature
  target : teacher feature at the split point (distillation target)
  recon  : the bottleneck's reconstruction of `target`
  info   : bottleneck rate info (likelihoods) or None
"""
import torch.nn as nn

from .bottlenecks import build_bottleneck


class SplitStudent(nn.Module):
    def __init__(self, teacher, bottleneck, split):
        super().__init__()
        assert split in ("early", "late"), f"unknown split '{split}'"
        self.split = split
        self.teacher = teacher
        self.bottleneck = bottleneck
        self._freeze_teacher()

    def _freeze_teacher(self):
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        # only the bottleneck trains; keep the frozen teacher in eval (BN stats fixed)
        super().train(mode)
        self.teacher.eval()
        return self

    # ---- split routing ----
    def _sender(self, x):
        """Returns (split_feat, ctx). ctx carries side features the receiver needs."""
        t = self.teacher
        if self.split == "late":
            return t.neck(t.backbone(x)), {}
        s4, high = t.backbone.forward_sender(x)            # early: pure-conv sender
        return high, {"s4": s4}                            # s4 bypasses the bottleneck

    def _receiver(self, feat, ctx):
        t = self.teacher
        if self.split == "late":
            d = t.decoder(feat)
        else:
            feats = t.backbone.forward_receiver(ctx["s4"], feat)   # s4 bypass + recon stage2
            d = t.decoder(t.neck(feats))
        return {k: t.heads[k](d) for k in t.heads}

    def forward(self, x, tasks=None):
        split_feat, ctx = self._sender(x)
        recon, info = self.bottleneck(split_feat)
        preds = self._receiver(recon, ctx)
        return {"preds": preds, "target": split_feat, "recon": recon, "info": info}

    # ---- deployment path (real bitstream) ----
    def compress(self, x):
        split_feat, ctx = self._sender(x)
        return self.bottleneck.compress(split_feat), ctx

    def decompress(self, payload, ctx):
        recon = self.bottleneck.decompress(payload)
        return self._receiver(recon, ctx)

    # ---- optimiser plumbing (delegate to the bottleneck) ----
    def codec_parameters(self):
        return self.bottleneck.codec_parameters()

    def aux_parameters(self):
        return self.bottleneck.aux_parameters()

    def aux_loss(self):
        return self.bottleneck.aux_loss()

    def update(self, force=True):
        return self.bottleneck.update(force=force)


def split_in_channels(teacher, split):
    """Channels of the feature that crosses the split (drives the bottleneck width)."""
    if split == "late":
        return teacher.neck.out_channels
    return teacher.backbone.out_channels["s8"]   # stage2 feature channels (c2)


def build_student(cfg, teacher):
    split = cfg.student.split
    bottleneck = build_bottleneck(cfg.bottleneck, in_channels=split_in_channels(teacher, split))
    return SplitStudent(teacher, bottleneck, split=split)
