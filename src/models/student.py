"""Split-inference student: a frozen teacher (MultiTaskNet) with a trainable
communication bottleneck spliced in at a configurable split point.

Two splits (config `student.split`), reusing the SAME teacher and bottleneck code —
only the tap point and the receiver modules differ (a controlled split-point ablation):

  - "late"  : split at the neck output (semantic waist, single fused tensor).
              sender = backbone + neck ; receiver = decoder + heads.
  - "early" : split at the backbone's stage2 feature (pre dual-resolution fork; the
              sender stem+stage1+stage2 is pure conv -> Hailo-foldable). receiver =
              stage3..5 + DAPPM + neck + decoder + heads. The high-res s4 scale (needed
              by the head but produced before stage2) ALSO crosses the channel:
                * student.compress_s4: true  -> a SECOND bottleneck compresses s4 and its
                  bits count toward the rate (fair bpp vs late; item ③).
                * student.compress_s4: false -> s4 BYPASSES the bottleneck uncompressed
                  (its bits are NOT counted -> early bpp is understated; flagged).

The teacher is FROZEN: it supplies the distillation target feature(s) at the split and
the frozen tail that turns the reconstruction into task predictions. Only the
bottleneck(s) train, isolating the split point as the variable.

forward(x) -> {"preds","target","recon","info"[, "target_aux","recon_aux"]}:
  preds  : task predictions via the frozen receiver on the reconstructed feature(s)
  target/recon         : main split feature + its reconstruction
  target_aux/recon_aux : (early + compress_s4) the s4 feature + its reconstruction
  info   : rate info; info["likelihoods"] sums ALL transmitted tensors (main [+ s4])
"""
import torch.nn as nn

from .bottlenecks import build_bottleneck


class SplitStudent(nn.Module):
    def __init__(self, teacher, bottleneck, split, s4_bottleneck=None):
        super().__init__()
        assert split in ("early", "late"), f"unknown split '{split}'"
        self.split = split
        self.teacher = teacher
        self.bottleneck = bottleneck
        self.s4_bottleneck = s4_bottleneck          # early + compress_s4: compresses the s4 bypass
        self._freeze_teacher()

    def _freeze_teacher(self):
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()                          # only the bottleneck(s) train
        return self

    # ---- split routing ----
    def _sender(self, x):
        t = self.teacher
        if self.split == "late":
            return t.neck(t.backbone(x)), {}
        s4, high = t.backbone.forward_sender(x)      # early: pure-conv sender
        return high, {"s4": s4}

    def _receiver(self, feat, ctx):
        t = self.teacher
        if self.split == "late":
            d = t.decoder(feat)
        else:
            feats = t.backbone.forward_receiver(ctx["s4"], feat)
            d = t.decoder(t.neck(feats))
        return {k: t.heads[k](d) for k in t.heads}

    def forward(self, x, tasks=None):
        split_feat, ctx = self._sender(x)
        recon, info = self.bottleneck(split_feat)
        out = {"target": split_feat, "recon": recon}
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

    # ---- deployment path (real bitstream) ----
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

    # ---- optimiser plumbing (delegate to bottleneck(s)) ----
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


def split_in_channels(teacher, split):
    """Channels of the main feature that crosses the split (drives the bottleneck width)."""
    if split == "late":
        return teacher.neck.out_channels
    return teacher.backbone.out_channels["s8"]       # stage2 feature channels (c2)


def build_student(cfg, teacher):
    split = cfg.student.split
    bottleneck = build_bottleneck(cfg.bottleneck, in_channels=split_in_channels(teacher, split))
    s4_bottleneck = None
    if split == "early" and cfg.student.get("compress_s4", False):
        # same bottleneck recipe, sized for the s4 scale (c1 channels, 1/4 res)
        s4_bottleneck = build_bottleneck(cfg.bottleneck, in_channels=teacher.backbone.out_channels["s4"])
    return SplitStudent(teacher, bottleneck, split=split, s4_bottleneck=s4_bottleneck)
