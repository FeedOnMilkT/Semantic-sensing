"""Evaluator: runs the model over a loader and reports per-task metrics, only for
tasks present in the data (conditional)."""
import torch
import torch.nn.functional as F

from ..utils.metrics import build_meter


@torch.no_grad()
def slide_inference(model, img, task, num_classes, crop=512, stride=341):
    """mmseg-style sliding-window seg inference for a single image at native
    resolution. Pads up so every window is exactly crop x crop, slides with the
    given stride, accumulates per-window logits into a full-res canvas with an
    overlap count, averages, and crops back to the original size. Returns
    (1, num_classes, H, W) logits. The model always sees crop x crop inputs, so
    arbitrary (non-stride-aligned) image sizes are fine."""
    _, _, H, W = img.shape
    padH, padW = max(0, crop - H), max(0, crop - W)
    if padH or padW:
        img = F.pad(img, (0, padW, 0, padH))            # pad right/bottom up to crop
    _, _, Hp, Wp = img.shape
    h_grids = max((Hp - crop + stride - 1) // stride + 1, 1)
    w_grids = max((Wp - crop + stride - 1) // stride + 1, 1)
    preds = img.new_zeros((1, num_classes, Hp, Wp))
    count = img.new_zeros((1, 1, Hp, Wp))
    for hi in range(h_grids):
        for wi in range(w_grids):
            y1 = min(hi * stride, Hp - crop)
            x1 = min(wi * stride, Wp - crop)
            win = img[:, :, y1:y1 + crop, x1:x1 + crop]
            logit = model(win, tasks=[task])[task].float()
            if logit.shape[-2:] != (crop, crop):        # head may emit < crop res
                logit = F.interpolate(logit, (crop, crop), mode="bilinear", align_corners=False)
            preds[:, :, y1:y1 + crop, x1:x1 + crop] += logit
            count[:, :, y1:y1 + crop, x1:x1 + crop] += 1
    preds /= count.clamp(min=1)
    return preds[:, :, :H, :W]


class Evaluator:
    def __init__(self, tasks, meter_cfgs=None, device="cuda"):
        self.tasks = list(tasks)
        self.meter_cfgs = meter_cfgs or {}
        self.device = device

    @torch.no_grad()
    def evaluate(self, model, loader, slide=None):
        """slide=None -> whole-image forward (default). slide={'crop','stride'} ->
        sliding-window seg inference at native resolution (seg-only; loader must
        yield native-res images, e.g. batch_size 1 with data.val size=null)."""
        model.eval()
        meters = None
        for batch in loader:
            tasks = [t for t in batch["tasks"] if t in self.tasks]
            if meters is None:
                meters = {t: build_meter(t, self.meter_cfgs.get(t)) for t in tasks}
            x = batch["image"].to(self.device)
            if slide is not None:
                assert tasks == ["seg"], "slide inference is seg-only"
                nc = self.meter_cfgs.get("seg", {}).get("num_classes")
                for b in range(x.shape[0]):              # per-image (sizes may differ)
                    logits = slide_inference(model, x[b:b + 1], "seg", nc,
                                             crop=slide.get("crop", 512),
                                             stride=slide.get("stride", 341))
                    meters["seg"].update(logits.cpu(), batch["targets"]["seg"][b:b + 1])
            else:
                preds = model(x, tasks=tasks)
                for t in tasks:
                    meters[t].update(preds[t].float().cpu(), batch["targets"][t])
        return {t: m.compute() for t, m in (meters or {}).items()}
