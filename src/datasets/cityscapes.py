"""Cityscapes multi-task dataset (seg + depth). Conditional on cfg.tasks.

- seg:   gtFine/*_labelTrainIds.png (19 train ids, 255 = ignore). If the
         labelTrainIds.png isn't present (the official zip ships only labelIds),
         falls back to *_labelIds.png and remaps id->trainId on the fly via the
         standard 34->19 table — no cityscapesscripts preprocessing pass needed.
- depth: from disparity per CLAUDE.md formula  depth = 0.222 * 2262.0 / (disp + eps),
         with the 16-bit PNG decoded as disp = (raw - 1) / 256 for raw > 0,
         then clipped to [1e-3, 80] m. Invalid disparity (raw==0) -> 0 (masked out
         in loss/metrics). Depth is metric (meters), so spatial scaling/cropping in
         augmentation leaves the values untouched — only the grid is resampled.

Augmentation (train only, `augment: true`): standard Cityscapes recipe —
random scale (0.5-2.0) -> pad-to-crop (seg=ignore, depth=0) -> random crop
-> random horizontal flip. Eval just resizes to `size` (= native 1024x2048 for
full-resolution evaluation). seg/depth resample with NEAREST, image with BILINEAR.

Returns the sample contract {"image", "targets":{task:...}, "tasks":[...]}.
Requires the standard Cityscapes layout under `root`:
    leftImg8bit/{split}/<city>/*_leftImg8bit.png
    gtFine/{split}/<city>/*_gtFine_labelTrainIds.png
    disparity/{split}/<city>/*_disparity.png
"""
import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# Standard Cityscapes labelId (0..33) -> 19-class trainId; everything else = ignore.
_ID2TRAIN = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
             23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _build_id2train_lut(ignore_index):
    lut = np.full(256, ignore_index, dtype=np.int64)
    for k, v in _ID2TRAIN.items():
        lut[k] = v
    return lut


class CityscapesMultiTask(Dataset):
    def __init__(self, root, split="train", tasks=("seg",), size=(1024, 2048),
                 num_classes=19, ignore_index=255, max_depth=80.0,
                 augment=False, crop_size=None, scale_range=(0.5, 2.0)):
        assert "normal" not in tasks, "Cityscapes has no native normal GT (CLAUDE.md §8)"
        self.root, self.split, self.tasks = root, split, list(tasks)
        self.size, self.num_classes = tuple(size), num_classes
        self.ignore_index, self.max_depth = ignore_index, max_depth
        self.augment = bool(augment)
        self.crop_size = tuple(crop_size) if crop_size else tuple(size)
        self.scale_range = tuple(scale_range)
        self._lut = _build_id2train_lut(ignore_index)
        img_dir = os.path.join(root, "leftImg8bit", split)
        self.images = sorted(glob.glob(os.path.join(img_dir, "*", "*_leftImg8bit.png")))
        if not self.images:
            raise FileNotFoundError(f"no Cityscapes images under {img_dir}")

    def __len__(self):
        return len(self.images)

    def _paths(self, img_path):
        base = os.path.basename(img_path).replace("_leftImg8bit.png", "")
        city = os.path.basename(os.path.dirname(img_path))
        gt_dir = os.path.join(self.root, "gtFine", self.split, city)
        train_ids = os.path.join(gt_dir, base + "_gtFine_labelTrainIds.png")
        label_ids = os.path.join(gt_dir, base + "_gtFine_labelIds.png")
        seg = train_ids if os.path.exists(train_ids) else label_ids   # latter -> remap on load
        disp = os.path.join(self.root, "disparity", self.split, city, base + "_disparity.png")
        return seg, disp

    def _load_seg_native(self, seg_path):
        arr = np.asarray(Image.open(seg_path), np.int64)
        if seg_path.endswith("_labelIds.png"):      # raw ids -> trainIds via LUT
            arr = self._lut[arr]
        return arr

    def _load_depth_native(self, disp_path):
        raw = np.asarray(Image.open(disp_path), np.float32)
        disp = np.where(raw > 0, (raw - 1.0) / 256.0, 0.0)
        depth = np.where(disp > 0, 0.222 * 2262.0 / (disp + 1e-6), 0.0)
        return np.clip(depth, 0.0, self.max_depth).astype(np.float32)  # 0 = invalid

    @staticmethod
    def _resize(img, seg, depth, W, H):
        img = img.resize((W, H), Image.BILINEAR)
        if seg is not None:
            seg = np.asarray(Image.fromarray(seg.astype(np.int32)).resize((W, H), Image.NEAREST), np.int64)
        if depth is not None:
            depth = np.asarray(Image.fromarray(depth).resize((W, H), Image.NEAREST), np.float32)
        return np.asarray(img, np.uint8), seg, depth

    def _augment(self, img, seg, depth):
        """img: native PIL RGB; seg: native np int64; depth: native np float32 or None."""
        W0, H0 = img.size                                  # PIL .size is (W, H)
        s = random.uniform(*self.scale_range)
        nW, nH = max(1, int(round(W0 * s))), max(1, int(round(H0 * s)))
        img_np, seg, depth = self._resize(img, seg, depth, nW, nH)

        cH, cW = self.crop_size
        padH, padW = max(0, cH - nH), max(0, cW - nW)      # pad up if scaled below crop
        if padH or padW:
            img_np = np.pad(img_np, ((0, padH), (0, padW), (0, 0)), constant_values=0)
            if seg is not None:
                seg = np.pad(seg, ((0, padH), (0, padW)), constant_values=self.ignore_index)
            if depth is not None:
                depth = np.pad(depth, ((0, padH), (0, padW)), constant_values=0.0)

        H1, W1 = img_np.shape[:2]
        top = random.randint(0, H1 - cH)
        left = random.randint(0, W1 - cW)
        img_np = img_np[top:top + cH, left:left + cW]
        if seg is not None:
            seg = seg[top:top + cH, left:left + cW]
        if depth is not None:
            depth = depth[top:top + cH, left:left + cW]

        if random.random() < 0.5:                          # horizontal flip
            img_np = img_np[:, ::-1]
            if seg is not None:
                seg = seg[:, ::-1]
            if depth is not None:
                depth = depth[:, ::-1]
        return img_np.copy(), (seg.copy() if seg is not None else None), (depth.copy() if depth is not None else None)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        seg_path, disp_path = self._paths(img_path)
        img = Image.open(img_path).convert("RGB")          # native resolution
        seg = self._load_seg_native(seg_path) if "seg" in self.tasks else None
        depth = self._load_depth_native(disp_path) if "depth" in self.tasks else None

        if self.augment:
            img_np, seg, depth = self._augment(img, seg, depth)
        else:
            H, W = self.size
            img_np, seg, depth = self._resize(img, seg, depth, W, H)

        image = torch.from_numpy(np.ascontiguousarray(img_np.transpose(2, 0, 1)).astype(np.float32) / 255.0)
        image = (image - _MEAN) / _STD

        targets = {}
        if "seg" in self.tasks:
            targets["seg"] = torch.from_numpy(np.ascontiguousarray(seg))
        if "depth" in self.tasks:
            targets["depth"] = torch.from_numpy(np.ascontiguousarray(depth))[None]   # (1,H,W); 0 = invalid

        return {"image": image, "targets": targets, "tasks": list(self.tasks)}
