"""ADE20K semantic segmentation dataset for the paper's final ten-object task.

Loaded from the HuggingFace parquet mirror `merve/scene_parse_150` (datasets>=4
compatible; the official script-based `scene_parse_150` / `zhoubolei/...` FAIL on
datasets 5.0). Already cached on the data disk at
`/data/sidney/hf-cache/hub/datasets--merve--scene_parse_150` (train 20210 / val
2000 / test 3352; test is unlabeled). Public repo — no token needed.

`class_mode="ten_object"` maps the paper's ten foreground categories to labels
1..10 and all other source labels, including source 0, to background 0. Source
category ids follow CSAILVision's official one-based `objectInfo150.txt` table.
`class_mode="150"` keeps the earlier 150-class protocol and maps source 0 to
`ignore_index`.

Augmentation (train only, `augment: true`) follows RTFormer's paper recipe
(Sec 4.1, "other training settings are identical to those for Cityscapes"):
random scale in [0.5, 2.0] -> pad-to-crop (seg=ignore) -> random crop -> random
horizontal flip. NO photometric distortion (the paper lists color jitter only for
CamVid). The crop is 512x512 — the ADE20K community standard (mmseg/PaddleSeg);
the paper doesn't state an ADE20K crop explicitly, only that the *method* matches
Cityscapes (which crops 512x1024). Eval resizes to a fixed square `size` (default
512x512) so val images batch; this is a simplification of mmseg's ratio-preserving
+ slide eval and will read a touch lower than that protocol.

Returns the sample contract {"image", "targets":{"seg":...}, "tasks":["seg"]},
matching cityscapes.py so the trainer/collate treat it identically.
"""
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# our split name -> HF split name (HF uses "validation", we say "val" like cityscapes)
_HF_SPLIT = {"train": "train", "val": "validation", "test": "test"}

# Paper order: bed, cabinet, table, car, sofa, armchair, bathtub, door, desk, mirror.
# Source ids are the one-based indices in CSAILVision/ADE20K objectInfo150.txt.
TEN_OBJECT_SOURCE_IDS = (8, 11, 16, 21, 24, 31, 38, 15, 34, 28)
TEN_OBJECT_NAMES = ("bed", "cabinet", "table", "car", "sofa", "armchair",
                    "bathtub", "door", "desk", "mirror")
_TEN_OBJECT_LOOKUP = np.zeros(151, dtype=np.int64)
for _train_id, _source_id in enumerate(TEN_OBJECT_SOURCE_IDS, start=1):
    _TEN_OBJECT_LOOKUP[_source_id] = _train_id


def remap_ade20k_labels(annotation, class_mode="150", ignore_index=255):
    """Map source PNG ids to either the 150-class or final 11-class train space."""
    ann = np.asarray(annotation, dtype=np.int64)
    if ann.size and (ann.min() < 0 or ann.max() > 150):
        raise ValueError(f"ADE20K annotation ids outside expected 0..150: {ann.min()}..{ann.max()}")
    if class_mode == "ten_object":
        if ignore_index in range(11):
            raise ValueError("ignore_index must not overlap ten-object train labels 0..10")
        return _TEN_OBJECT_LOOKUP[ann]
    if class_mode == "150":
        seg = ann - 1
        seg[seg < 0] = ignore_index
        return seg
    raise ValueError(f"unsupported ADE20K class_mode={class_mode!r}")


class ADE20K(Dataset):
    def __init__(self, hf_repo="merve/scene_parse_150", split="train",
                 tasks=("seg",), size=(512, 512), num_classes=150,
                 ignore_index=255, augment=False, crop_size=(512, 512),
                 scale_range=(0.5, 2.0), class_mode="150"):
        assert tuple(tasks) == ("seg",), "ADE20K is a seg-only scene-parsing set"
        self.split, self.tasks = split, list(tasks)
        # size=None -> return native resolution (for slide-window eval; needs batch=1)
        self.size = tuple(size) if size is not None else None
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_mode = str(class_mode)
        if self.class_mode == "ten_object":
            if num_classes != len(TEN_OBJECT_SOURCE_IDS) + 1:
                raise ValueError("ten_object mode requires num_classes=11 (background + 10 objects)")
            if len(set(TEN_OBJECT_SOURCE_IDS)) != len(TEN_OBJECT_SOURCE_IDS):
                raise ValueError("ten-object ADE20K source ids must be unique")
            if ignore_index in range(num_classes):
                raise ValueError("ignore_index must not overlap ten-object train labels 0..10")
        elif self.class_mode == "150":
            if num_classes != 150:
                raise ValueError("150-class mode requires num_classes=150")
            self._source_to_train = None
        else:
            raise ValueError(f"unsupported ADE20K class_mode={self.class_mode!r}")
        self.augment = bool(augment)
        self.crop_size = tuple(crop_size) if crop_size else tuple(size)
        self.scale_range = tuple(scale_range)
        # memmapped arrow; built once in the main process, fork-shared by workers.
        self.ds = load_dataset(hf_repo, split=_HF_SPLIT[split])

    def __len__(self):
        return len(self.ds)

    def _load_native(self, idx):
        rec = self.ds[idx]
        img = rec["image"].convert("RGB")                 # some ADE20K imgs are grayscale
        ann = np.asarray(rec["annotation"], np.int64)     # 0=unlabeled, 1..150=official classes
        if self.class_mode == "ten_object":
            seg = remap_ade20k_labels(ann, self.class_mode, self.ignore_index)
        else:
            seg = remap_ade20k_labels(ann, self.class_mode, self.ignore_index)
        return img, seg

    @staticmethod
    def _resize(img, seg, W, H):
        img = img.resize((W, H), Image.BILINEAR)
        seg = np.asarray(Image.fromarray(seg.astype(np.int32)).resize((W, H), Image.NEAREST), np.int64)
        return np.asarray(img, np.uint8), seg

    def _augment(self, img, seg):
        """img: native PIL RGB; seg: native np int64 (ids already remapped)."""
        W0, H0 = img.size                                  # PIL .size is (W, H)
        s = random.uniform(*self.scale_range)
        nW, nH = max(1, int(round(W0 * s))), max(1, int(round(H0 * s)))
        img_np, seg = self._resize(img, seg, nW, nH)

        cH, cW = self.crop_size
        padH, padW = max(0, cH - nH), max(0, cW - nW)      # pad up if scaled below crop
        if padH or padW:
            img_np = np.pad(img_np, ((0, padH), (0, padW), (0, 0)), constant_values=0)
            seg = np.pad(seg, ((0, padH), (0, padW)), constant_values=self.ignore_index)

        H1, W1 = img_np.shape[:2]
        top = random.randint(0, H1 - cH)
        left = random.randint(0, W1 - cW)
        img_np = img_np[top:top + cH, left:left + cW]
        seg = seg[top:top + cH, left:left + cW]

        if random.random() < 0.5:                          # horizontal flip
            img_np = img_np[:, ::-1]
            seg = seg[:, ::-1]
        return img_np.copy(), seg.copy()

    def __getitem__(self, idx):
        img, seg = self._load_native(idx)
        if self.augment:
            img_np, seg = self._augment(img, seg)
        elif self.size is None:
            img_np = np.asarray(img, np.uint8)          # native resolution, seg untouched
        else:
            H, W = self.size
            img_np, seg = self._resize(img, seg, W, H)

        image = torch.from_numpy(np.ascontiguousarray(img_np.transpose(2, 0, 1)).astype(np.float32) / 255.0)
        image = (image - _MEAN) / _STD

        targets = {"seg": torch.from_numpy(np.ascontiguousarray(seg))}
        return {"image": image, "targets": targets, "tasks": list(self.tasks)}
