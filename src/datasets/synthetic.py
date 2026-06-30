"""Synthetic multi-task dataset for smoke tests — no files needed. Emits exactly
the same sample contract as the real datasets so the whole pipeline (forward,
conditional loss, eval, export) can be exercised green without Cityscapes."""
import torch
from torch.utils.data import Dataset


class SyntheticMultiTask(Dataset):
    def __init__(self, tasks=("seg",), size=(128, 256), num_classes=19, length=16,
                 max_depth=80.0, ignore_index=255):
        self.tasks = list(tasks)
        self.h, self.w = size
        self.num_classes = num_classes
        self.length = length
        self.max_depth = max_depth
        self.ignore_index = ignore_index

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(idx)        # deterministic per-index
        image = torch.randn(3, self.h, self.w, generator=g)
        targets = {}
        if "seg" in self.tasks:
            seg = torch.randint(0, self.num_classes, (self.h, self.w), generator=g)
            seg[:2, :] = self.ignore_index            # a strip of ignore pixels
            targets["seg"] = seg
        if "depth" in self.tasks:
            targets["depth"] = torch.rand(1, self.h, self.w, generator=g) * self.max_depth
        if "normal" in self.tasks:
            n = torch.randn(3, self.h, self.w, generator=g)
            targets["normal"] = torch.nn.functional.normalize(n, dim=0)
        return {"image": image, "targets": targets, "tasks": list(self.tasks)}
