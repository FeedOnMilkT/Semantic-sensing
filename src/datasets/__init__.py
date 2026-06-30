"""Dataset factory (no registry) + a collate that preserves the sample contract.

Every dataset yields {"image", "targets":{task:tensor}, "tasks":[...]}; within a
dataset all samples share the same active task list, so collate batches the
shared tasks and the trainer reads `batch["tasks"]` to know which heads to run.
"""
import torch

from .synthetic import SyntheticMultiTask
from .cityscapes import CityscapesMultiTask
from .ade20k import ADE20K

_DATASETS = {
    "synthetic": SyntheticMultiTask,
    "cityscapes": CityscapesMultiTask,
    "ade20k": ADE20K,
}


def build_dataset(cfg):
    return _DATASETS[cfg.name](**cfg.get("args", {}))


def collate(batch):
    tasks = batch[0]["tasks"]
    images = torch.stack([b["image"] for b in batch])
    targets = {t: torch.stack([b["targets"][t] for b in batch]) for t in tasks}
    return {"image": images, "targets": targets, "tasks": tasks}
