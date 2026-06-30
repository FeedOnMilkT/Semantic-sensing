"""Shared helpers for the CLI tools. Adds repo root to sys.path so `src` imports
work regardless of CWD."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, DistributedSampler

from src.datasets import build_dataset, collate


def make_loader(data_cfg, batch_size, shuffle, num_workers=4, distributed=False):
    """Build a DataLoader. When distributed=True, shard via DistributedSampler
    (one shard per rank). The sampler is exposed as loader.sampler so the trainer
    can call set_epoch() each epoch. batch_size here is PER-GPU."""
    ds = build_dataset(data_cfg)
    if distributed:
        sampler = DistributedSampler(ds, shuffle=shuffle, drop_last=shuffle)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                          collate_fn=collate, pin_memory=True, drop_last=shuffle)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate, pin_memory=True, drop_last=shuffle)


def tasks_from_cfg(cfg):
    return list(cfg.model.heads.keys())


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
