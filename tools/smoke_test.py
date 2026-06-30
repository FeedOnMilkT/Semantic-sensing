#!/usr/bin/env python
"""Smoke test — exercises the whole pipeline GREEN on synthetic data (no dataset
needed) and checks CLAUDE.md's acceptance criteria 1-4:

  #1  backbone.name swap (mit_b0 / rtformer / topformer), one line, both train.
  #2  add/remove a task in heads: -> trainer runs only the active heads.
  #3  Phase-1a seg config trains; Phase-1b adds depth head, conditional loss works.
  #4  export_onnx static-shape export raises no error (onnx.checker passes).

Prints PASS/FAIL per check; exits non-zero on any failure.
"""
import sys
import traceback

from _common import pick_device
from omegaconf import OmegaConf
import torch

from src.utils import load_config
from src.models import build_model
from src.losses import build_loss
from src.engine import Trainer, Evaluator
from src.datasets import collate
from src.datasets.synthetic import SyntheticMultiTask
from torch.utils.data import DataLoader

SIZE = (64, 128)          # tiny for speed
DEVICE = pick_device()
RESULTS = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        RESULTS.append(True)
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()
        RESULTS.append(False)


def synth_loader(tasks, bs=2, length=4):
    ds = SyntheticMultiTask(tasks=tasks, size=SIZE, num_classes=19, length=length)
    return DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate)


def train_a_few(cfg, tasks, steps=3):
    model = build_model(cfg.model)
    loss = build_loss(cfg.loss, tasks)
    cfg.train.warmup = 0
    tr = Trainer(cfg.train, model, loss, device=DEVICE)
    tr.fit(synth_loader(tasks), epochs=1, max_steps=steps, log_every=1)
    return model


# ---- #1 backbone swap (single-task seg) ----
def c1_backbone_swap():
    seg_cfg = load_config("configs/cityscapes_seg.yaml")
    for bb in ["mit_b0", "rtformer", "topformer"]:
        cfg = OmegaConf.create(OmegaConf.to_container(seg_cfg, resolve=True))
        cfg.model.backbone.name = bb           # the one-line swap
        train_a_few(cfg, ["seg"])
        print(f"        backbone={bb} trained 3 steps OK")


# ---- #3 multitask conditional loss (seg + depth) ----
def c3_multitask():
    mt_cfg = load_config("configs/cityscapes_multitask.yaml")
    model = train_a_few(mt_cfg, ["seg", "depth"])
    out = model(torch.randn(1, 3, *SIZE).to(DEVICE), tasks=["seg", "depth"])
    assert set(out.keys()) == {"seg", "depth"}, out.keys()
    assert out["seg"].shape[-2:] == SIZE and out["depth"].shape[1] == 1


# ---- #2 add/remove head -> only active heads run ----
def c2_add_remove_head():
    mt_cfg = load_config("configs/cityscapes_multitask.yaml")
    # remove depth from the YAML heads dict -> model has only seg
    cfg = OmegaConf.create(OmegaConf.to_container(mt_cfg, resolve=True))
    del cfg.model.heads["depth"]
    model = build_model(cfg.model).to(DEVICE)
    assert set(model.heads.keys()) == {"seg"}
    # a seg-only batch through the seg-only loss runs without touching depth
    loss = build_loss(cfg.loss, ["seg"])
    out = model(torch.randn(2, 3, *SIZE).to(DEVICE), tasks=["seg"])
    assert set(out.keys()) == {"seg"}
    total, parts = loss(out, {"seg": torch.randint(0, 19, (2, *SIZE)).to(DEVICE)}, ["seg"])
    assert "depth" not in parts and torch.isfinite(total)
    # and a model WITH both heads, fed a seg-only batch, only computes seg loss
    full = build_model(mt_cfg.model).to(DEVICE)
    floss = build_loss(mt_cfg.loss, ["seg", "depth"])
    fout = full(torch.randn(2, 3, *SIZE).to(DEVICE), tasks=["seg"])   # only seg requested
    _, fparts = floss(fout, {"seg": torch.randint(0, 19, (2, *SIZE)).to(DEVICE)}, ["seg"])
    assert set(fparts.keys()) == {"seg"}, fparts.keys()


# ---- eval pipeline (conditional metrics) ----
def c_eval():
    mt_cfg = load_config("configs/cityscapes_multitask.yaml")
    model = build_model(mt_cfg.model).to(DEVICE)
    ev = Evaluator(["seg", "depth"], mt_cfg.eval.meters, device=DEVICE)
    metrics = ev.evaluate(model, synth_loader(["seg", "depth"], bs=2))
    assert "seg" in metrics and "mIoU" in metrics["seg"]
    assert "depth" in metrics and "RMSE" in metrics["depth"]
    print(f"        metrics keys: { {k: list(v) for k, v in metrics.items()} }")


# ---- #4 ONNX static export ----
def c4_onnx_export():
    import onnx
    import torch.nn as nn
    seg_cfg = load_config("configs/cityscapes_seg.yaml")
    model = build_model(seg_cfg.model).eval()
    tasks = list(seg_cfg.model.heads.keys())

    class Wrap(nn.Module):
        def __init__(self, m, t): super().__init__(); self.m, self.t = m, t
        def forward(self, x): return tuple(self.m(x, tasks=self.t)[k] for k in self.t)

    out = "/tmp/smoke_seg.onnx"
    torch.onnx.export(Wrap(model, tasks), torch.randn(1, 3, *SIZE), out,
                      input_names=["image"], output_names=[f"out_{t}" for t in tasks],
                      opset_version=13, do_constant_folding=True, dynamic_axes=None)
    onnx.checker.check_model(onnx.load(out))
    print(f"        exported + checked {out}")


def main():
    print(f"[smoke] device={DEVICE} size={SIZE}")
    print("[#1] backbone swap (mit_b0 / rtformer / topformer):")
    check("backbone swap trains", c1_backbone_swap)
    print("[#3] multitask seg+depth conditional loss:")
    check("multitask trains + forwards both heads", c3_multitask)
    print("[#2] add/remove head -> only active heads run:")
    check("conditional heads", c2_add_remove_head)
    print("[eval] conditional metrics:")
    check("evaluator", c_eval)
    print("[#4] static ONNX export:")
    check("onnx export", c4_onnx_export)

    n_pass, n = sum(RESULTS), len(RESULTS)
    print(f"\n[smoke] {n_pass}/{n} checks passed")
    sys.exit(0 if n_pass == n else 1)


if __name__ == "__main__":
    main()
