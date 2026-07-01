# Semantic Sensing

Semantic Sensing is a PyTorch research codebase for channel-agnostic semantic source coding in split visual perception. The current mainline studies RTFormer-based semantic segmentation on ADE20K, with a learned communication bottleneck inserted between the sender-side convolutional encoder and receiver-side attention tail.

This repository focuses on source coding for machine perception. It does not simulate wireless channels, inject channel noise, or implement JSCC. The bottleneck produces a byte stream that can be handed to a separate channel coder.

## Current Focus

- Backbone/task: RTFormer on ADE20K single-task semantic segmentation.
- Phase 2 split: `mid`, after RTFormer stage3 at the convolution-to-attention boundary.
- Payload: dual features `(128@1/8, 256@1/16)` compressed by `rtformer_distill`.
- Bottleneck: two mean-scale hyperpriors packed into one byte container.
- Training: Stage 1 feature distillation plus rate loss, then Stage 2 receiver-tail task fine-tuning.

Multi-task depth/normal support is deferred, but the model framework still keeps dict heads and interchangeable task losses.

## Repository Layout

```text
configs/        YAML experiment configs and Phase 2 split settings
src/            models, datasets, losses, training loops, metrics
src/models/     RTFormer, SplitStudent, bottlenecks, heads, decoders
tools/          training, evaluation, smoke-test, export utilities
tests/          split equivalence, bottleneck contract, bitstream tests
notes/          research notes and progress logs
deploy/hailo/   Hailo deployment helpers
```

## Quick Checks

Run from the repository root:

```bash
python tools/smoke_test.py
python tools/smoke_student.py
python tools/smoke_student_split.py
CUDA_VISIBLE_DEVICES="" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
```

The Pytest command is CPU-oriented and avoids a broken external plugin entry point in the current container image.

## Phase 2 Example

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train_student.py configs/phase2_mid_stage1.yaml \
  --teacher-ckpt /data/sidney/checkpoints/semantic-sensing/ade20k/ade20k_rtformer_paperhead.pth \
  --beta 0.01 \
  --out /data/sidney/checkpoints/semantic-sensing/students/mid_b0.01_s1.pth
```

Use `/data/sidney/` for checkpoints, datasets, exported ONNX/HAR files, and long logs. Keep `/workspace` for code, configs, and lightweight documentation.

## Contributor Notes

Read `AGENTS.md`, `CLAUDE.md`, and the latest files in `notes/` before making architectural changes. The project intentionally uses plain PyTorch, OmegaConf YAML composition, and explicit `build_X(cfg)` factories instead of registry/decorator frameworks.
