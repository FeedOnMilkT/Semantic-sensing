# Repository Guidelines

## Project Structure & Module Organization

This workspace contains several nested projects. Treat `semantic-sensing/` as the active research codebase. Core code lives in `semantic-sensing/src/`; YAML experiments in `configs/`; runnable entry points in `tools/`; tests in `tests/`; research/progress records in `notes/`; Hailo deployment helpers in `deploy/hailo/`.

`demo/` contains frozen de-risking references or vendor-style demos; avoid changing it unless the task is explicitly demo-related. `bevformer-plugin/` is a separate mmdetection3d plugin.

## Current Project Direction

The current mainline is `RTFormer × ADE20K single-task semantic segmentation × Phase 2 split/semantic compression`. Multi-task depth/normal work is postponed, but the framework remains multi-task-capable through dict heads and interchangeable task losses. Cityscapes is now a cross-dataset reference rather than the primary target.

Phase 2 is not JSCC: do not simulate channels, inject channel noise, or output analog symbols. The project implements channel-agnostic semantic source coding; the byte stream produced by the bottleneck is the handoff to a separate channel coder.

## Architecture & Implementation Notes

The baseline model is assembled by `src/models/multitask_net.py`: `backbone -> neck -> bottleneck -> decoder -> heads`. Backbones return `{"s4","s8","s16","s32"}`. Components use module-level dictionaries plus `build_X(cfg)` factories; do not introduce registries, decorators, Lightning, or mmengine-style abstractions.

The active Phase 2 path is `src/models/student.py`. `SplitStudent` wraps a frozen RTFormer teacher and inserts a trainable communication bottleneck at `early`, `mid`, `late`, or legacy `neck` splits. `mid` is the mainline: cut after RTFormer stage3, exactly at the conv-to-attention boundary, with dual payload `(x_h: 128@1/8, x_l: 256@1/16)`. `late` is a non-deployable ablation because attention runs on the sender. `early` is the shallow ablation.

The main bottleneck is `rtformer_distill`: two independent mean-scale hyperpriors packed into one bytes container. Keep bpp normalized by original image pixels. `act: gdn` is the RD curve setting; `act: convbn` is the Hailo-foldable variant.

## Build, Test, and Development Commands

Run from `/workspace/semantic-sensing`:

```bash
python tools/smoke_test.py
python tools/smoke_student.py
python tools/smoke_student_split.py
CUDA_VISIBLE_DEVICES="" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
```

Phase 2 training examples:

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train_student.py configs/phase2_mid_stage1.yaml \
  --teacher-ckpt /data/sidney/checkpoints/semantic-sensing/ade20k/ade20k_rtformer_paperhead.pth \
  --beta 0.01 --out /data/sidney/checkpoints/semantic-sensing/students/mid_b0.01_s1.pth

CUDA_VISIBLE_DEVICES=1 python tools/train_student.py configs/phase2_mid_stage2.yaml --stage 2 \
  --teacher-ckpt /data/sidney/checkpoints/semantic-sensing/ade20k/ade20k_rtformer_paperhead.pth \
  --student-ckpt /data/sidney/checkpoints/semantic-sensing/students/mid_b0.01_s1.pth
```

Stage 1 trains the bottleneck with `feature_MSE + beta*bpp`; Stage 2 freezes sender/bottleneck and fine-tunes the receiver tail with `CE + Dice`. `--task-weight` exists for the task-driven/ICM arm, but the current implemented mainline is feature-distillation first.

## Testing Guidelines

Tests should preserve split equivalence, bottleneck contracts, bitstream roundtrip, and estimated-vs-real bpp sanity. rANS `compress/decompress` is CPU-only; keep bitstream tests CPU-side and use estimated bpp during GPU training.

## Workspace & Configuration Rules

Follow `/workspace/CLAUDE.md` and `semantic-sensing/CLAUDE.md` before substantial work. Keep `/workspace` for code/configs/docs only. Put checkpoints, datasets, ONNX/HAR files, and long logs under `/data/sidney/`. Before heavy GPU jobs, check `nvidia-smi` and set `CUDA_VISIBLE_DEVICES`.

## Commit & Pull Request Guidelines

Use short, specific commit subjects such as `Add mid split dual-latent bottleneck` or `Wire rate into student training`. PRs should list changed configs/code paths, commands run, teacher checkpoint used, and any `[POINT] bpp/mIoU` results.

## Raspberry Pi SSH (Tailscale userspace)

The board is **not** reachable by plain `ssh sidney@100.81.213.24` from this container.
Agents must use the userspace Tailscale under `/workspace/.tailscale`:

- socket: `/workspace/.tailscale/tailscaled.sock`
- SOCKS5: `127.0.0.1:1055` (`--tun=userspace-networking --socks5-server=...`)
- deploy key: `/workspace/.tailscale/siriberry_deploy_ed25519`
- known_hosts: `/workspace/.tailscale/siriberry_known_hosts`

```bash
# Ensure daemon (idempotent)
mkdir -p /workspace/.tailscale
if ! pgrep -x tailscaled >/dev/null; then
  nohup sudo /usr/sbin/tailscaled \
    --state=/workspace/.tailscale/tailscaled.state \
    --socket=/workspace/.tailscale/tailscaled.sock \
    --tun=userspace-networking \
    --socks5-server=127.0.0.1:1055 \
    >/workspace/.tailscale/tailscaled.log 2>&1 &
fi

ssh -i /workspace/.tailscale/siriberry_deploy_ed25519 -o IdentitiesOnly=yes \
  -o 'ProxyCommand=nc -X 5 -x 127.0.0.1:1055 %h %p' \
  -o BatchMode=yes -o ConnectTimeout=25 \
  -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile=/workspace/.tailscale/siriberry_known_hosts \
  sidney@siriberry.tailae65a5.ts.net
# or: sidney@100.81.213.24
```

Do **not** rely on `siriberry.local` (LAN mDNS) or MagicDNS without the SOCKS proxy from this host.
