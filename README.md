# Semantic Sensing

PyTorch research code for the paper's RTFormer semantic source-coding experiments. The main published path is an ADE20K teacher, followed by a learned feature codec (Stage 1) and a frozen-codec receiver adaptation (Stage 2). Optional ImageNet RTFormer backbone pretraining is a separate initialization step; it is not the paper's ADE20K teacher training recipe.

## Start here

- [Paper-to-code alignment and remaining gaps](docs/PAPER_CODE_ALIGNMENT.md)
- [Standalone Chinese ImageNet server prompt](docs/TRAINING_SERVER_PROMPT_ZH.md)
- [Repository instructions](AGENTS.md)

## Local checks

Python 3.11 is the supported starting point. Install the CUDA-specific PyTorch wheel separately on a GPU server after checking its GPU and driver; `requirements-research.txt` contains the additional research dependencies. On this M2 Pro workstation, use an isolated CPU/MPS environment and do not start a real training run.

```bash
uv venv --python 3.11 /tmp/semantic-sensing-check
uv pip install --python /tmp/semantic-sensing-check/bin/python -r requirements-research.txt torch torchvision
/tmp/semantic-sensing-check/bin/python tools/pretrain_imagenet.py --help
OMP_NUM_THREADS=2 /tmp/semantic-sensing-check/bin/python tools/pretrain_imagenet.py \
  --device cpu --precision fp32 --synthetic-data --dry-run --batch-size 2 \
  --workers 0 --img-size 64 --out /tmp/imagenet-smoke.pth
```

The synthetic CLI fixture executes one forward/backward optimizer step and one validation batch, then exits without writing weights. It checks wiring only, not ImageNet accuracy or CUDA/BF16 support.

Real data training entry points:

```bash
python tools/train.py configs/ade20k_rtformer_ten_object.yaml --max-steps 160000 \
  --out /data/checkpoints/semantic-sensing/ade20k_teacher.pth
python tools/train_student.py configs/phase2_mid_fused_stage1.yaml \
  --teacher-ckpt /data/checkpoints/ade20k_teacher.pth --beta 0.01 \
  --out /data/checkpoints/mid_fused_s1.pth
python tools/train_student.py configs/phase2_mid_fused_stage2.yaml --stage 2 \
  --teacher-ckpt /data/checkpoints/ade20k_teacher.pth \
  --student-ckpt /data/checkpoints/mid_fused_s1.pth
```

The first command follows the paper's 160,000-step teacher budget. The Stage 1/2 commands require compatible checkpoints and dataset access; see the alignment note before using configs, since older experiment files intentionally retain 150-class settings.

## Repository map

`src/` contains models, datasets, losses, training loops, and metrics; `configs/` contains experiment recipes; `tools/` contains runnable training and validation utilities; `tests/` contains focused checks. `notes/` retains selected source-paper notes and a historical ImageNet experiment record. Datasets, caches, logs, and weights belong outside the repository.
