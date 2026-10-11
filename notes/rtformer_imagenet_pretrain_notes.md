# Historical RTFormer ImageNet-1K run

This records an earlier four-GPU A100 run, not the paper's training recipe and not a promise for a different server. The RTFormer trunk (`forward_trunk`, through stage 5) plus a 1,000-class linear head was trained for 100 epochs and reached 71.82% validation top-1 in 7 h 40 min. That run used per-GPU batch 1024 (global 4096), AdamW, LR 2.8e-3, weight decay 0.05, five warm-up epochs, cosine decay, label smoothing 0.1, and FP16 autocast with GradScaler.

The run's purpose was to produce an ImageNet initialization for downstream segmentation. It is an older engineering result. The current server workflow is documented in `docs/TRAINING_SERVER_PROMPT_ZH.md`; it targets one GPU, BF16, batch 128 and a 100-epoch default. Those settings differ from this historical run and are not specified by the paper.

The earlier run also showed that oversized batches and persistent validation workers can exhaust container shared memory. Check `/dev/shm`, worker counts, free disk, and GPU memory on the target host rather than reusing its old hardware assumptions. The exported backbone excludes DAPPM because `forward_trunk()` does not execute that segmentation head.
