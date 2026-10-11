# Project context

This repository implements channel-independent semantic feature compression around RTFormer. The active paper setting uses the convolutional sender through stage 3, a learned bottleneck, and the receiver-side RTFormer tail. Training has distinct jobs: optional ImageNet backbone initialization; ADE20K teacher training; codec-only Stage 1; and receiver-tail Stage 2. Do not conflate their data, labels, precision, or epoch/step budgets.

The final evaluation protocol names ten ADE20K foreground objects and treats all remaining labels as background. The resulting 11-way output is an implementation inference because the paper does not explicitly state the final teacher head dimension. An earlier 150-class baseline is retained for comparison.

Read `README.md` for runnable entry points and `docs/PAPER_CODE_ALIGNMENT.md` for the exact paper sections, code paths, known divergences, and implementation decisions. The Chinese server prompt covers ImageNet initialization only; it does not run the ADE20K teacher or codec stages.

Keep training artifacts outside git. Check `git status` before edits, preserve existing changes, and do not run full training without explicit authorization. See `AGENTS.md` for the repository rules.
