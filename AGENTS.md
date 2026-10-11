# Repository guidance

- This is the training/research repository. Do not change the paper source or deployment projects unless explicitly requested.
- Before editing, inspect `git status` and preserve pre-existing changes. Do not reset, clean, or overwrite user work.
- The paper's architecture and training sections are the authority. See `docs/PAPER_CODE_ALIGNMENT.md` for section/line references, explicit engineering inferences, and known gaps.
- Keep datasets, caches, checkpoints, and long logs outside the repository. Never commit tokens, private machine paths, generated weights, or data.
- Run relevant syntax and minimal real forward/backward checks for model/training changes. Do not start a full training run unless explicitly requested.
- Keep docs concise. Retain legal/attribution and data-license material if present.

For the ImageNet-only server workflow requested for this project, use `docs/TRAINING_SERVER_PROMPT_ZH.md`. It is separate from ADE20K teacher and codec training.
