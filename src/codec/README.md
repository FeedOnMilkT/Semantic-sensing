# CPU entropy codec (latent → bitstream)

## Scope (Phase A)

This package accelerates **only** the deployment path:

```text
quantized latent symbols (y, z)  →  rANS  →  bytes
bytes  →  rANS  →  reconstructed symbols
```

It does **not** change:

- fuse / `g_a` / `h_a` / semantic-compression training
- Hailo ONNX / HEF graphs
- RTFF / RTFD container layout (Phase A keeps bit-compatible packing)

Neural analysis still runs on Hailo (or GPU/CPU for training). The handoff is
`latent_y` / `latent_z` tensors; entropy coding stays on the host CPU.

## Layout

| Path | Role |
|---|---|
| `entropy_coder.py` | Select `fast` (numpy buffers → C++ rANS) or `compressai` backend |
| `patch_compressai.py` | Monkey-patch `EntropyModel.compress/decompress` to use the fast path |
| `rans_fast/` | C++ extension (CompressAI-compatible rANS64 + pybind11) |

## Build

```bash
cd /workspace/semantic-sensing
PYTHONPATH=/workspace/.python-pkgs:$PYTHONPATH \
  python src/codec/rans_fast/setup_ext.py build_ext --inplace
```

Put the resulting `.so` next to `rans_fast/` (or `pip install -e` into a persistent
`/workspace` venv — site-packages inside an ephemeral container layer is lost on
`docker rm`).

## Env

| Variable | Values | Default |
|---|---|---|
| `SEMSENSE_ENTROPY_CODER` | `fast` / `compressai` / `auto` | `auto` (fast if extension loads, else CompressAI) |

## Phase B (optional, not implemented here)

Channel/group multi-stream rANS would need a **new bitstream magic** (not Docker).
Only pursue after Phase A is measured on Pi and single-core is still insufficient.
