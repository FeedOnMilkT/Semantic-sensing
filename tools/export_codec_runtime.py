#!/usr/bin/env python3
"""Export the CPU entropy-coding state required by the Pi full-module runtime.

The Hailo graphs cover the neural transforms but not CompressAI's learned
entropy models. A board runtime therefore needs the updated CDF tables, offsets,
entropy-bottleneck medians, Gaussian scale table and RTFF contract from the
exact student checkpoint. This tool writes NumPy-only immutable runtime assets.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import torch

from tools.export_modules_fused import _sha256_file
from src.models import build_model
from src.models.student import build_student


SCHEMA_VERSION = 1
RTFF_HEADER_BYTES = 21  # magic + eight u16 dimensions + batch u8


def _array(value, dtype: np.dtype) -> np.ndarray:
    return np.ascontiguousarray(value.detach().cpu().numpy(), dtype=dtype)


def _remap_legacy_entropy_keys(state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
    """Adapt CompressAI's old ``_matrix0`` naming to ParameterLists.

    CDF buffers are intentionally empty in ordinary training checkpoints and
    must be regenerated.  The learned entropy transform parameters themselves
    are semantically unchanged across this rename.
    """
    pattern = re.compile(r"^(bottleneck\.entropy_bottleneck)\._(matrix|bias|factor)(\d+)$")
    plurals = {"matrix": "matrices", "bias": "biases", "factor": "factors"}
    mapped: dict[str, torch.Tensor] = {}
    changed = 0
    for key, value in state.items():
        match = pattern.match(key)
        if match:
            key = f"{match.group(1)}.{plurals[match.group(2)]}.{match.group(3)}"
            changed += 1
        mapped[key] = value
    return mapped, changed


def _atomic_write_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def export_runtime(student_ckpt: Path, out_dir: Path, *, height: int, width: int) -> dict:
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing runtime asset directory: {out_dir}")
    if height <= 0 or width <= 0 or height % 32 or width % 32:
        raise ValueError("height and width must be positive multiples of 32")
    if not student_ckpt.is_file():
        raise FileNotFoundError(student_ckpt)

    # Ordinary training checkpoints intentionally store empty CDF buffers. The
    # project crossed a CompressAI parameter-name transition (``_matrixN`` ->
    # ``matrices.N``), so adapt just those learned entropy parameters before
    # materializing the exact tables required by the deployment runtime.
    checkpoint = torch.load(student_ckpt, map_location="cpu", weights_only=False)
    if "cfg" not in checkpoint or "student" not in checkpoint:
        raise KeyError("expected a student checkpoint with cfg and student entries")
    cfg = checkpoint["cfg"]
    act = cfg.bottleneck.args.get("act", "convbn")
    if act != "convbn":
        raise ValueError(f"expected ConvBN deployment checkpoint, found act={act!r}")

    cfg.model.backbone.args.pretrained = None
    teacher = build_model(cfg.model).eval()
    student = build_student(cfg, teacher).eval()
    compatible_state, renamed = _remap_legacy_entropy_keys(checkpoint["student"])
    loaded = student.load_state_dict(compatible_state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"incomplete entropy checkpoint compatibility load: {loaded}")
    bottleneck = student.bottleneck
    bottleneck.update(force=True)
    z_model, y_model = bottleneck.entropy_bottleneck, bottleneck.gaussian_conditional
    arrays = {
        "z_cdf": _array(z_model._quantized_cdf, np.int32),
        "z_cdf_lengths": _array(z_model._cdf_length, np.int32).reshape(-1),
        "z_offsets": _array(z_model._offset, np.int32).reshape(-1),
        "z_medians_nchw": _array(z_model._get_medians(), np.float32).reshape(1, -1, 1, 1),
        "y_cdf": _array(y_model._quantized_cdf, np.int32),
        "y_cdf_lengths": _array(y_model._cdf_length, np.int32).reshape(-1),
        "y_offsets": _array(y_model._offset, np.int32).reshape(-1),
        "y_scale_table": _array(y_model.scale_table, np.float32).reshape(-1),
        "y_scale_bound": np.asarray([float(y_model.scale_bound)], dtype=np.float32),
    }
    if arrays["z_cdf"].size == 0 or arrays["y_cdf"].size == 0:
        raise RuntimeError("entropy CDF update produced empty tables")

    y_channels = int(cfg.bottleneck.args.get("latent_channels", 192))
    z_channels = int(cfg.bottleneck.args.get("hyper_channels", 128))
    y_shape_nchw = [1, y_channels, height // 32, width // 32]
    z_shape_nchw = [1, z_channels, height // 64, width // 64]
    if tuple(arrays["z_medians_nchw"].shape) != (1, z_channels, 1, 1):
        raise ValueError(f"unexpected z medians shape: {arrays['z_medians_nchw'].shape}")
    if arrays["z_cdf"].shape[0] != z_channels:
        raise ValueError(f"z CDF rows do not match channels: {arrays['z_cdf'].shape}")

    out_dir.mkdir(parents=True)
    tables = out_dir / "codec_tables.npz"
    np.savez_compressed(tables, **arrays)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "kind": "semantic_sensing_pi_codec_runtime",
        "checkpoint": {"path": str(student_ckpt.resolve()), "sha256": _sha256_file(str(student_ckpt))},
        "model": {
            "activation": act, "split": str(cfg.student.split),
            "bottleneck": str(cfg.bottleneck.name),
            "gfa_mode": str(cfg.model.backbone.args.get("gfa_mode", "original")),
            "latent_channels": y_channels, "hyper_channels": z_channels,
            "image_nchw": [1, 3, height, width], "y_nchw": y_shape_nchw, "z_nchw": z_shape_nchw,
        },
        "tables": {
            "file": tables.name, "sha256": _sha256_file(str(tables)),
            "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                       for name, value in arrays.items()},
        },
        "cdf_provenance": {
            "source": "generated_by_CompressAI_update_from_checkpoint_entropy_parameters",
            "legacy_entropy_parameter_renames": renamed,
            "checkpoint_cdf_buffers_were_empty": True,
        },
        "quantization": {
            "z": "symbols = round(z - medians); z_hat = symbols + medians",
            "y": "symbols = round(y - means); y_hat = symbols + means",
            "rounding": "IEEE-754 round-to-nearest-even (numpy.rint / torch.round for finite values)",
            "gaussian_indexes": "clamp(scales, scale_bound), then CompressAI scale-table lookup",
            "cdf_precision_bits": 16,
        },
        "container": {
            "magic_ascii": "RTFF", "byte_order": "big-endian", "header_bytes": RTFF_HEADER_BYTES,
            "layout": "magic | z_h,z_w,y_h,y_w,x_h,x_w,x_l,x_w (u16) | batch (u8) | z strings | y strings; each string u32 length + bytes",
            "z_then_y": True,
        },
    }
    _atomic_write_json(out_dir / "codec_runtime.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()
    metadata = export_runtime(args.student_ckpt, args.out_dir, height=args.height, width=args.width)
    print(json.dumps({"out_dir": str(args.out_dir), "checkpoint_sha256": metadata["checkpoint"]["sha256"],
                      "tables_sha256": metadata["tables"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
