#!/usr/bin/env python3
"""Collect bootstrap calibration tensors for the three deployable Hailo graphs.

This is deliberately a *pre-PTQ* collector.  It sends a real CPU rANS
round-trip through the ConvBN bottleneck, then stores only the three graph
inputs in the Hailo ``load_calib.py`` / ``meta.json`` handoff format:

  analysis/image
  hyper/latent_z_hat
  synthesis_tail/latent_y_hat

The latter two are explicitly labelled ``bootstrap_fp_codec``.  They are good
for parser/NATIVE checks and the first analysis PTQ only; formal downstream
PTQ must be regenerated from quantized upstream outputs.  The matching demo
orchestrator enforces that ordering rather than silently treating these dumps
as a final chained-calibration set.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import ExifTags, Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if not hasattr(Image, "ExifTags"):
    Image.ExifTags = ExifTags

from src.datasets import build_dataset
from src.utils.config import load_config
from prepare_hailo_full_module_calib import (
    _ALIGN_ATOL,
    _DEFAULT_CONFIG,
    _ROUNDTRIP_SAMPLES,
    StreamingStats,
    _codec_roundtrip_boundary,
    _generated_loader,
    _load_student,
    _roundtrip_compare,
    _save_activation,
    _sha256,
)


_SEGMENT_INPUTS = {
    "analysis": [("image", "image")],
    "hyper": [("latent_z_hat", "latent_z_hat")],
    "synthesis_tail": [("latent_y_hat", "latent_y_hat")],
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-ckpt", type=Path, required=True)
    parser.add_argument("--segments-manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True,
                        help="new directory containing analysis/, hyper/, synthesis_tail/")
    parser.add_argument("--config", type=Path, default=Path(_DEFAULT_CONFIG))
    parser.add_argument("--dataset-split", choices=["train", "val"], default="train")
    parser.add_argument("--calib-selection", choices=["sequential", "shuffled"], default="shuffled")
    parser.add_argument("--calib-start", type=int, default=0)
    parser.add_argument("--calib-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gfa-mode", choices=["group_softmax"], default="group_softmax")
    return parser.parse_args(argv)


def _init_mapping(tensors: dict[str, torch.Tensor], segment: str) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for semantic, onnx_name in _SEGMENT_INPUTS[segment]:
        tensor = tensors[semantic]
        shape_nchw = [int(value) for value in tensor.shape]
        if len(shape_nchw) != 4:
            raise ValueError(f"{segment}/{semantic}: expected rank-4 NCHW, got {shape_nchw}")
        mapping[semantic] = {
            "onnx_name": onnx_name,
            "directory": semantic,
            "shape_nchw": shape_nchw,
            "shape_nhwc": [shape_nchw[0], shape_nchw[2], shape_nchw[3], shape_nchw[1]],
            "dtype": "float",
        }
    return mapping


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "semantic_sensing.segmented_onnx.v1":
        raise ValueError(f"unsupported segmented manifest schema: {manifest.get('schema')!r}")
    by_id = {entry.get("id"): entry for entry in manifest.get("segments", [])}
    if set(by_id) != set(_SEGMENT_INPUTS):
        raise ValueError(f"segments must be exactly {sorted(_SEGMENT_INPUTS)}, got {sorted(by_id)}")
    for segment, expected in _SEGMENT_INPUTS.items():
        onnx_path = Path(by_id[segment].get("onnx", ""))
        if not onnx_path.is_file():
            raise FileNotFoundError(f"{segment} ONNX missing: {onnx_path}")
        if by_id[segment].get("input_names") != [name for _, name in expected]:
            raise ValueError(f"{segment} ONNX input contract changed: {by_id[segment].get('input_names')}")
        if _sha256(onnx_path) != by_id[segment].get("sha256"):
            raise ValueError(f"{segment} ONNX checksum does not match manifest")
    return manifest


def _dataset_indices(dataset_len: int, args: argparse.Namespace) -> list[int]:
    available = list(range(args.calib_start, dataset_len))
    if args.calib_count <= 0 or args.calib_count > len(available):
        raise ValueError(f"invalid --calib-count={args.calib_count}; available={len(available)}")
    if args.calib_selection == "shuffled":
        return random.Random(args.seed).sample(available, args.calib_count)
    return available[:args.calib_count]


def _metadata(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    entry: dict[str, Any],
    segment: str,
    mapping: dict[str, dict[str, Any]],
    indices: list[int],
    dataset_len: int,
    statistics: dict[str, StreamingStats],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "semantic_sensing.segmented_calibration.v1",
        "segment": segment,
        "calibration_mode": "bootstrap_fp_codec",
        "formal_use": (
            "analysis input may calibrate first-stage analysis PTQ; downstream segments require "
            "quantized-upstream chained calibration materialized later"
        ),
        "checkpoint": manifest["checkpoint"],
        "model": manifest["model"],
        "onnx": {
            key: entry.get(key)
            for key in ("onnx", "sha256", "opset", "input_names", "output_names", "in_shapes_nchw", "out_shapes_nchw")
        },
        "dataset": {
            "name": "ADE20K",
            "split": args.dataset_split,
            "dataset_length": dataset_len,
            "indices": indices,
            "selection": args.calib_selection,
            "seed": args.seed,
        },
        "sample_count": len(indices),
        "preprocessing": {
            "config": str(args.config.resolve()),
            "resize_hw": [512, 512],
            "augment": False,
            "normalization": {
                "recipe": "(RGB/255 - mean) / std",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "source": "src/datasets/ade20k.py via build_dataset(config.data.val)",
            },
        },
        "tensor_mapping": mapping,
        "storage": {"dtype": "float32", "layout": "NHWC", "per_sample_npy": True},
        "codec": {
            "path": "roundtrip",
            "definition": "CPU entropy_bottleneck + gaussian_conditional compress/decompress with real rANS bytes",
            "boundary": {
                "analysis": "pre-codec image",
                "hyper": "latent_z_hat after z rANS decode",
                "synthesis_tail": "latent_y_hat after y rANS decode",
            }[segment],
        },
        "roundtrip_alignment": alignment,
        "statistics": {semantic: accumulator.result() for semantic, accumulator in statistics.items()},
        "script_parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "software": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__},
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    checkpoint = args.student_ckpt.resolve()
    manifest_path = args.segments_manifest.resolve()
    out_root = args.out_root.resolve()
    config_path = args.config.resolve()
    if out_root.exists():
        raise FileExistsError(f"refusing to overwrite {out_root}")
    if not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError(checkpoint if not checkpoint.is_file() else config_path)
    manifest = _load_manifest(manifest_path)
    if _sha256(checkpoint) != manifest.get("checkpoint", {}).get("sha256"):
        raise ValueError("checkpoint hash does not match segmented ONNX manifest")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device requests CUDA but CUDA is unavailable")
    device = torch.device(args.device)

    data_cfg = load_config(str(config_path))
    val_cfg = data_cfg.data.val
    val_cfg.args.split = args.dataset_split
    val_cfg.args.augment = False
    if val_cfg.name != "ade20k" or list(val_cfg.args.size) != [512, 512]:
        raise ValueError("requires ADE20K 512x512 validation preprocessing")
    dataset = build_dataset(val_cfg)
    indices = _dataset_indices(len(dataset), args)
    student, checkpoint_cfg = _load_student(checkpoint, device, args.gfa_mode)
    if checkpoint_cfg.bottleneck.args.get("act") != "convbn":
        raise ValueError("segmented Hailo chain requires ConvBN bottleneck")
    cpu_bottleneck = copy.deepcopy(student.bottleneck).cpu().eval()
    cpu_bottleneck.update(force=True)

    alignment_samples: list[dict[str, Any]] = []
    alignment_max = {"x_h_hat": 0.0, "x_l_hat": 0.0}
    for dataset_idx in indices[: min(_ROUNDTRIP_SAMPLES, len(indices))]:
        image = dataset[dataset_idx]["image"].unsqueeze(0).to(device)
        with torch.inference_mode():
            feats, _ = student._sender(image)
        _, diffs = _roundtrip_compare(cpu_bottleneck, tuple(t.detach().cpu() for t in feats))
        alignment_samples.append({"dataset_index": dataset_idx, "max_abs_diff": diffs})
        for key, value in diffs.items():
            alignment_max[key] = max(alignment_max[key], value)
    alignment = {
        "sample_count": len(alignment_samples),
        "atol": _ALIGN_ATOL,
        "passed": all(value <= _ALIGN_ATOL for value in alignment_max.values()),
        "max_abs_diff": alignment_max,
        "samples": alignment_samples,
    }
    if not alignment["passed"]:
        raise RuntimeError(f"CPU rANS alignment gate failed: {alignment_max}")

    temporary = out_root.parent / f".{out_root.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    entries = {entry["id"]: entry for entry in manifest["segments"]}
    mappings: dict[str, dict[str, dict[str, Any]]] = {}
    statistics: dict[str, dict[str, StreamingStats]] = {segment: {} for segment in _SEGMENT_INPUTS}
    try:
        for segment in _SEGMENT_INPUTS:
            (temporary / segment).mkdir()
        for saved_idx, dataset_idx in enumerate(indices):
            image = dataset[dataset_idx]["image"].unsqueeze(0).to(device)
            with torch.inference_mode():
                feats, _ = student._sender(image)
            boundary = _codec_roundtrip_boundary(cpu_bottleneck, tuple(t.detach().cpu() for t in feats))
            tensors = {
                "analysis": {"image": image},
                "hyper": {"latent_z_hat": boundary["latent_z_hat"]},
                "synthesis_tail": {"latent_y_hat": boundary["latent_y_hat"]},
            }
            for segment, values in tensors.items():
                if segment not in mappings:
                    mappings[segment] = _init_mapping(values, segment)
                    for semantic in mappings[segment]:
                        (temporary / segment / semantic).mkdir()
                        statistics[segment][semantic] = StreamingStats(
                            args.seed + sum(map(ord, f"{segment}:{semantic}"))
                        )
                for semantic, value in values.items():
                    _save_activation(temporary / segment, semantic, saved_idx, value,
                                     mappings[segment], statistics[segment])
            if (saved_idx + 1) % 50 == 0 or saved_idx + 1 == len(indices):
                print(f"saved {saved_idx + 1}/{len(indices)}", flush=True)

        for segment, entry in entries.items():
            segment_root = temporary / segment
            metadata = _metadata(
                args=args, manifest=manifest, entry=entry, segment=segment,
                mapping=mappings[segment], indices=indices, dataset_len=len(dataset),
                statistics=statistics[segment], alignment=alignment,
            )
            (segment_root / "meta.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (segment_root / "load_calib.py").write_text(_generated_loader(), encoding="utf-8")
        root_manifest = {
            "schema": "semantic_sensing.segmented_calibration_root.v1",
            "segments_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
            "checkpoint": manifest["checkpoint"],
            "sample_count": len(indices),
            "calibration_mode": "bootstrap_fp_codec",
            "segments": {segment: {"meta": f"{segment}/meta.json"} for segment in _SEGMENT_INPUTS},
            "formal_next_step": "PTQ analysis, then regenerate downstream calibration from quantized analysis/hyper outputs.",
        }
        (temporary / "calibration_manifest.json").write_text(
            json.dumps(root_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, out_root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    print(json.dumps({"status": "PASS", "out_root": str(out_root), "samples": len(indices),
                      "roundtrip_alignment": alignment["max_abs_diff"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
