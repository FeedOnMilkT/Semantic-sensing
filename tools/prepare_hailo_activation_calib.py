#!/usr/bin/env python
"""Collect real fused mid-split activations for Hailo module calibration.

The first supported module is ``receiver_stage45``. Inputs are produced by the
frozen deployment pipeline:

    ADE20K image -> RTFormer mid sender -> fused codec -> (x_h_hat, x_l_hat)

The first eight samples always compare eval-mode quantization with a real CPU
compress/decompress (rANS) roundtrip. ``--codec-path auto`` uses the faster eval
path only when that comparison passes; otherwise it uses real roundtrips.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _common import pick_device  # noqa: F401  also adds repository root to sys.path
import numpy as np
import torch

from src.datasets import build_dataset
from src.models import build_model
from src.models.student import build_student
from src.utils.config import load_config


_MODULE = "receiver_stage45"
_DEFAULT_CKPT = "/data/sidney/checkpoints/semantic-sensing/students/midf_b0.01_s2.pth"
_DEFAULT_CONFIG = str(Path(__file__).resolve().parents[1] / "configs/phase2_mid_fused_stage2.yaml")
_DEFAULT_OUT = "/data/sidney/hailo/shared_with_docker/calib"
_PRIMARY_ONNX = "/data/sidney/hailo/shared_with_docker/onnx/modules/receiver_stage45.onnx"
_FALLBACK_ONNX = (
    "/data/sidney/checkpoints/semantic-sensing/hailo/onnx/modules/receiver_stage45.onnx"
)
_ROUNDTRIP_SAMPLES = 8
_ALIGN_ATOL = 1e-4
_QUANTILES = (0.001, 0.01, 0.5, 0.99, 0.999)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _resolve_onnx(value: str | None) -> Path:
    candidates = [Path(value)] if value else [Path(_PRIMARY_ONNX), Path(_FALLBACK_ONNX)]
    for path in candidates:
        path = path.expanduser().resolve()
        if path.is_file():
            return path
    raise FileNotFoundError(f"receiver_stage45 ONNX not found; checked: {candidates}")


def _onnx_inputs(path: Path) -> tuple[list[dict[str, Any]], int | None]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    initializer_names = {item.name for item in model.graph.initializer}
    inputs = []
    for value in model.graph.input:
        if value.name in initializer_names:
            continue
        tensor_type = value.type.tensor_type
        shape = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(int(dim.dim_value))
            elif dim.HasField("dim_param"):
                shape.append(dim.dim_param)
            else:
                shape.append(None)
        dtype_name = onnx.TensorProto.DataType.Name(tensor_type.elem_type).lower()
        inputs.append({"name": value.name, "shape_nchw": shape, "dtype": dtype_name})
    opset = next((int(x.version) for x in model.opset_import if x.domain in ("", "ai.onnx")), None)
    if len(inputs) != 2:
        raise ValueError(f"{path} must have exactly two runtime inputs, got {inputs}")
    return inputs, opset


def _map_outputs(
    recon: tuple[torch.Tensor, torch.Tensor], contracts: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    semantics = {"x_h_hat": recon[0], "x_l_hat": recon[1]}
    mapping: dict[str, dict[str, Any]] = {}
    unused = set(semantics)
    for contract in contracts:
        expected = contract["shape_nchw"]
        matches = [
            semantic
            for semantic in unused
            if len(expected) == 4
            and all(
                exp in (None, "", actual)
                for exp, actual in zip(expected[1:], semantics[semantic].shape[1:])
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                f"cannot uniquely map ONNX input {contract['name']!r} shape {expected}; "
                f"candidate semantic shapes="
                f"{ {k: list(v.shape) for k, v in semantics.items() if k in unused} }"
            )
        semantic = matches[0]
        unused.remove(semantic)
        mapping[semantic] = {
            "onnx_name": contract["name"],
            "directory": "x_h" if semantic == "x_h_hat" else "x_l",
            "shape_nchw": [int(x) for x in semantics[semantic].shape],
            "shape_nhwc": [
                int(semantics[semantic].shape[0]),
                int(semantics[semantic].shape[2]),
                int(semantics[semantic].shape[3]),
                int(semantics[semantic].shape[1]),
            ],
            "dtype": contract["dtype"],
        }
    return mapping


@dataclass
class StreamingStats:
    seed: int
    reservoir_size: int = 1_000_000
    per_sample: int = 4096

    def __post_init__(self):
        self.count = 0
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self.total = 0.0
        self.total_sq = 0.0
        self._parts: list[np.ndarray] = []
        self._rng = np.random.default_rng(self.seed)

    def update(self, array: np.ndarray) -> None:
        flat = array.reshape(-1)
        self.count += int(flat.size)
        self.minimum = min(self.minimum, float(flat.min()))
        self.maximum = max(self.maximum, float(flat.max()))
        values64 = flat.astype(np.float64, copy=False)
        self.total += float(values64.sum())
        self.total_sq += float(np.square(values64).sum())
        take = min(self.per_sample, flat.size)
        if take == flat.size:
            sample = flat.copy()
        else:
            sample = flat[self._rng.choice(flat.size, size=take, replace=False)].copy()
        self._parts.append(sample)
        current = sum(part.size for part in self._parts)
        if current > self.reservoir_size * 2:
            merged = np.concatenate(self._parts)
            keep = self._rng.choice(merged.size, size=self.reservoir_size, replace=False)
            self._parts = [merged[keep]]

    def result(self) -> dict[str, Any]:
        if not self.count:
            raise RuntimeError("no values accumulated")
        merged = np.concatenate(self._parts)
        if merged.size > self.reservoir_size:
            keep = self._rng.choice(merged.size, size=self.reservoir_size, replace=False)
            merged = merged[keep]
        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        quantile_values = np.quantile(merged, _QUANTILES)
        return {
            "element_count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": mean,
            "std": variance**0.5,
            "quantiles": {
                "q0.1": float(quantile_values[0]),
                "q1": float(quantile_values[1]),
                "q50": float(quantile_values[2]),
                "q99": float(quantile_values[3]),
                "q99.9": float(quantile_values[4]),
            },
            "quantile_method": {
                "kind": "deterministic bounded stratified sample",
                "per_activation_max_values": self.per_sample,
                "global_max_values": self.reservoir_size,
                "sampled_values": int(merged.size),
                "seed": self.seed,
            },
        }


def _load_student(path: Path, device: torch.device):
    state = torch.load(path, map_location="cpu")
    cfg = state["cfg"]
    cfg.model.backbone.args.pretrained = None
    teacher = build_model(cfg.model).eval()
    student = build_student(cfg, teacher).eval()
    student.load_state_dict(state["student"], strict=True)
    if cfg.student.split != "mid":
        raise ValueError(f"expected student.split=mid, got {cfg.student.split!r}")
    if cfg.bottleneck.name != "rtformer_fused":
        raise ValueError(f"expected rtformer_fused, got {cfg.bottleneck.name!r}")
    if cfg.bottleneck.args.get("act") != "gdn":
        raise ValueError(f"expected GDN checkpoint, got act={cfg.bottleneck.args.get('act')!r}")
    student.update(force=True)
    student.eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    return student.to(device), cfg


def _roundtrip_compare(
    cpu_bottleneck: torch.nn.Module,
    feats_cpu: tuple[torch.Tensor, torch.Tensor],
) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, float]]:
    with torch.inference_mode():
        eval_recon, _ = cpu_bottleneck(feats_cpu)
        payload = cpu_bottleneck.compress(feats_cpu)
        roundtrip_recon = cpu_bottleneck.decompress(payload)
    diffs = {
        "x_h_hat": float((eval_recon[0] - roundtrip_recon[0]).abs().max()),
        "x_l_hat": float((eval_recon[1] - roundtrip_recon[1]).abs().max()),
    }
    return roundtrip_recon, diffs


def _roundtrip_only(
    cpu_bottleneck: torch.nn.Module,
    feats_cpu: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        payload = cpu_bottleneck.compress(feats_cpu)
        return cpu_bottleneck.decompress(payload)


def _generated_loader() -> str:
    return '''"""Load validated receiver_stage45 calibration inputs."""
import json
import os
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_ROOT, "meta.json"), encoding="utf-8") as f:
    META = json.load(f)


def load_calib(limit=None):
    """Return ``dict[ONNX input name, NHWC float32 batch]``."""
    count = int(META["sample_count"])
    if limit is not None:
        if not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative integer or None")
        selected = min(limit, count)
    else:
        selected = count
    result = {}
    for semantic, spec in META["tensor_mapping"].items():
        directory = os.path.join(_ROOT, spec["directory"])
        files = sorted(name for name in os.listdir(directory) if name.endswith(".npy"))
        if len(files) != count:
            raise ValueError(
                f"{semantic}: expected {count} files, found {len(files)} in {directory}"
            )
        expected_names = [f"{i:05d}.npy" for i in range(count)]
        if files != expected_names:
            raise ValueError(f"{semantic}: file sequence is not contiguous")
        expected_shape = tuple(spec["shape_nhwc"][1:])
        arrays = []
        for name in files[:selected]:
            array = np.load(os.path.join(directory, name), allow_pickle=False)
            if array.dtype != np.float32:
                raise TypeError(f"{semantic}/{name}: expected float32, got {array.dtype}")
            if array.shape != expected_shape:
                raise ValueError(
                    f"{semantic}/{name}: expected {expected_shape}, got {array.shape}"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"{semantic}/{name}: contains non-finite values")
            arrays.append(array)
        if arrays:
            result[spec["onnx_name"]] = np.stack(arrays, axis=0)
        else:
            result[spec["onnx_name"]] = np.empty(
                (0, *expected_shape), dtype=np.float32
            )
    return result
'''


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=[_MODULE], required=True)
    parser.add_argument("--limit", type=int, default=1024)
    parser.add_argument("--student-ckpt", default=_DEFAULT_CKPT)
    parser.add_argument("--out-dir", default=_DEFAULT_OUT, help="calibration output root")
    parser.add_argument("--codec-path", choices=["auto", "roundtrip", "eval"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--onnx", default=None, help="explicit receiver_stage45 ONNX")
    parser.add_argument("--config", default=_DEFAULT_CONFIG, help="project config for data.val")
    parser.add_argument("--device", default=None, help="torch device; default CUDA if available")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")

    repo = Path(__file__).resolve().parents[1]
    checkpoint = Path(args.student_ckpt).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    onnx_path = _resolve_onnx(args.onnx)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    data_cfg = load_config(str(config_path))
    val_cfg = data_cfg.data.val
    val_args = val_cfg.args
    if (
        val_cfg.name != "ade20k"
        or val_args.split != "val"
        or bool(val_args.augment)
        or list(val_args.size) != [512, 512]
    ):
        raise ValueError(
            "calibration requires project data.val = ADE20K val, augment=False, size=[512,512]"
        )
    dataset = build_dataset(val_cfg)
    sample_count = min(args.limit, len(dataset))
    indices = list(range(sample_count))
    contracts, opset = _onnx_inputs(onnx_path)

    print(f"device={device}; checkpoint={checkpoint}")
    print(f"onnx={onnx_path}; samples={sample_count}/{len(dataset)}")
    student, checkpoint_cfg = _load_student(checkpoint, device)
    cpu_bottleneck = copy.deepcopy(student.bottleneck).cpu().eval()
    cpu_bottleneck.update(force=True)
    for parameter in cpu_bottleneck.parameters():
        parameter.requires_grad_(False)

    alignment_samples: list[dict[str, Any]] = []
    alignment_max = {"x_h_hat": 0.0, "x_l_hat": 0.0}
    alignment_count = min(_ROUNDTRIP_SAMPLES, sample_count)
    print(f"running {alignment_count}-sample CPU rANS alignment gate", flush=True)
    for dataset_idx in indices[:alignment_count]:
        image = dataset[dataset_idx]["image"].unsqueeze(0).to(device)
        with torch.inference_mode():
            split_feats, _ = student._sender(image)
        feats_cpu = tuple(t.detach().cpu() for t in split_feats)
        _, diffs = _roundtrip_compare(cpu_bottleneck, feats_cpu)
        alignment_samples.append({"val_index": dataset_idx, "max_abs_diff": diffs})
        for semantic, value in diffs.items():
            alignment_max[semantic] = max(alignment_max[semantic], value)
    aligned = all(value <= _ALIGN_ATOL for value in alignment_max.values())
    if args.codec_path == "auto":
        formal_codec_path = "eval" if aligned else "roundtrip"
    elif args.codec_path == "eval" and not aligned:
        raise RuntimeError(
            "eval codec path requested but real roundtrip alignment failed: "
            f"{alignment_max} > atol {_ALIGN_ATOL}"
        )
    else:
        formal_codec_path = args.codec_path
    print(
        f"roundtrip alignment max={alignment_max}, atol={_ALIGN_ATOL}; "
        f"formal codec path={formal_codec_path}",
        flush=True,
    )

    out_root.mkdir(parents=True, exist_ok=True)
    target = out_root / _MODULE
    temporary = out_root / f".{_MODULE}.tmp-{os.getpid()}"
    backup = out_root / f".{_MODULE}.backup-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()

    mapping = None
    stats = {
        "x_h_hat": StreamingStats(args.seed + 101),
        "x_l_hat": StreamingStats(args.seed + 202),
    }

    try:
        for output_idx, dataset_idx in enumerate(indices):
            image = dataset[dataset_idx]["image"].unsqueeze(0).to(device)
            with torch.inference_mode():
                split_feats, _ = student._sender(image)
            if formal_codec_path == "roundtrip":
                feats_cpu = tuple(t.detach().cpu() for t in split_feats)
                recon = _roundtrip_only(cpu_bottleneck, feats_cpu)
            else:
                with torch.inference_mode():
                    recon, _ = student.bottleneck(split_feats)
                recon = tuple(t.detach().cpu() for t in recon)

            if mapping is None:
                mapping = _map_outputs(recon, contracts)
                for spec in mapping.values():
                    if spec["dtype"] != "float":
                        raise TypeError(
                            f"ONNX input {spec['onnx_name']} expects {spec['dtype']}, not float32"
                        )
                    (temporary / spec["directory"]).mkdir()

            for semantic, tensor in zip(("x_h_hat", "x_l_hat"), recon):
                spec = mapping[semantic]
                nchw = tensor.detach().cpu().numpy()
                if nchw.dtype != np.float32:
                    raise TypeError(f"{semantic}: expected float32, got {nchw.dtype}")
                if list(nchw.shape) != spec["shape_nchw"]:
                    raise ValueError(
                        f"{semantic}: expected NCHW {spec['shape_nchw']}, got {list(nchw.shape)}"
                    )
                nhwc = np.ascontiguousarray(np.transpose(nchw, (0, 2, 3, 1))[0])
                if list(nhwc.shape) != spec["shape_nhwc"][1:]:
                    raise ValueError(
                        f"{semantic}: expected NHWC {spec['shape_nhwc'][1:]}, got {list(nhwc.shape)}"
                    )
                if nhwc.dtype != np.float32 or not np.isfinite(nhwc).all():
                    raise ValueError(f"{semantic}: invalid dtype or non-finite activation")
                np.save(temporary / spec["directory"] / f"{output_idx:05d}.npy", nhwc)
                stats[semantic].update(nhwc)

            if (output_idx + 1) % 50 == 0 or output_idx + 1 == sample_count:
                print(f"saved {output_idx + 1}/{sample_count}", flush=True)

        assert mapping is not None
        checkpoint_data_args = checkpoint_cfg.data.val.args
        metadata = {
            "collector": str(Path(__file__).resolve()),
            "module": _MODULE,
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": _sha256(checkpoint),
                "identity": {
                    "student_split": checkpoint_cfg.student.split,
                    "bottleneck": checkpoint_cfg.bottleneck.name,
                    "activation": checkpoint_cfg.bottleneck.args.get("act"),
                },
            },
            "onnx": {
                "path": str(onnx_path),
                "sha256": _sha256(onnx_path),
                "opset": opset,
                "inputs": contracts,
                "original_names_note": (
                    "Use tensor_mapping[*].onnx_name as Hailo original_names/input mapping."
                ),
            },
            "dataset": {
                "name": "ADE20K",
                "hf_repo": val_args.get("hf_repo", "merve/scene_parse_150"),
                "split": "validation",
                "dataset_length": len(dataset),
                "val_indices": indices,
            },
            "sample_count": sample_count,
            "preprocessing": {
                "config": str(config_path),
                "resize_hw": [512, 512],
                "augment": False,
                "normalization": {
                    "recipe": "(RGB/255 - mean) / std",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "source": "src/datasets/ade20k.py via build_dataset(config.data.val)",
                },
                "checkpoint_data_val": {
                    "name": checkpoint_cfg.data.val.name,
                    "split": checkpoint_data_args.split,
                    "augment": bool(checkpoint_data_args.augment),
                    "size": list(checkpoint_data_args.size),
                },
            },
            "tensor_mapping": mapping,
            "storage": {"dtype": "float32", "layout": "NHWC", "per_sample_npy": True},
            "codec": {
                "requested_path": args.codec_path,
                "formal_path": formal_codec_path,
                "eval_definition": (
                    "frozen eval-mode CompressAI quantization path; enabled only after alignment gate"
                ),
                "roundtrip_definition": "CPU compress()/decompress() with real rANS byte stream",
            },
            "roundtrip_alignment": {
                "sample_count": len(alignment_samples),
                "atol": _ALIGN_ATOL,
                "passed": all(value <= _ALIGN_ATOL for value in alignment_max.values()),
                "max_abs_diff": alignment_max,
                "samples": alignment_samples,
            },
            "statistics": {semantic: accumulator.result() for semantic, accumulator in stats.items()},
            "seed": args.seed,
            "script_parameters": vars(args),
            "git_commit": _git_commit(repo),
            "software": {
                "python": sys.version,
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
        }
        with (temporary / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        (temporary / "load_calib.py").write_text(_generated_loader(), encoding="utf-8")

        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(temporary, target)
        except BaseException:
            if backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    print(f"complete: {target}")
    print(f"formal codec path: {formal_codec_path}")
    print(f"roundtrip max abs diff: {alignment_max}")


if __name__ == "__main__":
    main()
