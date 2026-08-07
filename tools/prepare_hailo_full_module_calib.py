#!/usr/bin/env python
"""Collect real per-module activations for formal Hailo full-module PTQ.

Loads the trusted fused mid + group_softmax GFA student checkpoint and walks the
frozen deployment pipeline on selected ADE20K split indices to emit NHWC float32
per-sample activations for every exported module boundary.

Codec boundary policy:
  * Always gate on >=8 CPU rANS compress()/decompress() samples.
  * If eval-mode reconstruction matches roundtrip within atol=1e-4 on both
    x_h_hat and x_l_hat, formal collection may use the eval path.
  * Otherwise formal collection must use real roundtrip for all samples.
  * bn_h_s calib uses latent_z_hat; bn_g_s uses latent_y_hat; receiver inputs
    come from the same formal codec reconstruction boundary.
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

from _common import pick_device  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from PIL import ExifTags, Image

# datasets 2.21 looks up ``PIL.Image.ExifTags``; Pillow 9 exposes the same
# enum as ``PIL.ExifTags``. Keep this collector usable in the pinned Hailo
# environment without changing image preprocessing or decoded pixels.
if not hasattr(Image, "ExifTags"):
    Image.ExifTags = ExifTags

from src.datasets import build_dataset
from src.models import build_model
from src.models.student import build_student
from src.utils.config import load_config


_MODULES = (
    "sender_backbone",
    "bn_align_h",
    "bn_g_a",
    "bn_h_a",
    "bn_h_s",
    "bn_g_s",
    "receiver_stage4",
    "receiver_stage5",
    "receiver_dappm",
    "receiver_head",
)

# semantic tensor name -> ONNX input name (must match export_modules_fused.py)
_MODULE_INPUTS: dict[str, list[tuple[str, str]]] = {
    "sender_backbone": [("image", "image")],
    "bn_align_h": [("x_h", "x_h")],
    "bn_g_a": [("aligned_x_h", "aligned_x_h"), ("x_l", "x_l")],
    "bn_h_a": [("latent_y", "latent_y")],
    "bn_h_s": [("latent_z_hat", "latent_z_hat")],
    "bn_g_s": [("latent_y_hat", "latent_y_hat")],
    # receiver_stage4 ONNX keeps historical names x_h/x_l; semantic is reconstructed.
    "receiver_stage4": [("x_h_hat", "x_h"), ("x_l_hat", "x_l")],
    "receiver_stage5": [("stage4_high", "stage4_high"), ("stage4_low", "stage4_low")],
    "receiver_dappm": [("high_feat", "high_feat"), ("low_feat", "low_feat")],
    "receiver_head": [("s8", "s8")],
}

_ROUNDTRIP_SAMPLES = 8
_ALIGN_ATOL = 1e-4
_QUANTILES = (0.001, 0.01, 0.5, 0.99, 0.999)
_DEFAULT_CONFIG = str(
    Path(__file__).resolve().parents[1] / "configs/phase2_mid_fused_stage2_gfa_groupsoftmax.yaml"
)
_GOLDEN_COUNT = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _tensor_to_nhwc(nchw: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(nchw, torch.Tensor):
        arr = nchw.detach().float().cpu().numpy()
    else:
        arr = np.asarray(nchw, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"expected NCHW rank-4, got shape {arr.shape}")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return np.ascontiguousarray(np.transpose(arr, (0, 2, 3, 1))[0])


def _nchw_list(t: torch.Tensor) -> list[int]:
    return [int(x) for x in t.shape]


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
            "finite": True,
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


def _load_student(path: Path, device: torch.device, gfa_mode: str):
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state["cfg"]
    cfg.model.backbone.args.pretrained = None
    cfg.model.backbone.args.gfa_mode = gfa_mode
    teacher = build_model(cfg.model).eval()
    student = build_student(cfg, teacher).eval()
    student.load_state_dict(state["student"], strict=True)
    if cfg.student.split != "mid":
        raise ValueError(f"expected student.split=mid, got {cfg.student.split!r}")
    if cfg.bottleneck.name != "rtformer_fused":
        raise ValueError(f"expected rtformer_fused, got {cfg.bottleneck.name!r}")
    act = cfg.bottleneck.args.get("act")
    if act not in ("gdn", "convbn"):
        raise ValueError(f"expected act in {{gdn, convbn}}, got act={act!r}")
    if cfg.model.backbone.args.get("gfa_mode") != "group_softmax":
        raise ValueError(
            f"expected gfa_mode=group_softmax, got {cfg.model.backbone.args.get('gfa_mode')!r}"
        )
    student.update(force=True)
    student.eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    return student.to(device), cfg


def _aligned_x_h(bn, x_h: torch.Tensor, x_l: torch.Tensor) -> torch.Tensor:
    aligned = bn.align_h(x_h)
    if aligned.shape[-2:] != x_l.shape[-2:]:
        aligned = F.interpolate(
            aligned, size=x_l.shape[-2:], mode="bilinear", align_corners=False
        )
    return aligned


def _codec_eval_boundary(bn, feats: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
    x_h, x_l = feats
    aligned = _aligned_x_h(bn, x_h, x_l)
    y = bn.g_a(torch.cat([aligned, x_l], dim=1))
    if bn.attn_a is not None:
        y = bn.attn_a(y)
    z = bn.h_a(y)
    z_hat, _ = bn.entropy_bottleneck(z)
    means, scales = bn._hyper_params(z_hat, y.shape[-2:])
    y_hat, _ = bn.gaussian_conditional(y, scales, means=means)
    x_h_hat, x_l_hat = bn._synthesize(y_hat, x_h.shape[-2:], x_l.shape[-2:])
    return {
        "aligned_x_h": aligned,
        "latent_y": y,
        "latent_z": z,
        "latent_z_hat": z_hat,
        "latent_y_hat": y_hat,
        "means": means,
        "scales": scales,
        "x_h_hat": x_h_hat,
        "x_l_hat": x_l_hat,
    }


def _codec_roundtrip_boundary(bn, feats: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Mirror compress()/decompress() while capturing z_hat / y_hat."""
    x_h, x_l = feats
    aligned = _aligned_x_h(bn, x_h, x_l)
    y = bn.g_a(torch.cat([aligned, x_l], dim=1))
    if bn.attn_a is not None:
        y = bn.attn_a(y)
    z = bn.h_a(y)
    z_strings = bn.entropy_bottleneck.compress(z)
    z_hat = bn.entropy_bottleneck.decompress(z_strings, z.shape[-2:])
    means, scales = bn._hyper_params(z_hat, y.shape[-2:])
    indexes = bn.gaussian_conditional.build_indexes(scales)
    y_strings = bn.gaussian_conditional.compress(y, indexes, means=means)
    y_hat = bn.gaussian_conditional.decompress(y_strings, indexes, means=means)
    x_h_hat, x_l_hat = bn._synthesize(y_hat, x_h.shape[-2:], x_l.shape[-2:])
    return {
        "aligned_x_h": aligned,
        "latent_y": y,
        "latent_z": z,
        "latent_z_hat": z_hat,
        "latent_y_hat": y_hat,
        "means": means,
        "scales": scales,
        "x_h_hat": x_h_hat,
        "x_l_hat": x_l_hat,
    }


def _roundtrip_compare(cpu_bn, feats_cpu):
    with torch.inference_mode():
        eval_b = _codec_eval_boundary(cpu_bn, feats_cpu)
        rt_b = _codec_roundtrip_boundary(cpu_bn, feats_cpu)
    diffs = {
        "x_h_hat": float((eval_b["x_h_hat"] - rt_b["x_h_hat"]).abs().max()),
        "x_l_hat": float((eval_b["x_l_hat"] - rt_b["x_l_hat"]).abs().max()),
    }
    return rt_b, diffs


def _onnx_manifest_map(onnx_dir: Path) -> dict[str, dict[str, Any]]:
    manifest_path = onnx_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_name = {m["name"]: m for m in manifest["modules"]}
    missing = [name for name in _MODULES if name not in by_name]
    if missing:
        raise FileNotFoundError(f"onnx manifest missing modules: {missing}")
    for name in _MODULES:
        path = Path(by_name[name]["onnx"])
        if not path.is_file():
            # tolerate relative names under onnx_dir
            alt = onnx_dir / f"{name}.onnx"
            if not alt.is_file():
                raise FileNotFoundError(path)
            by_name[name]["onnx"] = str(alt)
            by_name[name]["sha256"] = _sha256(alt)
    return by_name


def _generated_loader() -> str:
    return '''"""Load validated per-module Hailo calibration inputs."""
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
            result[spec["onnx_name"]] = np.empty((0, *expected_shape), dtype=np.float32)
    return result
'''


def _save_activation(
    root: Path,
    semantic: str,
    index: int,
    tensor: torch.Tensor,
    mapping: dict[str, dict[str, Any]],
    stats: dict[str, StreamingStats],
) -> np.ndarray:
    spec = mapping[semantic]
    nchw = tensor.detach().float().cpu()
    if list(nchw.shape) != spec["shape_nchw"]:
        # lock shape on first sample; subsequent must match
        raise ValueError(
            f"{semantic}: expected NCHW {spec['shape_nchw']}, got {list(nchw.shape)}"
        )
    nhwc = _tensor_to_nhwc(nchw)
    if list(nhwc.shape) != spec["shape_nhwc"][1:]:
        raise ValueError(
            f"{semantic}: expected NHWC {spec['shape_nhwc'][1:]}, got {list(nhwc.shape)}"
        )
    if nhwc.dtype != np.float32 or not np.isfinite(nhwc).all():
        raise ValueError(f"{semantic}: invalid dtype or non-finite activation")
    out = root / spec["directory"] / f"{index:05d}.npy"
    np.save(out, nhwc)
    stats[semantic].update(nhwc)
    return nhwc


def _init_mapping_for_module(
    module: str, tensors: dict[str, torch.Tensor]
) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for semantic, onnx_name in _MODULE_INPUTS[module]:
        t = tensors[semantic]
        shape_nchw = _nchw_list(t)
        mapping[semantic] = {
            "onnx_name": onnx_name,
            "directory": semantic,
            "shape_nchw": shape_nchw,
            "shape_nhwc": [shape_nchw[0], shape_nchw[2], shape_nchw[3], shape_nchw[1]],
            "dtype": "float",
        }
    return mapping


def _forward_module_chain(student, boundary: dict[str, torch.Tensor], x_h, x_l, image):
    """Produce all module inputs and selected outputs for one sample."""
    bb = student.teacher.backbone
    with torch.inference_mode():
        stage4_high, stage4_low = bb.stage4(boundary["x_h_hat"], boundary["x_l_hat"])
        high_feat, low_feat = bb.stage5(stage4_high, stage4_low)
        ctx = F.interpolate(
            bb.dappm(low_feat),
            size=high_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        s8 = F.relu(high_feat + ctx, inplace=False)
        feats = {"s8": s8}
        logits = student.teacher.heads["seg"](
            student.teacher.decoder(student.teacher.neck(feats))
        )
    module_inputs = {
        "sender_backbone": {"image": image},
        "bn_align_h": {"x_h": x_h},
        "bn_g_a": {"aligned_x_h": boundary["aligned_x_h"], "x_l": x_l},
        "bn_h_a": {"latent_y": boundary["latent_y"]},
        "bn_h_s": {"latent_z_hat": boundary["latent_z_hat"]},
        "bn_g_s": {"latent_y_hat": boundary["latent_y_hat"]},
        "receiver_stage4": {"x_h_hat": boundary["x_h_hat"], "x_l_hat": boundary["x_l_hat"]},
        "receiver_stage5": {"stage4_high": stage4_high, "stage4_low": stage4_low},
        "receiver_dappm": {"high_feat": high_feat, "low_feat": low_feat},
        "receiver_head": {"s8": s8},
    }
    module_outputs = {
        "sender_backbone": {"x_h": x_h, "x_l": x_l},
        "bn_align_h": {"aligned_x_h": boundary["aligned_x_h"]},
        "bn_g_a": {"latent_y": boundary["latent_y"]},
        "bn_h_a": {"latent_z": boundary["latent_z"]},
        "bn_h_s": {"means": boundary["means"], "scales": boundary["scales"]},
        "bn_g_s": {"x_h_hat": boundary["x_h_hat"], "x_l_hat": boundary["x_l_hat"]},
        "receiver_stage4": {"stage4_high": stage4_high, "stage4_low": stage4_low},
        "receiver_stage5": {"high_feat": high_feat, "low_feat": low_feat},
        "receiver_dappm": {"s8": s8},
        "receiver_head": {"seg_logits": logits},
    }
    return module_inputs, module_outputs, logits


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-ckpt", required=True)
    parser.add_argument("--onnx-dir", required=True, help="directory with per-module ONNX + manifest")
    parser.add_argument("--calib-out", required=True, help="calibration root (<handoff>/calib)")
    parser.add_argument("--eval-out", default=None, help="optional evaluation root (<handoff>/evaluation)")
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument(
        "--dataset-split",
        choices=["train", "val"],
        default="val",
        help="ADE20K source split; default preserves the historical validation protocol.",
    )
    parser.add_argument(
        "--calib-selection",
        choices=["sequential", "shuffled"],
        default="sequential",
        help="Select the calibration indices in order or as a deterministic seed-controlled subset.",
    )
    parser.add_argument("--calib-start", type=int, default=0)
    parser.add_argument("--calib-count", type=int, default=1024)
    parser.add_argument("--eval-start", type=int, default=1024)
    parser.add_argument("--eval-count", type=int, default=976)
    parser.add_argument("--codec-path", choices=["auto", "roundtrip", "eval"], default="auto")
    parser.add_argument("--gfa-mode", default="group_softmax", choices=["group_softmax"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--modules", nargs="*", default=list(_MODULES))
    parser.add_argument(
        "--skip-calib-write",
        action="store_true",
        help=(
            "Reuse an existing calibration tree under --calib-out. Still applies "
            "--calib-start/--calib-count for held-out meta and rANS alignment indices; "
            "does not rewrite calibration tensors."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo = Path(__file__).resolve().parents[1]
    checkpoint = Path(args.student_ckpt).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    onnx_dir = Path(args.onnx_dir).expanduser().resolve()
    calib_root = Path(args.calib_out).expanduser().resolve()
    eval_root = Path(args.eval_out).expanduser().resolve() if args.eval_out else None
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    modules = list(args.modules)
    unknown = [m for m in modules if m not in _MODULES]
    if unknown:
        raise ValueError(f"unknown modules: {unknown}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    data_cfg = load_config(str(config_path))
    val_cfg = data_cfg.data.val
    val_args = val_cfg.args
    val_args.split = args.dataset_split
    val_args.augment = False
    if (
        val_cfg.name != "ade20k"
        or bool(val_args.augment)
        or list(val_args.size) != [512, 512]
    ):
        raise ValueError(
            "requires ADE20K, augment=False, size=[512,512]"
        )
    dataset = build_dataset(val_cfg)
    if len(dataset) < args.calib_start + args.calib_count:
        raise ValueError(
            f"dataset length {len(dataset)} too small for calib "
            f"{args.calib_start}..{args.calib_start + args.calib_count - 1}"
        )
    available_indices = list(range(args.calib_start, len(dataset)))
    if args.calib_count > len(available_indices):
        raise ValueError("requested calibration count exceeds available selected split indices")
    if args.calib_selection == "shuffled":
        calib_indices = random.Random(args.seed).sample(available_indices, args.calib_count)
    else:
        calib_indices = available_indices[: args.calib_count]
    eval_indices = []
    if eval_root is not None:
        if len(dataset) < args.eval_start + args.eval_count:
            raise ValueError("dataset too small for held-out evaluation")
        eval_indices = list(range(args.eval_start, args.eval_start + args.eval_count))
        overlap = sorted(set(calib_indices) & set(eval_indices))
        if overlap:
            raise RuntimeError(f"calib/eval index overlap: {overlap[:16]}...")

    onnx_by_name = _onnx_manifest_map(onnx_dir)
    print(f"device={device}; checkpoint={checkpoint}")
    print(f"calib samples={len(calib_indices)}; eval samples={len(eval_indices)}")
    student, checkpoint_cfg = _load_student(checkpoint, device, args.gfa_mode)
    cpu_bottleneck = copy.deepcopy(student.bottleneck).cpu().eval()
    cpu_bottleneck.update(force=True)
    for parameter in cpu_bottleneck.parameters():
        parameter.requires_grad_(False)

    # ---- rANS alignment gate on first 8 calib samples ----
    alignment_samples: list[dict[str, Any]] = []
    alignment_max = {"x_h_hat": 0.0, "x_l_hat": 0.0}
    alignment_count = min(_ROUNDTRIP_SAMPLES, len(calib_indices))
    print(f"running {alignment_count}-sample CPU rANS alignment gate", flush=True)
    for dataset_idx in calib_indices[:alignment_count]:
        image = dataset[dataset_idx]["image"].unsqueeze(0).to(device)
        with torch.inference_mode():
            split_feats, _ = student._sender(image)
        feats_cpu = tuple(t.detach().cpu() for t in split_feats)
        _, diffs = _roundtrip_compare(cpu_bottleneck, feats_cpu)
        alignment_samples.append({"dataset_index": dataset_idx, "max_abs_diff": diffs})
        for semantic, value in diffs.items():
            alignment_max[semantic] = max(alignment_max[semantic], value)
    aligned = all(value <= _ALIGN_ATOL for value in alignment_max.values())
    if args.codec_path == "auto":
        formal_codec_path = "eval" if aligned else "roundtrip"
    elif args.codec_path == "eval" and not aligned:
        raise RuntimeError(
            f"eval codec path requested but roundtrip alignment failed: {alignment_max}"
        )
    else:
        formal_codec_path = args.codec_path
    print(
        f"roundtrip alignment max={alignment_max}, atol={_ALIGN_ATOL}; "
        f"formal codec path={formal_codec_path}",
        flush=True,
    )

    script_sha = _sha256(Path(__file__).resolve())
    checkpoint_sha = _sha256(checkpoint)

    def resolve_boundary(feats_dev):
        if formal_codec_path == "roundtrip":
            feats_cpu = tuple(t.detach().cpu() for t in feats_dev)
            boundary = _codec_roundtrip_boundary(cpu_bottleneck, feats_cpu)
            # move tensors back to device for receiver chain
            return {k: v.to(device) for k, v in boundary.items()}
        with torch.inference_mode():
            return _codec_eval_boundary(student.bottleneck, feats_dev)

    if args.skip_calib_write:
        missing = [m for m in modules if not (calib_root / m / "meta.json").is_file()]
        if missing:
            raise FileNotFoundError(
                f"--skip-calib-write requires existing calib modules under {calib_root}: {missing}"
            )
        print(
            f"skipping calib write; reusing existing tree at {calib_root} "
            f"(declared indices {calib_indices[0]}..{calib_indices[-1]})",
            flush=True,
        )
    else:
        # ---- prepare calib temp dirs ----
        calib_root.mkdir(parents=True, exist_ok=True)
        temp_root = calib_root / f".tmp-full-{os.getpid()}"
        if temp_root.exists():
            shutil.rmtree(temp_root)
        temp_root.mkdir()
        module_dirs = {m: temp_root / m for m in modules}
        for d in module_dirs.values():
            d.mkdir()

        mappings: dict[str, dict[str, dict[str, Any]]] = {}
        stats: dict[str, dict[str, StreamingStats]] = {
            m: {} for m in modules
        }
        seed_base = args.seed + 1000

        try:
            for output_idx, dataset_idx in enumerate(calib_indices):
                sample = dataset[dataset_idx]
                image = sample["image"].unsqueeze(0).to(device)
                with torch.inference_mode():
                    (x_h, x_l), _ = student._sender(image)
                boundary = resolve_boundary((x_h, x_l))
                module_inputs, _, _ = _forward_module_chain(
                    student, boundary, x_h, x_l, image
                )

                for module in modules:
                    tensors = module_inputs[module]
                    if module not in mappings:
                        mappings[module] = _init_mapping_for_module(module, tensors)
                        for semantic in mappings[module]:
                            (module_dirs[module] / semantic).mkdir(exist_ok=True)
                            stats[module][semantic] = StreamingStats(
                                seed_base + (sum(ord(c) for c in (module + ":" + semantic)) % 10_000)
                            )
                    for semantic, tensor in tensors.items():
                        _save_activation(
                            module_dirs[module],
                            semantic,
                            output_idx,
                            tensor,
                            mappings[module],
                            stats[module],
                        )

                if (output_idx + 1) % 50 == 0 or output_idx + 1 == len(calib_indices):
                    print(f"calib saved {output_idx + 1}/{len(calib_indices)}", flush=True)

            for module in modules:
                onnx_info = onnx_by_name[module]
                onnx_path = Path(onnx_info["onnx"])
                metadata = {
                    "collector": str(Path(__file__).resolve()),
                    "collector_sha256": script_sha,
                    "module": module,
                    "checkpoint": {
                        "path": str(checkpoint),
                        "sha256": checkpoint_sha,
                        "identity": {
                            "student_split": checkpoint_cfg.student.split,
                            "bottleneck": checkpoint_cfg.bottleneck.name,
                            "activation": checkpoint_cfg.bottleneck.args.get("act"),
                            "gfa_mode": checkpoint_cfg.model.backbone.args.get("gfa_mode"),
                        },
                    },
                    "onnx": {
                        "path": str(onnx_path),
                        "sha256": onnx_info.get("sha256") or _sha256(onnx_path),
                        "opset": onnx_info.get("opset", 13),
                        "input_names": onnx_info.get("input_names"),
                        "output_names": onnx_info.get("output_names"),
                        "in_shapes_nchw": onnx_info.get("in_shapes_nchw"),
                        "original_names_note": (
                            "Use tensor_mapping[*].onnx_name as Hailo original_names/input mapping."
                        ),
                    },
                    "dataset": {
                        "name": "ADE20K",
                        "hf_repo": val_args.get("hf_repo", "merve/scene_parse_150"),
                        "split": args.dataset_split,
                        "dataset_length": len(dataset),
                        "indices": calib_indices,
                    },
                    "sample_count": len(calib_indices),
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
                    },
                    "tensor_mapping": mappings[module],
                    "storage": {"dtype": "float32", "layout": "NHWC", "per_sample_npy": True},
                    "codec": {
                        "requested_path": args.codec_path,
                        "formal_path": formal_codec_path,
                        "boundary_note": (
                            "bn_h_s uses latent_z_hat; bn_g_s uses latent_y_hat; "
                            "receiver_stage4 uses reconstructed x_h_hat/x_l_hat"
                        ),
                        "eval_definition": (
                            "frozen eval-mode CompressAI quantization path; "
                            "enabled only after alignment gate"
                        ),
                        "roundtrip_definition": (
                            "CPU compress()/decompress() with real rANS byte stream"
                        ),
                    },
                    "roundtrip_alignment": {
                        "sample_count": len(alignment_samples),
                        "atol": _ALIGN_ATOL,
                        "passed": aligned,
                        "max_abs_diff": alignment_max,
                        "samples": alignment_samples,
                    },
                    "statistics": {
                        semantic: accumulator.result()
                        for semantic, accumulator in stats[module].items()
                    },
                    "seed": args.seed,
                    "script_parameters": {
                        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                    },
                    "git_commit": _git_commit(repo),
                    "software": {
                        "python": sys.version,
                        "torch": torch.__version__,
                        "numpy": np.__version__,
                    },
                }
                with (module_dirs[module] / "meta.json").open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                (module_dirs[module] / "load_calib.py").write_text(
                    _generated_loader(), encoding="utf-8"
                )

                target = calib_root / module
                if target.exists():
                    raise FileExistsError(
                        f"refusing to overwrite existing calib module dir: {target}"
                    )
                os.replace(module_dirs[module], target)
                print(f"complete calib module: {target}", flush=True)

            # remove empty temp shell
            if temp_root.exists():
                shutil.rmtree(temp_root)

        except BaseException:
            if temp_root.exists():
                shutil.rmtree(temp_root)
            raise

    # ---- held-out evaluation package ----
    if eval_root is not None:
        images_dir = eval_root / "images"
        gt_dir = eval_root / "ground_truth"
        golden_dir = eval_root / "golden"
        images_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        golden_dir.mkdir(parents=True, exist_ok=True)

        golden_indices = eval_indices[:_GOLDEN_COUNT]
        golden_manifest: dict[str, Any] = {
            "checkpoint_sha256": checkpoint_sha,
            "formal_codec_path": formal_codec_path,
            "roundtrip_alignment": {
                "passed": aligned,
                "max_abs_diff": alignment_max,
                "atol": _ALIGN_ATOL,
                "samples": alignment_samples,
            },
            "golden_indices": golden_indices,
            "tensors": {},
        }

        print(f"writing held-out evaluation ({len(eval_indices)} images)", flush=True)
        for i, dataset_idx in enumerate(eval_indices):
            sample = dataset[dataset_idx]
            image_t = sample["image"]
            seg = sample["targets"]["seg"]
            image_nchw = image_t.detach().float().cpu().numpy().astype(np.float32)
            if image_nchw.shape != (3, 512, 512):
                raise ValueError(f"unexpected image shape {image_nchw.shape}")
            if not np.isfinite(image_nchw).all():
                raise ValueError(f"non-finite image at val_index={dataset_idx}")
            np.save(images_dir / f"{dataset_idx}.npy", image_nchw)
            seg_np = seg.detach().cpu().numpy()
            np.save(gt_dir / f"{dataset_idx}.npy", seg_np)

            if dataset_idx in golden_indices:
                image = image_t.unsqueeze(0).to(device)
                with torch.inference_mode():
                    (x_h, x_l), _ = student._sender(image)
                boundary = resolve_boundary((x_h, x_l))
                module_inputs, module_outputs, logits = _forward_module_chain(
                    student, boundary, x_h, x_l, image
                )
                sample_dir = golden_dir / str(dataset_idx)
                sample_dir.mkdir(exist_ok=True)
                for module in _MODULES:
                    mod_dir = sample_dir / module
                    mod_dir.mkdir(exist_ok=True)
                    for kind, bundle in (
                        ("inputs", module_inputs[module]),
                        ("outputs", module_outputs[module]),
                    ):
                        kind_dir = mod_dir / kind
                        kind_dir.mkdir(exist_ok=True)
                        for name, tensor in bundle.items():
                            arr = tensor.detach().float().cpu().numpy().astype(np.float32)
                            path = kind_dir / f"{name}.npy"
                            np.save(path, arr)
                            rel = str(path.relative_to(eval_root))
                            golden_manifest["tensors"][rel] = {
                                "val_index": dataset_idx,
                                "module": module,
                                "kind": kind,
                                "name": name,
                                "layout": "NCHW",
                                "shape": list(arr.shape),
                                "dtype": "float32",
                                "sha256": _sha256(path),
                            }
                logits_path = sample_dir / "seg_logits.npy"
                logits_np = logits.detach().float().cpu().numpy().astype(np.float32)
                np.save(logits_path, logits_np)
                rel = str(logits_path.relative_to(eval_root))
                golden_manifest["tensors"][rel] = {
                    "val_index": dataset_idx,
                    "module": "receiver_head",
                    "kind": "final",
                    "name": "seg_logits",
                    "layout": "NCHW",
                    "shape": list(logits_np.shape),
                    "dtype": "float32",
                    "sha256": _sha256(logits_path),
                }

            if (i + 1) % 50 == 0 or i + 1 == len(eval_indices):
                print(f"eval saved {i + 1}/{len(eval_indices)}", flush=True)

        eval_meta = {
            "dataset": "ADE20K",
            "split": "validation",
            "val_indices": eval_indices,
            "sample_count": len(eval_indices),
            "calibration_indices": calib_indices,
            "intersection_with_calibration": sorted(set(calib_indices) & set(eval_indices)),
            "intersection_empty": len(set(calib_indices) & set(eval_indices)) == 0,
            "preprocessing": {
                "resize_hw": [512, 512],
                "augment": False,
                "normalization": {
                    "recipe": "(RGB/255 - mean) / std",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                "image_storage": {
                    "path_pattern": "images/<val_index>.npy",
                    "layout": "NCHW",
                    "dtype": "float32",
                    "note": "model-ready normalized tensors (not uint8 RGB)",
                },
                "ground_truth_storage": {
                    "path_pattern": "ground_truth/<val_index>.npy",
                    "dtype": "int64_or_int",
                    "ignore_index": 255,
                },
            },
            "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
            "formal_codec_path": formal_codec_path,
            "roundtrip_alignment": {
                "passed": aligned,
                "max_abs_diff": alignment_max,
                "atol": _ALIGN_ATOL,
                "samples": alignment_samples,
            },
            "golden_indices": golden_indices,
            "note": (
                "Held-out board evaluation package stays on x86 shared disk; "
                "do not copy wholesale to Raspberry Pi."
            ),
        }
        with (eval_root / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(eval_meta, f, indent=2, ensure_ascii=False)
        with (eval_root / "golden_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(golden_manifest, f, indent=2, ensure_ascii=False)
        print(f"evaluation complete: {eval_root}", flush=True)

    summary = {
        "status": "PASS",
        "formal_codec_path": formal_codec_path,
        "roundtrip_alignment": {"passed": aligned, "max_abs_diff": alignment_max},
        "calib_modules": modules,
        "calib_count": len(calib_indices),
        "eval_count": len(eval_indices),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
