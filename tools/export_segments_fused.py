#!/usr/bin/env python3
"""Export the deployable three-segment ConvBN fused-bottleneck graphs.

The semantic codec has two CPU rANS boundaries, therefore it cannot become one
continuous Hailo graph.  This exporter creates the three *hardware* graphs
around those boundaries:

  analysis       image -> latent_y, latent_z        (sender)
  hyper          latent_z_hat -> means, scales      (sender and receiver)
  synthesis_tail latent_y_hat -> seg_logits         (receiver)

It deliberately exports only static ONNX plus an immutable interface manifest.
It never invokes Hailo DFC or quantization.  The matching calibration and DFC
orchestrators consume ``segments_manifest.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from export_modules_fused import (
    GAAnalysis,
    HAAnalysis,
    HSSynthesis,
    GSSynthesis,
    ReceiverDappm,
    ReceiverHead,
    ReceiverStage,
    SenderBackbone,
    ModuleSpec,
    _DEFAULT_CKPT,
    _sha256_file,
    export_one,
    load_student,
)



class AnalysisSegment(nn.Module):
    """Sender backbone + align_h + g_a + h_a, preserving the old wrappers."""

    def __init__(self, backbone: nn.Module, bottleneck: nn.Module) -> None:
        super().__init__()
        self.sender = SenderBackbone(backbone)
        self.align_h = bottleneck.align_h
        self.g_a = GAAnalysis(bottleneck.g_a)
        self.h_a = HAAnalysis(bottleneck.h_a)

    def forward(self, image: torch.Tensor):
        x_h, x_l = self.sender(image)
        latent_y = self.g_a(self.align_h(x_h), x_l)
        latent_z = self.h_a(latent_y)
        return latent_y, latent_z


class HyperSegment(nn.Module):
    """H_s is the one reusable graph required on both codec endpoints."""

    def __init__(self, bottleneck: nn.Module) -> None:
        super().__init__()
        self.h_s = HSSynthesis(bottleneck.h_s)

    def forward(self, latent_z_hat: torch.Tensor):
        return self.h_s(latent_z_hat)


class SynthesisTailSegment(nn.Module):
    """g_s reconstruction followed by the complete receiver tail and head."""

    def __init__(self, student: nn.Module, h8: int, w8: int, h16: int, w16: int) -> None:
        super().__init__()
        backbone = student.teacher.backbone
        self.g_s = GSSynthesis(student.bottleneck, (h8, w8), (h16, w16))
        self.stage4 = ReceiverStage(backbone.stage4)
        self.stage5 = ReceiverStage(backbone.stage5)
        self.dappm = ReceiverDappm(backbone)
        self.head = ReceiverHead(student.teacher)

    def forward(self, latent_y_hat: torch.Tensor):
        x_h, x_l = self.g_s(latent_y_hat)
        stage4_high, stage4_low = self.stage4(x_h, x_l)
        high_feat, low_feat = self.stage5(stage4_high, stage4_low)
        s8 = self.dappm(high_feat, low_feat)
        return self.head(s8)


def _t(*shape: int) -> torch.Tensor:
    return torch.randn(*shape)


def build_specs(student: nn.Module, height: int, width: int, cfg) -> list[ModuleSpec]:
    """Build static graph contracts at the requested image shape."""
    bb = student.teacher.backbone
    bargs = cfg.bottleneck.args
    if student.bottleneck.attn_a is not None:
        raise ValueError("analysis attention is not part of the frozen three-segment ConvBN contract")
    h8, w8 = height // 8, width // 8
    h32, w32 = height // 32, width // 32
    latent_channels = int(bargs.get("latent_channels", 192))
    hyper_channels = int(bargs.get("hyper_channels", 128))
    y_h, y_w = h32, w32
    z_h, z_w = y_h // 2, y_w // 2

    return [
        ModuleSpec(
            "analysis",
            "sender backbone + align_h + g_a + h_a; image -> latent_y, latent_z",
            AnalysisSegment(bb, student.bottleneck).eval(),
            (_t(1, 3, height, width),),
            ["image"],
            ["latent_y", "latent_z"],
            "high",
            "Merged sender analysis graph. Includes ConvBN align/g_a and h_a; no entropy op.",
        ),
        ModuleSpec(
            "hyper",
            "hyper synthesis; latent_z_hat -> means, scales; deployed at both rANS endpoints",
            HyperSegment(student.bottleneck).eval(),
            (_t(1, hyper_channels, z_h, z_w),),
            ["latent_z_hat"],
            ["means", "scales"],
            "low",
            "H_s only; Gaussian index construction and rANS remain on host CPU.",
        ),
        ModuleSpec(
            "synthesis_tail",
            "g_s reconstruction + receiver stage4/stage5/DAPPM/head; latent_y_hat -> logits",
            SynthesisTailSegment(student, h8, w8, height // 16, width // 16).eval(),
            (_t(1, latent_channels, y_h, y_w),),
            ["latent_y_hat"],
            ["seg_logits"],
            "high",
            "Largest receiver graph: bounded ConvBN synthesis plus RTFormer attention tail.",
        ),
    ]


def _entry(result, role: str, components: list[str], deployment: list[str]) -> dict:
    return {
        "id": result.name,
        "role": role,
        "deployment": deployment,
        "components": components,
        "onnx": result.path,
        "sha256": result.sha256,
        "nodes": result.nodes,
        "size_kb": round(result.size_kb, 1),
        "opset": result.opset,
        "input_names": result.input_names,
        "output_names": result.output_names,
        "in_shapes_nchw": result.in_shapes,
        "out_shapes_nchw": result.out_shapes,
        "ops": result.ops,
        "risky_ops": result.risky_ops,
        "hailo_risk": result.hailo_risk,
        "risk_note": result.risk_note,
        "ort_max_abs": result.max_abs,
        "ort_ok": result.ort_ok,
        "simplified": result.simplified,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-ckpt", type=Path, default=Path(_DEFAULT_CKPT))
    parser.add_argument("--shape", type=int, nargs=4, default=[1, 3, 512, 512])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--no-ort", action="store_true")
    parser.add_argument("--gfa-mode", choices=["original", "group_softmax", "memory_softmax"],
                        default="group_softmax")
    args = parser.parse_args(argv)

    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    if list(args.shape[:2]) != [1, 3] or any(value <= 0 for value in args.shape):
        raise ValueError("--shape must be static 1x3xHxW")
    _, _, height, width = args.shape
    if height % 32 or width % 32:
        raise ValueError("H and W must be divisible by 32 for the fixed mid split")
    if not args.student_ckpt.is_file():
        raise FileNotFoundError(args.student_ckpt)

    torch.manual_seed(0)
    student, act, cfg = load_student(str(args.student_ckpt), gfa_mode=args.gfa_mode)
    gfa_mode = cfg.model.backbone.args.get("gfa_mode")
    if gfa_mode != args.gfa_mode:
        raise ValueError(f"GFA override did not stick: expected {args.gfa_mode!r}, got {gfa_mode!r}")
    if act != "convbn":
        raise ValueError(f"segmented deployment requires ConvBN checkpoint, got act={act!r}")

    specs = build_specs(student, height, width, cfg)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for spec in specs:
        result = export_one(spec, str(args.out_dir), args.opset, check_ort=not args.no_ort)
        if result.ort_ok is False:
            raise RuntimeError(f"Torch/ORT parity failed for {spec.name}: {result.max_abs}")
        results.append(result)
        print(f"[PASS] {spec.name:14s} nodes={result.nodes:4d} ORT={result.max_abs}")

    by_name = {result.name: result for result in results}
    manifest = {
        "schema": "semantic_sensing.segmented_onnx.v1",
        "checkpoint": {"path": str(args.student_ckpt.resolve()), "sha256": _sha256_file(str(args.student_ckpt))},
        "model": {
            "split": cfg.student.split,
            "bottleneck": cfg.bottleneck.name,
            "activation": act,
            "gfa_mode": gfa_mode,
        },
        "net_input_nchw": list(args.shape),
        "opset": args.opset,
        "segments": [
            _entry(by_name["analysis"], "sender_analysis",
                   ["sender_backbone", "bn_align_h", "bn_g_a", "bn_h_a"], ["sender"]),
            _entry(by_name["hyper"], "hyper_synthesis",
                   ["bn_h_s"], ["sender", "receiver"]),
            _entry(by_name["synthesis_tail"], "receiver_synthesis_tail",
                   ["bn_g_s", "receiver_stage4", "receiver_stage5", "receiver_dappm", "receiver_head"],
                   ["receiver"]),
        ],
        "cpu_codec_boundaries": [
            {
                "id": "z",
                "producer": "analysis.latent_z",
                "consumer": "hyper.latent_z_hat",
                "operation": "entropy_bottleneck rANS compress/decompress",
                "contract": "NCHW float tensor after receiver decode; not an inter-HEF FP boundary",
            },
            {
                "id": "y",
                "producer": "analysis.latent_y",
                "params": "hyper.means,hyper.scales",
                "consumer": "synthesis_tail.latent_y_hat",
                "operation": "gaussian_conditional rANS compress/decompress",
                "contract": "NCHW float tensor after receiver decode; not an inter-HEF FP boundary",
            },
        ],
        "quantization_contract": {
            "rule": "No direct tensor boundary crosses between independent HEFs.",
            "reason": "Only coded y/z streams traverse endpoints; host codec owns float conversion and rANS.",
            "required_chain_calibration": [
                "analysis quantized output -> rANS(z) -> hyper calibration",
                "analysis+hyper quantized outputs -> rANS(y) -> synthesis_tail calibration",
            ],
        },
    }
    manifest_path = args.out_dir / "segments_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[PASS] manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
