#!/usr/bin/env python3
"""Create a latency-only 19-logit variant of the frozen 150-class head.

The graph keeps the trained pre-head, classifier geometry, and x8 bilinear
upsample unchanged.  It only selects the fixed-19 rows of the final 1x1
classifier.  This is *not* a 19-class semantic model: its purpose is to
measure Hailo output-channel/runtime sensitivity before authorising training.

The tool writes an ONNX file plus a JSON provenance/parity report.  When
``--check-input`` is supplied (NCHW float32 ``s8`` tensors), it verifies that
every output element equals the corresponding selected channel from the source
150-logit ONNX graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper


FIXED19_ZERO_BASED = [
    7, 10, 15, 20, 23, 30, 35, 37, 48, 51,
    80, 83, 84, 90, 101, 102, 103, 109, 122,
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_initializer(graph: onnx.GraphProto, name: str, value: np.ndarray) -> None:
    matches = [item for item in graph.initializer if item.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one initializer {name!r}, found {len(matches)}")
    graph.initializer.remove(matches[0])
    graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(value), name=name))


def update_channel_dimensions(model: onnx.ModelProto, output_name: str, channels: int) -> None:
    """Update both public output and classifier intermediate shape metadata.

    The source export contains a value_info entry for the 150-channel Conv.
    Leaving it stale is legal enough for ONNX checker but makes ORT warn and can
    confuse a downstream parser.  Setting ``dim_value`` is sufficient: shape
    dimensions are a protobuf oneof, so setting ``dim_param`` afterwards would
    incorrectly clear the numeric value.
    """
    targets = {output_name, "/seg_head/cls/Conv_output_0"}
    found: set[str] = set()
    for value in [*model.graph.output, *model.graph.value_info]:
        if value.name not in targets:
            continue
        shape = value.type.tensor_type.shape.dim
        if len(shape) != 4:
            raise ValueError(f"expected 4D tensor {value.name!r}, found {len(shape)} dimensions")
        shape[1].dim_value = channels
        found.add(value.name)
    if output_name not in found:
        raise ValueError(f"missing graph output {output_name!r}")


def set_metadata(model: onnx.ModelProto, values: dict[str, str]) -> None:
    retained = [item for item in model.metadata_props if item.key not in values]
    del model.metadata_props[:]
    model.metadata_props.extend(retained)
    for key, value in values.items():
        item = model.metadata_props.add()
        item.key = key
        item.value = value


def run_ort(path: Path, s8_nchw: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    if s8_nchw.dtype != np.float32 or s8_nchw.shape != (1, 128, 64, 64):
        raise ValueError(f"expected float32 [1,128,64,64], got {s8_nchw.dtype}/{s8_nchw.shape}")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {session.get_inputs()[0].name: s8_nchw})
    if len(outputs) != 1:
        raise ValueError(f"expected one ORT output, got {len(outputs)}")
    return np.asarray(outputs[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--check-input", type=Path, action="append", default=[])
    args = parser.parse_args()

    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if args.out.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite --out or --report")
    if args.out.parent != args.report.parent:
        raise ValueError("--out and --report must share one isolated artifact directory")
    args.out.parent.mkdir(parents=True, exist_ok=False)

    model = onnx.load(str(args.source), load_external_data=False)
    source_weight = next((item for item in model.graph.initializer if item.name == "seg_head.cls.weight"), None)
    source_bias = next((item for item in model.graph.initializer if item.name == "seg_head.cls.bias"), None)
    if source_weight is None or source_bias is None:
        raise ValueError("expected seg_head.cls.weight and seg_head.cls.bias initializers")
    weight = numpy_helper.to_array(source_weight)
    bias = numpy_helper.to_array(source_bias)
    if weight.shape != (150, 128, 1, 1) or bias.shape != (150,):
        raise ValueError(f"unexpected classifier shapes: weight={weight.shape}, bias={bias.shape}")

    indices = np.asarray(FIXED19_ZERO_BASED, dtype=np.int64)
    selected_weight = weight[indices]
    selected_bias = bias[indices]
    replace_initializer(model.graph, "seg_head.cls.weight", selected_weight)
    replace_initializer(model.graph, "seg_head.cls.bias", selected_bias)
    update_channel_dimensions(model, "seg_logits", len(indices))
    set_metadata(model, {
        "semantic_sensing.latency_probe": "true",
        "semantic_sensing.source_onnx_sha256": sha256(args.source),
        "semantic_sensing.selected_class_indices_zero_based": json.dumps(FIXED19_ZERO_BASED),
        "semantic_sensing.accuracy_contract": "latency_only_not_a_trained_19class_model",
    })
    onnx.checker.check_model(model)
    onnx.save(model, str(args.out))

    parity: list[dict[str, Any]] = []
    for input_path in args.check_input:
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        s8 = np.load(input_path, allow_pickle=False)
        source_logits = run_ort(args.source, s8)
        probe_logits = run_ort(args.out, s8)
        expected = source_logits[:, indices, :, :]
        if probe_logits.shape != expected.shape:
            raise ValueError(f"output shape mismatch {probe_logits.shape} != {expected.shape}")
        delta = np.abs(probe_logits - expected)
        parity.append({
            "input": str(input_path),
            "shape": list(probe_logits.shape),
            "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()),
            "exact_equal": bool(np.array_equal(probe_logits, expected)),
        })

    status = "PASS" if all(item["max_abs"] == 0.0 for item in parity) else "PASS_NO_PARITY_INPUT"
    if parity and status != "PASS":
        raise AssertionError(f"ONNX selected-channel parity failed: {parity}")
    report = {
        "schema": "phase2.head19_latency_probe_onnx.v1",
        "status": status,
        "accuracy_scope": "latency_only_not_a_trained_19class_model",
        "source": {"path": str(args.source), "sha256": sha256(args.source)},
        "probe": {"path": str(args.out), "sha256": sha256(args.out)},
        "fixed19_zero_based": FIXED19_ZERO_BASED,
        "classifier": {
            "source_weight_shape": list(weight.shape),
            "probe_weight_shape": list(selected_weight.shape),
            "source_bias_shape": list(bias.shape),
            "probe_bias_shape": list(selected_bias.shape),
            "selected_weight_sha256": hashlib.sha256(selected_weight.tobytes()).hexdigest(),
            "selected_bias_sha256": hashlib.sha256(selected_bias.tobytes()).hexdigest(),
        },
        "onnx_runtime_selected_channel_parity": parity,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out": str(args.out), "report": str(args.report)}, sort_keys=True))


if __name__ == "__main__":
    main()
