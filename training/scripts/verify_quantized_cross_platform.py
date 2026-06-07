#!/usr/bin/env python3
"""
Cross-platform quantized inference parity verifier.

Compares:
- Hardware int8 output tensor captured from firmware (QREF)
- Local Python TFLite int8 output tensor, using the exact same int8 input tensor
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


@dataclass
class QuantParityMetrics:
    exact_match: bool
    max_abs_diff: int
    mean_abs_diff: float
    mismatched_indices: int
    total_outputs: int
    hw_argmax: int
    py_argmax: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify cross-platform int8 output parity")
    p.add_argument(
        "--qref-json",
        default="training/outputs/qref_tensors.json",
        help="Parsed QREF tensor JSON from hardware",
    )
    p.add_argument(
        "--model",
        default="training/outputs/model.tflite",
        help="INT8 TFLite model path",
    )
    p.add_argument(
        "--summary-out",
        default="training/outputs/quantized_cross_platform_summary.json",
        help="Output summary JSON",
    )
    p.add_argument(
        "--acceptance-mode",
        choices=["exact", "behavior", "threshold"],
        default="behavior",
        help=(
            "Acceptance policy: exact=byte-identical output; "
            "behavior=same argmax + same thresholded decision; "
            "threshold=same thresholded decision only."
        ),
    )
    p.add_argument(
        "--target-class-index",
        type=int,
        default=2,
        help="Target class index for thresholded behavior checks",
    )
    p.add_argument(
        "--decision-threshold",
        type=float,
        default=0.75,
        help="Detection threshold used for behavior/threshold acceptance modes",
    )
    p.add_argument(
        "--max-abs-diff-int8",
        type=int,
        default=1,
        help="Maximum allowed per-logit int8 absolute difference for non-exact acceptance",
    )
    return p.parse_args()


def load_qref(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    in_arr = np.array(payload["input_int8"], dtype=np.int8)
    out_arr = np.array(payload["output_int8"], dtype=np.int8)
    fw_prob = float(payload.get("firmware_probability", 0.0))
    return in_arr, out_arr, fw_prob


def run_python_int8(model_path: Path, input_int8: np.ndarray) -> tuple[np.ndarray, dict, dict]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_shape = tuple(int(v) for v in input_details["shape"])
    expected_elems = int(np.prod(input_shape))
    if input_int8.size != expected_elems:
        raise ValueError(
            f"Input tensor size mismatch: qref={input_int8.size}, model expects {expected_elems}"
        )

    input_tensor = input_int8.reshape(input_shape).astype(np.int8)
    interpreter.set_tensor(input_details["index"], input_tensor)
    interpreter.invoke()

    output_tensor = interpreter.get_tensor(output_details["index"]).astype(np.int8).reshape(-1)
    return output_tensor, input_details, output_details


def compare(hw_out: np.ndarray, py_out: np.ndarray) -> QuantParityMetrics:
    if hw_out.shape != py_out.shape:
        raise ValueError(f"Output shape mismatch: hw={hw_out.shape}, py={py_out.shape}")

    diff = np.abs(hw_out.astype(np.int16) - py_out.astype(np.int16))
    mismatched = int(np.count_nonzero(diff))
    return QuantParityMetrics(
        exact_match=bool(mismatched == 0),
        max_abs_diff=int(np.max(diff)) if diff.size else 0,
        mean_abs_diff=float(np.mean(diff)) if diff.size else 0.0,
        mismatched_indices=mismatched,
        total_outputs=int(diff.size),
        hw_argmax=int(np.argmax(hw_out)) if hw_out.size else -1,
        py_argmax=int(np.argmax(py_out)) if py_out.size else -1,
    )


def dequantize_int8(values: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (values.astype(np.float32) - float(zero_point)) * float(scale)


def evaluate_acceptance(
    *,
    mode: str,
    metrics: QuantParityMetrics,
    same_threshold_decision: bool,
    max_abs_diff_limit: int | None,
) -> tuple[bool, dict[str, Any]]:
    drift_within_limit = True
    if max_abs_diff_limit is not None:
        drift_within_limit = metrics.max_abs_diff <= max_abs_diff_limit

    if mode == "exact":
        passed = metrics.exact_match
        rule = {
            "mode": mode,
            "require_exact_int8_output_match": True,
        }
    elif mode == "behavior":
        passed = (
            (metrics.hw_argmax == metrics.py_argmax)
            and same_threshold_decision
            and drift_within_limit
        )
        rule = {
            "mode": mode,
            "require_same_argmax": True,
            "require_same_threshold_decision": True,
            "max_abs_diff_int8": max_abs_diff_limit,
        }
    elif mode == "threshold":
        passed = same_threshold_decision and drift_within_limit
        rule = {
            "mode": mode,
            "require_same_threshold_decision": True,
            "max_abs_diff_int8": max_abs_diff_limit,
        }
    else:
        raise ValueError(f"Unsupported acceptance mode: {mode}")

    return passed, rule


def main() -> int:
    args = parse_args()
    qref_path = Path(args.qref_json)
    model_path = Path(args.model)
    summary_path = Path(args.summary_out)

    if not qref_path.exists():
        raise SystemExit(f"QREF JSON not found: {qref_path}")
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    input_int8, hw_output_int8, fw_probability = load_qref(qref_path)
    py_output_int8, in_details, out_details = run_python_int8(model_path, input_int8)
    metrics = compare(hw_output_int8, py_output_int8)

    scale = float(out_details["quantization"][0])
    zero = int(out_details["quantization"][1])
    hw_output_deq = dequantize_int8(hw_output_int8, scale, zero)
    py_output_deq = dequantize_int8(py_output_int8, scale, zero)

    if args.target_class_index < 0 or args.target_class_index >= metrics.total_outputs:
        raise SystemExit(
            f"Invalid --target-class-index={args.target_class_index}. "
            f"Valid range is [0, {metrics.total_outputs - 1}]."
        )

    fw_target_prob = float(hw_output_deq[args.target_class_index])
    py_target_prob = float(py_output_deq[args.target_class_index])
    fw_detected = fw_target_prob >= args.decision_threshold
    py_detected = py_target_prob >= args.decision_threshold
    same_threshold_decision = fw_detected == py_detected
    behavior_equivalent = (metrics.hw_argmax == metrics.py_argmax) and same_threshold_decision

    max_abs_diff_limit = None if args.max_abs_diff_int8 < 0 else int(args.max_abs_diff_int8)
    acceptance_pass, acceptance_rule = evaluate_acceptance(
        mode=args.acceptance_mode,
        metrics=metrics,
        same_threshold_decision=same_threshold_decision,
        max_abs_diff_limit=max_abs_diff_limit,
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Cross-platform INT8 inference parity verification",
        "outputs": {
            "qref_json": str(qref_path),
            "model": str(model_path),
        },
        "model_io": {
            "input_shape": [int(v) for v in in_details["shape"]],
            "input_dtype": str(in_details["dtype"]),
            "output_shape": [int(v) for v in out_details["shape"]],
            "output_dtype": str(out_details["dtype"]),
            "output_scale": scale,
            "output_zero_point": zero,
        },
        "results": {
            "exact_int8_output_match": metrics.exact_match,
            "behavior_equivalent": behavior_equivalent,
            "behavior_equivalence_rule": {
                "require_same_argmax": True,
                "require_same_threshold_decision": True,
                "target_class_index": args.target_class_index,
                "decision_threshold": args.decision_threshold,
            },
            "threshold_decision_equivalent": same_threshold_decision,
            "mismatched_output_indices": metrics.mismatched_indices,
            "total_output_indices": metrics.total_outputs,
            "max_abs_diff_int8": metrics.max_abs_diff,
            "mean_abs_diff_int8": metrics.mean_abs_diff,
            "firmware_output_argmax": metrics.hw_argmax,
            "python_output_argmax": metrics.py_argmax,
            "firmware_probability_on": fw_probability,
            "firmware_target_probability": fw_target_prob,
            "python_target_probability": py_target_prob,
            "firmware_target_detected": fw_detected,
            "python_target_detected": py_detected,
            "target_class_index": args.target_class_index,
            "decision_threshold": args.decision_threshold,
            "acceptance_mode": args.acceptance_mode,
            "acceptance_rule": acceptance_rule,
            "acceptance_pass": acceptance_pass,
            "firmware_output_dequantized": hw_output_deq.tolist(),
            "python_output_dequantized": py_output_deq.tolist(),
        },
        "notes": [
            "This test uses identical int8 input tensor values from hardware for both runtimes.",
            "Behavior acceptance allows limited quantized drift while preserving decision equivalence.",
        ],
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("Quantized Cross-Platform Parity")
    print(f"  exact_int8_output_match: {metrics.exact_match}")
    print(f"  behavior_equivalent: {behavior_equivalent}")
    print(f"  threshold_decision_equivalent: {same_threshold_decision}")
    print(f"  mismatches: {metrics.mismatched_indices}/{metrics.total_outputs}")
    print(f"  max_abs_diff_int8: {metrics.max_abs_diff}")
    print(f"  acceptance_mode: {args.acceptance_mode}")
    print(f"  acceptance_pass: {acceptance_pass}")
    print(f"  argmax (fw, py): ({metrics.hw_argmax}, {metrics.py_argmax})")
    print(f"Summary: {summary_path}")
    return 0 if acceptance_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
