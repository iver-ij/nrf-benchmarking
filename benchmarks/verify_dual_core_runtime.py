#!/usr/bin/env python3
"""Verify dual-core runtime records from UART logs and optional capture."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

INF_RE = re.compile(r"Inference:\s*([0-9]+(?:\.[0-9]+)?)\s*ms")

REQUIRED_MARKERS = [
    "DSP mode: dual-core offload",
    "Dual-core DSP IPC ready",
]
OPTIONAL_MARKERS = [
    "CPUNET DSP service ready",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate dual-core runtime UART records")
    p.add_argument("--uart-log", required=True, help="UART log file path")
    p.add_argument(
        "--summary-out",
        default="benchmarks/results/dual_core_runtime_summary.json",
        help="Output summary path",
    )
    p.add_argument("--min-inference-samples", type=int, default=5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log_path = Path(args.uart_log)
    if not log_path.exists():
        raise SystemExit(f"Missing UART log: {log_path}")

    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    text = "\n".join(lines)

    required_hits = {m: (m in text) for m in REQUIRED_MARKERS}
    optional_hits = {m: (m in text) for m in OPTIONAL_MARKERS}

    times = []
    for line in lines:
        m = INF_RE.search(line)
        if m:
            times.append(float(m.group(1)))

    required_pass = all(required_hits.values())
    inference_pass = len(times) >= args.min_inference_samples

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "uart_log": str(log_path),
        "checks": {
            "required_markers": required_hits,
            "optional_markers": optional_hits,
            "required_markers_pass": required_pass,
            "inference_sample_count": len(times),
            "inference_samples_pass": inference_pass,
            "inference_mean_ms": float(statistics.fmean(times)) if times else None,
            "overall_pass": required_pass and inference_pass,
        },
    }

    out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"required_markers_pass={required_pass}")
    print(f"inference_sample_count={len(times)}")
    print(f"overall_pass={required_pass and inference_pass}")
    print(f"summary={out}")
    return 0 if (required_pass and inference_pass) else 1


if __name__ == "__main__":
    raise SystemExit(main())
