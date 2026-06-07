#!/usr/bin/env python3
"""
Validate latency claim quality from baseline/current capture JSON files.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate latency A/B claim quality and thresholds")
    p.add_argument("--baseline", required=True, help="Baseline latency JSON")
    p.add_argument("--current", required=True, help="Current latency JSON")
    p.add_argument("--summary-out", default="benchmarks/results/latency_claim_validation.json")
    p.add_argument("--min-baseline-samples", type=int, default=30)
    p.add_argument("--min-current-samples", type=int, default=30)
    p.add_argument("--min-speedup-x", type=float, default=300.0)
    p.add_argument("--min-reduction-percent", type=float, default=99.67)
    return p.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    baseline = load_json(Path(args.baseline))
    current = load_json(Path(args.current))

    b_count = int(baseline.get("sample_count", 0))
    c_count = int(current.get("sample_count", 0))
    b_mean = float(baseline.get("mean_ms"))
    c_mean = float(current.get("mean_ms"))

    speedup = b_mean / c_mean if c_mean > 0 else 0.0
    reduction = (1.0 - (c_mean / b_mean)) * 100.0 if b_mean > 0 else 0.0

    sample_quality_pass = b_count >= args.min_baseline_samples and c_count >= args.min_current_samples
    claim_threshold_pass = speedup >= args.min_speedup_x and reduction >= args.min_reduction_percent

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "baseline": args.baseline,
            "current": args.current,
        },
        "thresholds": {
            "min_baseline_samples": args.min_baseline_samples,
            "min_current_samples": args.min_current_samples,
            "min_speedup_x": args.min_speedup_x,
            "min_reduction_percent": args.min_reduction_percent,
        },
        "metrics": {
            "baseline_samples": b_count,
            "current_samples": c_count,
            "baseline_mean_ms": b_mean,
            "current_mean_ms": c_mean,
            "speedup_x": speedup,
            "reduction_percent": reduction,
        },
        "quality": {
            "sample_quality_pass": sample_quality_pass,
            "claim_threshold_pass": claim_threshold_pass,
            "overall_pass": sample_quality_pass and claim_threshold_pass,
        },
    }

    out_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"speedup_x={speedup:.2f}")
    print(f"reduction_percent={reduction:.2f}")
    print(f"sample_quality_pass={sample_quality_pass}")
    print(f"claim_threshold_pass={claim_threshold_pass}")
    print(f"overall_pass={sample_quality_pass and claim_threshold_pass}")
    print(f"summary={out_path}")

    return 0 if (sample_quality_pass and claim_threshold_pass) else 1


if __name__ == "__main__":
    raise SystemExit(main())
