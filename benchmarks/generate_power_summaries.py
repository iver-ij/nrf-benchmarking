#!/usr/bin/env python3
"""Generate tracked PPK2 summaries from oversized raw CSV exports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


TRACE_GROUPS = {
    "matched": {
        "title": "PPK2 Matched Runtime Traces",
        "subtitle": "AI-only vs AI + WiFi, 50 ms bucketed mean current",
        "output": "ppk2-matched-runtime-traces.svg",
        "traces": [
            ("ai-only.csv", "AI only", "#2563eb"),
            ("ai+wifi.csv", "AI + WiFi", "#dc2626"),
        ],
    },
    "isolated": {
        "title": "PPK2 Isolated Stage Traces",
        "subtitle": "DSP-only vs NN-only, 50 ms bucketed mean current",
        "output": "ppk2-isolated-stage-traces.svg",
        "traces": [
            ("dsp-only.csv", "DSP only", "#2563eb"),
            ("nn-only.csv", "NN only", "#16a34a"),
        ],
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pick_column(fieldnames: Iterable[str], candidates: Iterable[str]) -> str:
    names = list(fieldnames)
    lowered = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate in names:
            return candidate
        match = lowered.get(candidate.lower())
        if match is not None:
            return match
    raise ValueError(f"Missing expected column. Found: {', '.join(names)}")


def read_trace(path: Path, bucket_ms: float) -> dict:
    file_hash = sha256_file(path)
    bucket_sums: list[float] = []
    bucket_counts: list[int] = []

    count = 0
    total_current_ma = 0.0
    min_current_ma = math.inf
    max_current_ma = -math.inf
    first_ts_ms: float | None = None
    last_ts_ms: float | None = None

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        time_col = pick_column(reader.fieldnames, ["Timestamp(ms)", "time_ms", "time"])
        current_col = pick_column(reader.fieldnames, ["Current(uA)", "current_ua", "current"])

        for row in reader:
            timestamp_ms = float(row[time_col])
            current_ma = float(row[current_col]) / 1000.0
            if first_ts_ms is None:
                first_ts_ms = timestamp_ms
            last_ts_ms = timestamp_ms
            count += 1
            total_current_ma += current_ma
            min_current_ma = min(min_current_ma, current_ma)
            max_current_ma = max(max_current_ma, current_ma)

            bucket = int((timestamp_ms - first_ts_ms) // bucket_ms)
            while len(bucket_sums) <= bucket:
                bucket_sums.append(0.0)
                bucket_counts.append(0)
            bucket_sums[bucket] += current_ma
            bucket_counts[bucket] += 1

    if count == 0 or first_ts_ms is None or last_ts_ms is None:
        raise ValueError(f"{path} contains no samples")

    bucket_points = [
        {
            "time_s": round((index * bucket_ms) / 1000.0, 6),
            "current_ma": bucket_sums[index] / bucket_counts[index],
        }
        for index in range(len(bucket_sums))
        if bucket_counts[index] > 0
    ]

    return {
        "file": path.name,
        "logical_path": f"benchmarks/traces/{path.name}",
        "size_bytes": path.stat().st_size,
        "sha256": file_hash,
        "sample_count": count,
        "duration_s": (last_ts_ms - first_ts_ms) / 1000.0,
        "sample_period_ms": (last_ts_ms - first_ts_ms) / (count - 1),
        "mean_current_ma": total_current_ma / count,
        "min_current_ma": min_current_ma,
        "max_current_ma": max_current_ma,
        "bucket_ms": bucket_ms,
        "bucket_count": len(bucket_points),
        "points": bucket_points,
    }


def nice_ticks(low: float, high: float, count: int) -> list[float]:
    if high <= low:
        return [low]
    step = (high - low) / max(count - 1, 1)
    return [low + i * step for i in range(count)]


def render_svg(group: dict, traces: list[dict], output: Path) -> None:
    width = 980
    height = 440
    left = 76
    right = 32
    top = 98
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom

    y_values = [point["current_ma"] for trace in traces for point in trace["points"]]
    x_max = max(trace["duration_s"] for trace in traces)
    y_min = math.floor((min(y_values) - 0.5) * 2) / 2
    y_max = math.ceil((max(y_values) + 0.5) * 2) / 2
    if y_max - y_min < 1.0:
        y_max = y_min + 1.0

    def sx(time_s: float) -> float:
        return left + (time_s / x_max) * plot_w

    def sy(current_ma: float) -> float:
        return top + (y_max - current_ma) / (y_max - y_min) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'  <title id="title">{html.escape(group["title"])}</title>',
        f'  <desc id="desc">{html.escape(group["subtitle"])}</desc>',
        f'  <rect width="{width}" height="{height}" fill="#f8fafc"/>',
        f'  <text x="40" y="48" font-family="Arial,sans-serif" font-size="25" '
        f'font-weight="700" fill="#0f172a">{html.escape(group["title"])}</text>',
        f'  <text x="40" y="76" font-family="Arial,sans-serif" font-size="14" '
        f'fill="#475569">{html.escape(group["subtitle"])}</text>',
        f'  <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" '
        f'stroke="#cbd5e1"/>',
    ]

    for tick in nice_ticks(y_min, y_max, 6):
        y = sy(tick)
        lines.append(f'  <line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        lines.append(
            f'  <text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="Arial,sans-serif" font-size="12" fill="#475569">{tick:.1f}</text>'
        )

    for tick in nice_ticks(0.0, x_max, 5):
        x = sx(tick)
        lines.append(f'  <line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#e2e8f0"/>')
        lines.append(
            f'  <text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" '
            f'font-family="Arial,sans-serif" font-size="12" fill="#475569">{tick:.0f}</text>'
        )

    for trace in traces:
        coords = [(sx(point["time_s"]), sy(point["current_ma"])) for point in trace["points"]]
        for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
            lines.append(
                f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{trace["color"]}" stroke-width="1.8" stroke-linecap="round"/>'
            )

    lines.extend(
        [
            f'  <text x="{left + plot_w / 2:.1f}" y="{height - 20}" text-anchor="middle" '
            f'font-family="Arial,sans-serif" font-size="13" fill="#334155">Time (s)</text>',
            f'  <text x="24" y="{top + plot_h / 2:.1f}" text-anchor="middle" '
            f'transform="rotate(-90 24 {top + plot_h / 2:.1f})" '
            f'font-family="Arial,sans-serif" font-size="13" fill="#334155">Current (mA)</text>',
        ]
    )

    legend_x = left + plot_w - 260
    legend_y = 38
    for index, trace in enumerate(traces):
        y = legend_y + index * 24
        lines.append(f'  <line x1="{legend_x}" y1="{y}" x2="{legend_x + 34}" y2="{y}" stroke="{trace["color"]}" stroke-width="3"/>')
        lines.append(
            f'  <text x="{legend_x + 44}" y="{y + 4}" font-family="Arial,sans-serif" '
            f'font-size="13" fill="#334155">{html.escape(trace["label"])} '
            f'({trace["mean_current_ma"]:.3f} mA mean)</text>'
        )

    lines.append(
        f'  <text x="40" y="92" font-family="Arial,sans-serif" font-size="12" '
        f'fill="#64748b">Raw CSV checksums: benchmarks/results/power_trace_manifest.json</text>'
    )
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_png(svg_path: Path) -> Path | None:
    png_path = svg_path.with_suffix(".png")
    rsvg = shutil.which("rsvg-convert")
    if rsvg is not None:
        subprocess.run([rsvg, str(svg_path), "-o", str(png_path)], check=True)
        return png_path

    magick = shutil.which("magick")
    if magick is None:
        return None

    subprocess.run([magick, str(svg_path), str(png_path)], check=True)
    return png_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True, type=Path, help="Directory containing raw PPK2 CSV exports")
    parser.add_argument("--bucket-ms", type=float, default=50.0, help="Bucket size used for SVG trace means")
    parser.add_argument("--assets-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/results/power_trace_manifest.json"))
    args = parser.parse_args()

    args.assets_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    trace_stats: dict[str, dict] = {}
    derived_outputs = []
    for group in TRACE_GROUPS.values():
        group_traces = []
        for filename, label, color in group["traces"]:
            if filename not in trace_stats:
                path = args.trace_dir / filename
                print(f"Reading {path}")
                trace_stats[filename] = read_trace(path, args.bucket_ms)
            group_trace = dict(trace_stats[filename])
            group_trace["label"] = label
            group_trace["color"] = color
            group_traces.append(group_trace)

        output = args.assets_dir / group["output"]
        render_svg(group, group_traces, output)
        derived_outputs.append(str(output))
        print(f"Wrote {output}")
        png_output = render_png(output)
        if png_output is not None:
            derived_outputs.append(str(png_output))
            print(f"Wrote {png_output}")
        else:
            print("ImageMagick not found; skipped PNG rendering")

    manifest_traces = {
        filename: {key: value for key, value in stats.items() if key != "points"}
        for filename, stats in trace_stats.items()
    }
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "raw_trace_policy": "Full PPK2 CSV exports are kept outside Git because each capture is about 166 MB.",
        "logical_trace_directory": "benchmarks/traces/",
        "source_trace_names": sorted(manifest_traces),
        "bucket_ms": args.bucket_ms,
        "derived_outputs": derived_outputs,
        "traces": manifest_traces,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
