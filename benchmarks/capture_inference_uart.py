#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import select
import statistics
import termios
import time
from pathlib import Path

INF_RE = re.compile(
    r"Inference:\s*([0-9]+(?:\.[0-9]+)?)\s*ms(?:\s*\(\s*(\d+)\s*us\))?"
)
PHASE_RE = re.compile(
    r"PHASE_TIMING\s+dsp_us=(\d+)\s+nn_us=(\d+)\s+total_us=(\d+)"
)


def configure(fd: int, baud: int) -> None:
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0
    attrs[1] = 0
    attrs[2] = termios.CLOCAL | termios.CREAD | termios.CS8
    attrs[3] = 0
    speed = getattr(termios, f"B{baud}")
    attrs[4] = speed
    attrs[5] = speed
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 1
    termios.tcsetattr(fd, termios.TCSANOW, attrs)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return float(vals[0])
    k = (len(vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return float(vals[f])
    d = k - f
    return float(vals[f] * (1 - d) + vals[c] * d)


def probe_for_data(port: str, baud: int, seconds: float) -> tuple[bool, bytes]:
    fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    configure(fd, baud)
    start = time.monotonic()
    got_data = False
    sample = b""
    try:
        while (time.monotonic() - start) < seconds:
            ready, _, _ = select.select([fd], [], [], 0.2)
            if not ready:
                continue
            chunk = os.read(fd, 1024)
            if chunk:
                got_data = True
                if len(sample) < 512:
                    sample += chunk[: 512 - len(sample)]
                break
    finally:
        os.close(fd)
    return got_data, sample


def looks_like_text_console(sample: bytes) -> bool:
    if not sample:
        return False
    text = sample.decode("utf-8", errors="ignore")
    if any(token in text for token in ("Inference:", "PHASE_TIMING", "[WiFi]", "[UDP]", "Tensor arena")):
        return True
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\r\n\t")
    return printable >= max(8, int(len(text) * 0.7))


def sample_score(sample: bytes) -> int:
    if not sample:
        return 0
    text = sample.decode("utf-8", errors="ignore")
    if "Inference:" in text:
        return 4
    if "PHASE_TIMING" in text:
        return 3
    if looks_like_text_console(sample):
        return 2
    return 1


def autodetect_port(baud: int) -> str:
    patterns = (
        "/dev/serial/by-id/*",
        "/dev/ttyACM*",
        "/dev/ttyUSB*",
        "/dev/cu.usbmodem*",
        "/dev/tty.usbmodem*",
    )
    candidates = []
    seen = set()
    for pattern in patterns:
        for candidate in sorted(glob.glob(pattern)):
            resolved = os.path.realpath(candidate)
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(candidate)
    if not candidates:
        raise RuntimeError("No serial console found under /dev/serial/by-id, /dev/ttyACM*, or USB modem paths")

    best_port: str | None = None
    best_score = -1
    for port in candidates:
        try:
            got_data, sample = probe_for_data(port, baud, 4.0)
        except OSError:
            continue
        if not got_data:
            continue
        score = sample_score(sample)
        if score > best_score:
            best_port = port
            best_score = score
        if score >= 3:
            return port

    if best_port:
        return best_port

    # Fall back to first tty/candidate if none produced immediate bytes.
    return candidates[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="Serial port (auto-detect if omitted)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--log-out", required=True)
    ap.add_argument("--json-out", required=True)
    args = ap.parse_args()
    if not args.port:
        args.port = autodetect_port(args.baud)
        print(f"Auto-selected port: {args.port}")

    log_path = Path(args.log_out)
    json_path = Path(args.json_out)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    fd = os.open(args.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    configure(fd, args.baud)

    start = time.monotonic()
    buf = b""
    lines: list[str] = []
    ms_vals: list[float] = []
    us_vals: list[int] = []
    dsp_us_vals: list[int] = []
    nn_us_vals: list[int] = []
    pipeline_us_vals: list[int] = []
    terminated_early = False
    termination_reason = "completed_window"

    try:
        while (time.monotonic() - start) < args.seconds:
            ready, _, _ = select.select([fd], [], [], 0.25)
            if not ready:
                continue
            try:
                chunk = os.read(fd, 4096)
            except OSError as e:
                terminated_early = True
                termination_reason = f"uart_read_error: {e}"
                lines.append(f"[UART_READ_ERROR] {e}")
                break

            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode("utf-8", errors="ignore").rstrip("\r")
                lines.append(line)
                m = INF_RE.search(line)
                if m:
                    ms_vals.append(float(m.group(1)))
                    if m.group(2):
                        us_vals.append(int(m.group(2)))
                p = PHASE_RE.search(line)
                if p:
                    dsp_us_vals.append(int(p.group(1)))
                    nn_us_vals.append(int(p.group(2)))
                    pipeline_us_vals.append(int(p.group(3)))
    except KeyboardInterrupt:
        terminated_early = True
        termination_reason = "keyboard_interrupt"
        lines.append("[INTERRUPTED] keyboard_interrupt")
    finally:
        os.close(fd)

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "port": args.port,
        "duration_s": args.seconds,
        "sample_count": len(ms_vals),
        "values_ms": ms_vals,
        "values_us": us_vals,
        "mean_ms": float(statistics.fmean(ms_vals)) if ms_vals else None,
        "median_ms": float(statistics.median(ms_vals)) if ms_vals else None,
        "mean_us": (
            float(statistics.fmean(us_vals))
            if us_vals and len(us_vals) == len(ms_vals)
            else None
        ),
        "min_ms": float(min(ms_vals)) if ms_vals else None,
        "max_ms": float(max(ms_vals)) if ms_vals else None,
        "p95_ms": percentile(ms_vals, 0.95),
        "explicit_us_count": len(us_vals),
        "phase_timing_count": len(pipeline_us_vals),
        "dsp_values_us": dsp_us_vals,
        "nn_values_us": nn_us_vals,
        "pipeline_values_us": pipeline_us_vals,
        "dsp_mean_us": float(statistics.fmean(dsp_us_vals)) if dsp_us_vals else None,
        "nn_mean_us": float(statistics.fmean(nn_us_vals)) if nn_us_vals else None,
        "pipeline_mean_us": (
            float(statistics.fmean(pipeline_us_vals)) if pipeline_us_vals else None
        ),
        "dsp_mean_ms": (
            float(statistics.fmean(dsp_us_vals)) / 1000.0 if dsp_us_vals else None
        ),
        "nn_mean_ms": (
            float(statistics.fmean(nn_us_vals)) / 1000.0 if nn_us_vals else None
        ),
        "pipeline_mean_ms": (
            float(statistics.fmean(pipeline_us_vals)) / 1000.0
            if pipeline_us_vals
            else None
        ),
        "terminated_early": terminated_early,
        "termination_reason": termination_reason,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Captured {len(ms_vals)} inference samples")
    print(f"Log: {log_path}")
    print(f"JSON: {json_path}")
    print(f"Terminated early: {terminated_early} ({termination_reason})")
    if ms_vals:
        summary = (
            f"mean={summary['mean_ms']:.3f} ms, "
            f"median={summary['median_ms']:.3f} ms, "
            f"p95={summary['p95_ms']:.3f} ms"
        )
        if summary["mean_us"] is not None:
            summary += f", mean_us={summary['mean_us']:.1f}"
        if summary["phase_timing_count"] > 0:
            summary += (
                f", dsp_mean_ms={summary['dsp_mean_ms']:.3f}"
                f", nn_mean_ms={summary['nn_mean_ms']:.3f}"
                f", pipeline_mean_ms={summary['pipeline_mean_ms']:.3f}"
            )
        print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
