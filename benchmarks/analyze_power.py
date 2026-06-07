#!/usr/bin/env python3
"""
Analyze Nordic PPK2 CSV traces and estimate per-inference energy.

This script is intentionally offline-first: it consumes exported CSV traces from
PPK2/Power Profiler apps and generates a reproducible JSON summary.

It does not simulate measurements. If no trace is provided, it exits.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


def integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


@dataclass
class TraceData:
    time_ms: np.ndarray
    current_ma: np.ndarray
    power_mw: np.ndarray
    time_col: str
    current_col: str
    power_col: Optional[str]


@dataclass
class EventStats:
    start_ms: float
    end_ms: float
    duration_ms: float
    energy_uj: float
    avg_current_ma: float
    peak_current_ma: float


@dataclass(frozen=True)
class DetectionProfile:
    name: str
    threshold_sigma: float
    min_delta_ma: float
    min_event_ms: float


DSP_AVG_US_RE = re.compile(
    r"DSP Frontend\s*\(.*?\):.*?Average:\s*([0-9]+(?:\.[0-9]+)?)\s*us",
    re.IGNORECASE | re.DOTALL,
)
NN_AVG_US_RE = re.compile(
    r"Neural Network Inference:\s*.*?Average:\s*([0-9]+(?:\.[0-9]+)?)\s*us",
    re.IGNORECASE | re.DOTALL,
)
PHASE_TIMING_RE = re.compile(
    r"PHASE_TIMING\s+dsp_us=(\d+)\s+nn_us=(\d+)\s+total_us=(\d+)",
    re.IGNORECASE,
)

DETECTION_PROFILES = {
    "burst": DetectionProfile(
        name="burst",
        threshold_sigma=4.0,
        min_delta_ma=0.2,
        min_event_ms=5.0,
    ),
    "steady_inference": DetectionProfile(
        name="steady_inference",
        threshold_sigma=1.5,
        min_delta_ma=0.05,
        min_event_ms=0.5,
    ),
}


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _infer_time_multiplier(col_name: str) -> float:
    n = col_name.lower()
    if "[us]" in n or "(us)" in n or "_us" in n:
        return 1.0 / 1000.0
    if "[s]" in n or "(s)" in n or n.endswith(" s"):
        return 1000.0
    return 1.0  # default ms


def _infer_current_multiplier(col_name: str) -> float:
    n = col_name.lower()
    if "[ua]" in n or "(ua)" in n or "_ua" in n:
        return 1.0 / 1000.0
    if "[a]" in n or "(a)" in n or n.endswith(" a"):
        return 1000.0
    return 1.0  # default mA


def _infer_power_multiplier(col_name: str) -> float:
    n = col_name.lower()
    if "[uw]" in n or "(uw)" in n or "_uw" in n:
        return 1.0 / 1000.0
    if "[w]" in n or "(w)" in n or n.endswith(" w"):
        return 1000.0
    return 1.0  # default mW


def _pick_column(fieldnames: Sequence[str], tokens: Sequence[str], fallback: Optional[int] = None) -> Optional[str]:
    for f in fieldnames:
        nf = _normalize(f)
        if all(tok in nf for tok in tokens):
            return f
    for tok in tokens:
        for f in fieldnames:
            if tok in _normalize(f):
                return f
    if fallback is not None and 0 <= fallback < len(fieldnames):
        return fieldnames[fallback]
    return None


def load_trace(csv_path: Path, voltage_v: float) -> TraceData:
    if not csv_path.exists():
        raise FileNotFoundError(f"Trace CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header. Export with column names from PPK2 tool.")

        fieldnames = [fn.strip() for fn in reader.fieldnames]

        time_col = _pick_column(fieldnames, ["time"], fallback=0)
        current_col = _pick_column(fieldnames, ["current"], fallback=1)
        power_col = _pick_column(fieldnames, ["power"], fallback=None)

        if time_col is None or current_col is None:
            raise ValueError(
                "Could not identify required columns. Need time and current columns in CSV."
            )

        t_mul = _infer_time_multiplier(time_col)
        i_mul = _infer_current_multiplier(current_col)
        p_mul = _infer_power_multiplier(power_col) if power_col else 1.0

        times: List[float] = []
        currents: List[float] = []
        powers: List[float] = []

        for row in reader:
            try:
                t_ms = float(row[time_col]) * t_mul
                i_ma = float(row[current_col]) * i_mul
                if not np.isfinite(t_ms) or not np.isfinite(i_ma):
                    continue
                times.append(t_ms)
                currents.append(i_ma)

                if power_col is not None:
                    p_mw = float(row[power_col]) * p_mul
                else:
                    p_mw = i_ma * voltage_v
                powers.append(p_mw)
            except (KeyError, TypeError, ValueError):
                continue

    if len(times) < 3:
        raise ValueError("Trace has too few valid samples after parsing")

    time_ms = np.asarray(times, dtype=np.float64)
    current_ma = np.asarray(currents, dtype=np.float64)
    power_mw = np.asarray(powers, dtype=np.float64)

    # Ensure time is strictly monotonic for stable integration.
    order = np.argsort(time_ms)
    time_ms = time_ms[order]
    current_ma = current_ma[order]
    power_mw = power_mw[order]

    unique = np.concatenate(([True], np.diff(time_ms) > 0))
    time_ms = time_ms[unique]
    current_ma = current_ma[unique]
    power_mw = power_mw[unique]

    return TraceData(
        time_ms=time_ms,
        current_ma=current_ma,
        power_mw=power_mw,
        time_col=time_col,
        current_col=current_col,
        power_col=power_col,
    )


def detect_activity_events(
    time_ms: np.ndarray,
    current_ma: np.ndarray,
    power_mw: np.ndarray,
    threshold_sigma: float,
    min_delta_ma: float,
    min_event_ms: float,
) -> Tuple[float, float, List[EventStats]]:
    baseline_ma = float(np.percentile(current_ma, 10.0))
    median_ma = float(np.median(current_ma))
    mad = float(np.median(np.abs(current_ma - median_ma)))
    robust_sigma = 1.4826 * mad

    threshold_ma = max(
        baseline_ma + min_delta_ma,
        baseline_ma + threshold_sigma * robust_sigma,
    )

    active = current_ma > threshold_ma

    events: List[EventStats] = []
    start = None
    for idx, is_active in enumerate(active):
        if is_active and start is None:
            start = idx
            continue

        if not is_active and start is not None:
            end = idx - 1
            duration = float(time_ms[end] - time_ms[start])
            if duration >= min_event_ms:
                energy_uj = integrate_trapezoid(power_mw[start:end + 1], time_ms[start:end + 1])
                events.append(
                    EventStats(
                        start_ms=float(time_ms[start]),
                        end_ms=float(time_ms[end]),
                        duration_ms=duration,
                        energy_uj=energy_uj,
                        avg_current_ma=float(np.mean(current_ma[start:end + 1])),
                        peak_current_ma=float(np.max(current_ma[start:end + 1])),
                    )
                )
            start = None

    if start is not None:
        end = len(time_ms) - 1
        duration = float(time_ms[end] - time_ms[start])
        if duration >= min_event_ms:
            energy_uj = integrate_trapezoid(power_mw[start:end + 1], time_ms[start:end + 1])
            events.append(
                EventStats(
                    start_ms=float(time_ms[start]),
                    end_ms=float(time_ms[end]),
                    duration_ms=duration,
                    energy_uj=energy_uj,
                    avg_current_ma=float(np.mean(current_ma[start:end + 1])),
                    peak_current_ma=float(np.max(current_ma[start:end + 1])),
                )
            )

    return baseline_ma, threshold_ma, events


def detect_activity_events_with_profiles(
    time_ms: np.ndarray,
    current_ma: np.ndarray,
    power_mw: np.ndarray,
    profiles: Sequence[DetectionProfile],
    target_event_ms: Optional[float] = None,
) -> tuple[float, float, List[EventStats], DetectionProfile, list[dict[str, float | int | str]]]:
    attempts: list[dict[str, float | int | str]] = []
    fallback_result: Optional[tuple[float, float, List[EventStats], DetectionProfile]] = None

    for profile in profiles:
        baseline_ma, threshold_ma, events = detect_activity_events(
            time_ms,
            current_ma,
            power_mw,
            threshold_sigma=profile.threshold_sigma,
            min_delta_ma=profile.min_delta_ma,
            min_event_ms=profile.min_event_ms,
        )
        mean_duration_ms = float(np.mean([e.duration_ms for e in events])) if events else 0.0
        attempts.append(
            {
                "profile": profile.name,
                "threshold_sigma": profile.threshold_sigma,
                "min_delta_ma": profile.min_delta_ma,
                "min_event_ms": profile.min_event_ms,
                "baseline_current_ma": baseline_ma,
                "activity_threshold_ma": threshold_ma,
                "event_count": len(events),
                "mean_event_duration_ms": mean_duration_ms,
            }
        )
        if not events:
            continue

        if target_event_ms is None:
            return baseline_ma, threshold_ma, events, profile, attempts

        # Reject noisy micro-spike detections when phase timing says a real
        # inference window should be much longer.
        if mean_duration_ms >= max(0.25 * target_event_ms, profile.min_event_ms):
            return baseline_ma, threshold_ma, events, profile, attempts

        if fallback_result is None:
            fallback_result = (baseline_ma, threshold_ma, events, profile)

    if fallback_result is not None:
        baseline_ma, threshold_ma, events, profile = fallback_result
        return baseline_ma, threshold_ma, events, profile, attempts

    baseline_ma, threshold_ma, events = detect_activity_events(
        time_ms,
        current_ma,
        power_mw,
        threshold_sigma=profiles[0].threshold_sigma,
        min_delta_ma=profiles[0].min_delta_ma,
        min_event_ms=profiles[0].min_event_ms,
    )
    return baseline_ma, threshold_ma, events, profiles[0], attempts


def _integrate_window(time_ms: np.ndarray, power_mw: np.ndarray, t0: float, t1: float) -> float:
    mask = (time_ms >= t0) & (time_ms <= t1)
    if np.count_nonzero(mask) < 2:
        return 0.0
    return integrate_trapezoid(power_mw[mask], time_ms[mask])


def parse_phase_durations_from_uart_log(
    log_path: Path,
) -> tuple[Optional[float], Optional[float], dict[str, Any], list[tuple[float, float]]]:
    if not log_path.exists():
        return None, None, {
            "source": "uart_log",
            "uart_log": str(log_path),
            "uart_log_present": False,
            "dsp_matches": 0,
            "nn_matches": 0,
            "phase_timing_matches": 0,
        }, []

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    dsp_matches = [float(v) for v in DSP_AVG_US_RE.findall(text)]
    nn_matches = [float(v) for v in NN_AVG_US_RE.findall(text)]
    phase_matches = [
        (float(dsp_us) / 1000.0, float(nn_us) / 1000.0)
        for dsp_us, nn_us, _ in PHASE_TIMING_RE.findall(text)
        if int(nn_us) > 0
    ]

    if phase_matches:
        dsp_ms = float(np.mean([pair[0] for pair in phase_matches]))
        nn_ms = float(np.mean([pair[1] for pair in phase_matches]))
        details = {
            "source": "uart_phase_timing_markers",
            "uart_log": str(log_path),
            "uart_log_present": True,
            "dsp_matches": len(dsp_matches),
            "nn_matches": len(nn_matches),
            "phase_timing_matches": len(phase_matches),
            "dsp_ms_selected": dsp_ms,
            "nn_ms_selected": nn_ms,
        }
        return dsp_ms, nn_ms, details, phase_matches

    dsp_ms = (dsp_matches[-1] / 1000.0) if dsp_matches else None
    nn_ms = (nn_matches[-1] / 1000.0) if nn_matches else None
    details = {
        "source": "uart_benchmark_averages",
        "uart_log": str(log_path),
        "uart_log_present": True,
        "dsp_matches": len(dsp_matches),
        "nn_matches": len(nn_matches),
        "phase_timing_matches": 0,
        "dsp_us_selected": dsp_matches[-1] if dsp_matches else None,
        "nn_us_selected": nn_matches[-1] if nn_matches else None,
    }
    return dsp_ms, nn_ms, details, []


def build_summary(
    trace: TraceData,
    voltage_v: float,
    baseline_ma: float,
    threshold_ma: float,
    events: Sequence[EventStats],
    dsp_ms: Optional[float],
    nn_ms: Optional[float],
    phase_split_config: Optional[dict[str, Any]] = None,
    phase_windows_ms: Optional[Sequence[Tuple[float, float]]] = None,
    detection_details: Optional[dict[str, Any]] = None,
) -> dict:
    dt = np.diff(trace.time_ms)
    median_dt_ms = float(np.median(dt)) if len(dt) else 0.0
    sample_rate_hz = float(1000.0 / median_dt_ms) if median_dt_ms > 0 else 0.0

    durations = np.asarray([e.duration_ms for e in events], dtype=np.float64)
    energies = np.asarray([e.energy_uj for e in events], dtype=np.float64)
    avg_currents = np.asarray([e.avg_current_ma for e in events], dtype=np.float64)

    summary = {
        "trace_columns": {
            "time": trace.time_col,
            "current": trace.current_col,
            "power": trace.power_col,
        },
        "measurement": {
            "sample_count": int(len(trace.time_ms)),
            "median_dt_ms": median_dt_ms,
            "estimated_sample_rate_hz": sample_rate_hz,
            "voltage_v": float(voltage_v),
            "baseline_current_ma": baseline_ma,
            "baseline_power_mw": baseline_ma * voltage_v,
            "activity_threshold_ma": threshold_ma,
        },
        "events": {
            "count": int(len(events)),
            "duration_ms": {
                "mean": float(np.mean(durations)) if len(durations) else 0.0,
                "min": float(np.min(durations)) if len(durations) else 0.0,
                "max": float(np.max(durations)) if len(durations) else 0.0,
            },
            "energy_uj": {
                "mean": float(np.mean(energies)) if len(energies) else 0.0,
                "std": float(np.std(energies)) if len(energies) else 0.0,
                "min": float(np.min(energies)) if len(energies) else 0.0,
                "max": float(np.max(energies)) if len(energies) else 0.0,
            },
            "avg_current_ma": float(np.mean(avg_currents)) if len(avg_currents) else 0.0,
        },
    }

    if phase_split_config is not None:
        summary["phase_split_config"] = phase_split_config
    if detection_details is not None:
        summary["detection"] = detection_details

    if dsp_ms is not None and nn_ms is not None and dsp_ms > 0 and nn_ms > 0 and len(events) > 0:
        mean_event_duration_ms = float(np.mean(durations)) if len(durations) else 0.0
        full_pipeline_ms = dsp_ms + nn_ms
        if mean_event_duration_ms < (0.75 * full_pipeline_ms):
            summary["phase_energy_warning"] = {
                "reason": "detected_event_window_shorter_than_full_pipeline",
                "mean_event_duration_ms": mean_event_duration_ms,
                "expected_pipeline_ms": full_pipeline_ms,
                "dsp_ms": dsp_ms,
                "nn_ms": nn_ms,
                "notes": (
                    "Detected power windows are materially shorter than the UART-derived "
                    "DSP+NN pipeline. Per-phase energy attribution is suppressed because "
                    "the thresholded event likely captures only the highest-current subset "
                    "of the full pipeline."
                ),
            }
            return summary

        dsp_energy = 0.0
        nn_energy = 0.0
        phase_pair_source = "mean_windows"
        phase_pair_count = 0
        phase_pairs: Sequence[Tuple[float, float]] = []

        if phase_windows_ms:
            phase_pairs = list(phase_windows_ms)
            phase_pair_count = len(phase_pairs)
            if len(phase_pairs) == len(events):
                phase_pair_source = "event_aligned_windows"

        for idx, ev in enumerate(events):
            current_dsp_ms = dsp_ms
            current_nn_ms = nn_ms
            if phase_pair_source == "event_aligned_windows":
                current_dsp_ms, current_nn_ms = phase_pairs[idx]
            dsp_start = ev.start_ms
            dsp_end = min(ev.end_ms, dsp_start + current_dsp_ms)
            nn_end = min(ev.end_ms, dsp_end + current_nn_ms)
            dsp_energy += _integrate_window(trace.time_ms, trace.power_mw, dsp_start, dsp_end)
            nn_energy += _integrate_window(trace.time_ms, trace.power_mw, dsp_end, nn_end)

        attributed_total = dsp_energy + nn_energy
        event_total = float(np.sum(energies))
        summary["phase_energy_uj"] = {
            "dsp_total": dsp_energy,
            "nn_total": nn_energy,
            "attributed_total": attributed_total,
            "event_total": event_total,
            "unattributed_total": max(event_total - attributed_total, 0.0),
            "dsp_percent_of_attributed": (
                100.0 * dsp_energy / attributed_total
            ) if attributed_total > 0 else 0.0,
            "nn_percent_of_attributed": (
                100.0 * nn_energy / attributed_total
            ) if attributed_total > 0 else 0.0,
            "attributed_fraction_of_event_energy": (
                attributed_total / event_total
            ) if event_total > 0 else 0.0,
            "phase_window_source": phase_pair_source,
            "phase_window_count": phase_pair_count,
            "notes": (
                "Split uses UART-derived DSP/NN phase windows from each detected "
                "event start and clips windows to event boundaries."
            ),
        }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze PPK2 CSV trace for inference energy")
    parser.add_argument("--csv", required=True, help="Input CSV exported from PPK2/Power Profiler")
    parser.add_argument(
        "--voltage",
        type=float,
        default=3.6,
        help="Supply voltage used for power conversion (nRF7002DK+PPK2 reference uses 3.6V)",
    )
    parser.add_argument("--threshold-sigma", type=float, default=4.0, help="Activity threshold in robust sigma units")
    parser.add_argument("--min-delta-ma", type=float, default=0.2, help="Minimum threshold above baseline current")
    parser.add_argument("--min-event-ms", type=float, default=5.0, help="Minimum event duration to count")
    parser.add_argument(
        "--detection-profile",
        choices=["auto", *DETECTION_PROFILES.keys()],
        default="auto",
        help=(
            "Detection profile to use. 'auto' tries the default burst profile first "
            "and falls back to a steady-inference profile for low-contrast recurring workloads."
        ),
    )
    parser.add_argument(
        "--expect-min-mean-ma",
        type=float,
        default=1.0,
        help="Fail if trace mean current is below this (guards against broken wiring / idle captures)",
    )
    parser.add_argument("--dsp-ms", type=float, default=None, help="Optional DSP phase duration for energy split")
    parser.add_argument("--nn-ms", type=float, default=None, help="Optional NN phase duration for energy split")
    parser.add_argument(
        "--phase-uart-log",
        default=None,
        help=(
            "Optional UART log path. If --dsp-ms/--nn-ms are omitted, "
            "the script extracts DSP/NN durations from PHASE_TIMING markers "
            "or benchmark averages in this log."
        ),
    )
    parser.add_argument(
        "--require-phase-split",
        action="store_true",
        help="Fail if DSP/NN phase durations are not available for split attribution",
    )
    parser.add_argument("--output", default="benchmarks/power_summary.json", help="Output JSON summary path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    phase_dsp_ms = args.dsp_ms
    phase_nn_ms = args.nn_ms
    phase_split_config: dict[str, Any] | None = None
    phase_windows_ms: list[tuple[float, float]] = []

    if args.phase_uart_log:
        parsed_dsp_ms, parsed_nn_ms, parse_details, phase_windows_ms = parse_phase_durations_from_uart_log(
            Path(args.phase_uart_log)
        )
        if phase_dsp_ms is None:
            phase_dsp_ms = parsed_dsp_ms
        if phase_nn_ms is None:
            phase_nn_ms = parsed_nn_ms
        phase_split_config = {
            **parse_details,
            "dsp_ms": phase_dsp_ms,
            "nn_ms": phase_nn_ms,
            "phase_window_count": len(phase_windows_ms),
            "manual_override": bool(args.dsp_ms is not None or args.nn_ms is not None),
        }
    elif phase_dsp_ms is not None or phase_nn_ms is not None:
        phase_split_config = {
            "source": "manual_args",
            "dsp_ms": phase_dsp_ms,
            "nn_ms": phase_nn_ms,
        }

    if (phase_dsp_ms is None) != (phase_nn_ms is None):
        raise ValueError(
            "Incomplete phase split configuration: both --dsp-ms and --nn-ms are required."
        )
    if phase_dsp_ms is not None and phase_dsp_ms <= 0:
        raise ValueError(f"Invalid --dsp-ms: {phase_dsp_ms}. Expected > 0.")
    if phase_nn_ms is not None and phase_nn_ms <= 0:
        raise ValueError(f"Invalid --nn-ms: {phase_nn_ms}. Expected > 0.")
    if args.require_phase_split and (phase_dsp_ms is None or phase_nn_ms is None):
        raise ValueError(
            "Phase split required but DSP/NN durations are missing. Provide "
            "--dsp-ms/--nn-ms or --phase-uart-log containing PHASE_TIMING markers "
            "or benchmark averages."
        )

    trace = load_trace(Path(args.csv), args.voltage)
    mean_current_ma = float(np.mean(trace.current_ma))
    if mean_current_ma < args.expect_min_mean_ma:
        raise ValueError(
            "Trace mean current is too low for a running nRF7002DK workload "
            f"({mean_current_ma:.6f} mA < {args.expect_min_mean_ma:.3f} mA). "
            "Likely capture/setup issue. For nRF7002DK power profiling: remove jumper P23 (VBAT), "
            "connect PPK2 VOUT -> P23 pin 1, GND -> board GND (e.g. P21 pin 1), "
            "set PPK2 Source Meter to 3.6V, then recapture."
        )
    override_thresholds = any(
        (
            args.threshold_sigma != DETECTION_PROFILES["burst"].threshold_sigma,
            args.min_delta_ma != DETECTION_PROFILES["burst"].min_delta_ma,
            args.min_event_ms != DETECTION_PROFILES["burst"].min_event_ms,
        )
    )
    if override_thresholds:
        profiles = [
            DetectionProfile(
                name="manual",
                threshold_sigma=args.threshold_sigma,
                min_delta_ma=args.min_delta_ma,
                min_event_ms=args.min_event_ms,
            )
        ]
    elif args.detection_profile == "auto":
        profiles = [
            DETECTION_PROFILES["burst"],
            DETECTION_PROFILES["steady_inference"],
        ]
    else:
        profiles = [DETECTION_PROFILES[args.detection_profile]]

    target_event_ms = phase_nn_ms if phase_nn_ms is not None else None
    baseline_ma, threshold_ma, events, selected_profile, detection_attempts = detect_activity_events_with_profiles(
        trace.time_ms,
        trace.current_ma,
        trace.power_mw,
        profiles,
        target_event_ms=target_event_ms,
    )

    if not events:
        print("No activity events detected. Adjust threshold args or verify trace quality.")

    summary = build_summary(
        trace,
        voltage_v=args.voltage,
        baseline_ma=baseline_ma,
        threshold_ma=threshold_ma,
        events=events,
        dsp_ms=phase_dsp_ms,
        nn_ms=phase_nn_ms,
        phase_split_config=phase_split_config,
        phase_windows_ms=phase_windows_ms,
        detection_details={
            "selected_profile": selected_profile.name,
            "manual_thresholds": override_thresholds,
            "attempts": detection_attempts,
        },
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Power Analysis Summary ===")
    print(f"Samples: {summary['measurement']['sample_count']}")
    print(f"Baseline current: {summary['measurement']['baseline_current_ma']:.3f} mA")
    if "detection" in summary:
        print(f"Detection profile: {summary['detection']['selected_profile']}")
    print(f"Detected events: {summary['events']['count']}")
    print(f"Mean event energy: {summary['events']['energy_uj']['mean']:.2f} uJ")
    phase_energy = summary.get("phase_energy_uj")
    if phase_energy is not None:
        print(
            "Phase split: "
            f"DSP={phase_energy['dsp_percent_of_attributed']:.2f}% "
            f"NN={phase_energy['nn_percent_of_attributed']:.2f}% "
            f"(attributed={phase_energy['attributed_fraction_of_event_energy'] * 100.0:.2f}% "
            "of event energy)"
        )
    print(f"Summary: {output_path}")

    return 0 if len(events) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
