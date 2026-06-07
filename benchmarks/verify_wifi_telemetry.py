#!/usr/bin/env python3
"""Verify WiFi UDP telemetry behavior from UART logs."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

PATTERNS = {
    "udp_sent": re.compile(r"\[UDP\] Sent:"),
    "udp_send_failed": re.compile(r"\[UDP\] Send failed"),
    "udp_pending": re.compile(r"\[UDP\] Pending send"),
    "udp_drop_wifi": re.compile(r"\[UDP\] Drop \(wifi not connected\)"),
    "udp_drop_socket": re.compile(r"\[UDP\] Drop \(socket not ready\)"),
    "socket_connected": re.compile(r"\[UDP\] Socket connected"),
    "wifi_connected": re.compile(r"WiFi connected!|\[WiFi\] Connected"),
    "wifi_disconnected": re.compile(r"WiFi disconnected|\[WiFi\] Disconnected"),
    "ignored_late_116": re.compile(r"Ignoring late connect failure event:\s*-116"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify WiFi telemetry behavior from UART logs")
    p.add_argument("--uart-log", required=True, help="UART log path")
    p.add_argument(
        "--summary-out",
        default="benchmarks/results/wifi_telemetry_summary.json",
        help="Output summary JSON",
    )
    p.add_argument("--min-success-count", type=int, default=5)
    p.add_argument("--min-attempt-count", type=int, default=5)
    p.add_argument("--max-fail-ratio", type=float, default=0.2)
    p.add_argument("--max-disconnect-events", type=int, default=2)
    return p.parse_args()


def count_matches(text: str) -> dict[str, int]:
    return {name: len(pattern.findall(text)) for name, pattern in PATTERNS.items()}


def main() -> int:
    args = parse_args()
    log_path = Path(args.uart_log)
    if not log_path.exists():
        raise SystemExit(f"Missing UART log: {log_path}")

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    counts = count_matches(text)

    attempts = counts["udp_sent"] + counts["udp_send_failed"]
    fail_ratio = (counts["udp_send_failed"] / attempts) if attempts > 0 else 1.0

    enough_success = counts["udp_sent"] >= args.min_success_count
    enough_attempts = attempts >= args.min_attempt_count
    fail_ratio_pass = (fail_ratio <= args.max_fail_ratio) if enough_attempts else False
    disconnects_pass = counts["wifi_disconnected"] <= args.max_disconnect_events

    overall_pass = enough_success and enough_attempts and fail_ratio_pass and disconnects_pass

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "uart_log": str(log_path),
        "thresholds": {
            "min_success_count": args.min_success_count,
            "min_attempt_count": args.min_attempt_count,
            "max_fail_ratio": args.max_fail_ratio,
            "max_disconnect_events": args.max_disconnect_events,
        },
        "counts": counts,
        "metrics": {
            "attempts": attempts,
            "success_count": counts["udp_sent"],
            "failure_count": counts["udp_send_failed"],
            "failure_ratio": fail_ratio,
        },
        "checks": {
            "enough_success": enough_success,
            "enough_attempts": enough_attempts,
            "fail_ratio_pass": fail_ratio_pass,
            "disconnects_pass": disconnects_pass,
            "overall_pass": overall_pass,
        },
    }

    out_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"attempts={attempts}")
    print(f"success={counts['udp_sent']}")
    print(f"failure={counts['udp_send_failed']}")
    print(f"failure_ratio={fail_ratio:.3f}")
    print(f"overall_pass={overall_pass}")
    print(f"summary={out_path}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
