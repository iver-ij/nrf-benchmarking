#!/usr/bin/env python3
"""
Parse QREF UART logs into machine-readable quantized tensors.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


BEGIN_RE = re.compile(
    r"^QREF_BEGIN\s+input_bytes=(\d+)\s+output_bytes=(\d+)\s+prob=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$"
)
IN_RE = re.compile(r"^QREF_IN\s+(\d+)\s+(-?\d+)$")
OUT_RE = re.compile(r"^QREF_OUT\s+(\d+)\s+(-?\d+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse quantized reference block from UART log")
    p.add_argument("--input", required=True, help="Path to captured UART log")
    p.add_argument(
        "--output",
        default="training/outputs/qref_tensors.json",
        help="Output JSON path",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise SystemExit(f"Input log not found: {in_path}")

    expected_in = None
    expected_out = None
    probability = None
    in_block = False
    input_vals: dict[int, int] = {}
    output_vals: dict[int, int] = {}

    for raw in in_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()

        begin = BEGIN_RE.match(line)
        if begin:
            in_block = True
            expected_in = int(begin.group(1))
            expected_out = int(begin.group(2))
            probability = float(begin.group(3))
            input_vals.clear()
            output_vals.clear()
            continue

        if not in_block:
            continue

        if line.startswith("QREF_END"):
            break

        m_in = IN_RE.match(line)
        if m_in:
            input_vals[int(m_in.group(1))] = int(m_in.group(2))
            continue

        m_out = OUT_RE.match(line)
        if m_out:
            output_vals[int(m_out.group(1))] = int(m_out.group(2))
            continue

    if expected_in is None or expected_out is None or probability is None:
        raise SystemExit("No complete QREF_BEGIN/QREF_END block found")

    missing_in = [i for i in range(expected_in) if i not in input_vals]
    missing_out = [i for i in range(expected_out) if i not in output_vals]
    if missing_in or missing_out:
        msg = (
            f"Incomplete QREF block. "
            f"input {expected_in - len(missing_in)}/{expected_in}, "
            f"output {expected_out - len(missing_out)}/{expected_out}."
        )
        if missing_in:
            msg += f" First missing input index: {missing_in[0]}."
        if missing_out:
            msg += f" First missing output index: {missing_out[0]}."
        raise SystemExit(msg)

    input_arr = [input_vals[i] for i in range(expected_in)]
    output_arr = [output_vals[i] for i in range(expected_out)]

    payload = {
        "input_bytes": expected_in,
        "output_bytes": expected_out,
        "firmware_probability": probability,
        "input_int8": input_arr,
        "output_int8": output_arr,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Parsed QREF block: input={expected_in} output={expected_out}")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
