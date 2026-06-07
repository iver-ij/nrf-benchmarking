#!/usr/bin/env python3
"""
Parse firmware UART log containing C reference spectrogram lines.

Expected firmware lines:
  C_REF_BEGIN rows=49 cols=40
  C_REF <idx> <value> [0xXXXXXXXX]
  ...
  C_REF_END

Produces:
  training/outputs/c_reference_spectrogram.txt
(one value per line, row-major order)
and
  training/outputs/c_reference_spectrogram_bits.txt
(one float32 hex bit-pattern per line)
"""

from __future__ import annotations

import argparse
import re
import struct
from pathlib import Path

SPECTROGRAM_ROWS = 49
NUM_MEL_BINS = 40
EXPECTED = SPECTROGRAM_ROWS * NUM_MEL_BINS

LINE_RE = re.compile(
    r"^C_REF\s+(\d+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?:\s+0x([0-9A-Fa-f]{8}))?\s*$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse C spectrogram dump from UART log")
    p.add_argument("--input", required=True, help="Path to captured UART log")
    p.add_argument(
        "--output",
        default="training/outputs/c_reference_spectrogram.txt",
        help="Path to output flattened spectrogram text file",
    )
    p.add_argument(
        "--bits-output",
        default="training/outputs/c_reference_spectrogram_bits.txt",
        help="Path to output float32 hex-bit file (one 0xXXXXXXXX per line)",
    )
    return p.parse_args()


def float32_from_u32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def u32_from_float32(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    bits_out_path = Path(args.bits_output)

    if not in_path.exists():
        raise SystemExit(f"Input log not found: {in_path}")

    values = [None] * EXPECTED
    bits = [None] * EXPECTED
    in_block = False

    for raw in in_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line.startswith("C_REF_BEGIN"):
            in_block = True
            continue
        if line.startswith("C_REF_END"):
            in_block = False
            continue
        if not in_block:
            continue

        m = LINE_RE.match(line)
        if not m:
            continue

        idx = int(m.group(1))
        parsed_value = float(m.group(2))
        parsed_hex = m.group(3)
        if 0 <= idx < EXPECTED:
            if parsed_hex is not None:
                u32 = int(parsed_hex, 16)
                val = float32_from_u32(u32)
            else:
                val = parsed_value
                u32 = u32_from_float32(val)
            values[idx] = val
            bits[idx] = u32

    missing = [i for i, v in enumerate(values) if v is None]
    if missing:
        raise SystemExit(
            f"Incomplete dump: got {EXPECTED - len(missing)}/{EXPECTED} values. "
            f"First missing index: {missing[0]}"
        )

    missing_bits = [i for i, b in enumerate(bits) if b is None]
    if missing_bits:
        raise SystemExit(
            f"Incomplete bit-pattern dump: got {EXPECTED - len(missing_bits)}/{EXPECTED} values. "
            f"First missing index: {missing_bits[0]}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bits_out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(f"{v:.9f}" for v in values) + "\n", encoding="utf-8")
    bits_out_path.write_text("\n".join(f"0x{b:08X}" for b in bits) + "\n", encoding="utf-8")

    print(f"Parsed {EXPECTED} values")
    print(f"Wrote: {out_path}")
    print(f"Wrote: {bits_out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
