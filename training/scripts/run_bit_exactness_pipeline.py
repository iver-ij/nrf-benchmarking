#!/usr/bin/env python3
"""
One-command bit-exactness pipeline:
1) flash export-enabled firmware
2) capture UART C_REF block
3) parse C reference spectrogram
4) run numerical equivalence verifier
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_NCS_WORKSPACE = Path(os.environ.get("NCS_WORKSPACE", "~/ncs")).expanduser()


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run bit-exactness verification pipeline")
    p.add_argument(
        "--workspace",
        default=str(DEFAULT_WORKSPACE),
        help="Repository workspace path",
    )
    p.add_argument(
        "--ncs-workspace",
        default=str(DEFAULT_NCS_WORKSPACE),
        help="NCS west workspace path. Defaults to $NCS_WORKSPACE or ~/ncs.",
    )
    p.add_argument(
        "--west",
        default=None,
        help="Optional west executable override",
    )
    p.add_argument(
        "--build-dir",
        default=str(DEFAULT_WORKSPACE / "build_c_ref"),
        help="Build dir containing export-enabled firmware",
    )
    p.add_argument("--skip-flash", action="store_true", help="Skip flashing step")
    p.add_argument("--port", default=None, help="Serial port override")
    p.add_argument("--baud", type=int, default=115200, help="UART baud")
    p.add_argument("--capture-timeout", type=float, default=60.0, help="UART capture timeout")
    p.add_argument(
        "--tolerance",
        type=float,
        default=5e-1,
        help="Absolute tolerance passed to numerical equivalence verifier",
    )
    p.add_argument(
        "--uart-log",
        default="training/outputs/c_ref_uart.log",
        help="UART log output path (workspace-relative)",
    )
    p.add_argument(
        "--c-reference",
        default="training/outputs/c_reference_spectrogram.txt",
        help="Parsed C reference output path (workspace-relative)",
    )
    p.add_argument(
        "--c-reference-bits",
        default="training/outputs/c_reference_spectrogram_bits.txt",
        help="Parsed C reference float32 bit-pattern output path (workspace-relative)",
    )
    p.add_argument(
        "--summary-out",
        default="training/outputs/numerical_equivalence_summary.json",
        help="Verifier summary path (workspace-relative)",
    )
    p.add_argument(
        "--bit-exact-mode",
        choices=["cross_impl", "replay_c_reference"],
        default="cross_impl",
        help="Bit-exact verification mode passed to verify_numerical_equivalence.py",
    )
    return p.parse_args()


def resolve_west(args: argparse.Namespace, workspace: Path) -> str:
    candidates: list[Path] = []

    if args.west:
        candidates.append(Path(args.west).expanduser())

    west_env = os.environ.get("WEST")
    if west_env:
        candidates.append(Path(west_env).expanduser())

    candidates.append(workspace / ".venv/bin/west")

    path_west = shutil.which("west")
    if path_west:
        candidates.append(Path(path_west))

    toolchains_root = Path("/opt/nordic/ncs/toolchains")
    if toolchains_root.exists():
        candidates.extend(sorted(toolchains_root.glob("*/Cellar/python@*/**/bin/west")))

    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return key

    raise FileNotFoundError(
        "west not found. Pass --west, set $WEST, or install west in PATH / Nordic toolchain."
    )


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    ncs = Path(args.ncs_workspace).expanduser().resolve()
    build_dir = Path(args.build_dir)
    if not build_dir.is_absolute():
        build_dir = workspace / build_dir
    build_dir = build_dir.resolve()

    python = sys.executable
    west = resolve_west(args, workspace)

    uart_log = str((workspace / args.uart_log).resolve())
    c_reference = str((workspace / args.c_reference).resolve())
    c_reference_bits = str((workspace / args.c_reference_bits).resolve())
    summary_out = str((workspace / args.summary_out).resolve())

    try:
        if not args.skip_flash:
            run([west, "flash", "-d", str(build_dir)], cwd=ncs)

        capture_cmd = [
            python,
            str(workspace / "training/scripts/capture_c_reference_uart.py"),
            "--baud",
            str(args.baud),
            "--timeout",
            str(args.capture_timeout),
            "--output",
            uart_log,
        ]
        if args.port:
            capture_cmd.extend(["--port", args.port])
        run(capture_cmd, cwd=workspace)

        run(
            [
                python,
                str(workspace / "training/scripts/parse_c_reference_uart.py"),
                "--input",
                uart_log,
                "--output",
                c_reference,
                "--bits-output",
                c_reference_bits,
            ],
            cwd=workspace,
        )

        run(
            [
                python,
                str(workspace / "training/scripts/verify_numerical_equivalence.py"),
                "--c-reference",
                c_reference,
                "--c-reference-bits",
                c_reference_bits,
                "--summary-out",
                summary_out,
                "--tolerance",
                str(args.tolerance),
                "--bit-exact-mode",
                args.bit_exact_mode,
                "--fail-on-missing-c",
            ],
            cwd=workspace,
        )

    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed at step with exit code {e.returncode}")
        return e.returncode

    print("Bit-exactness pipeline completed.")
    print(f"Summary: {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
