#!/usr/bin/env python3
"""
One-command pipeline for quantized cross-platform parity:
1) flash export-enabled firmware
2) capture UART QREF block
3) parse int8 input/output tensors
4) compare hardware int8 output vs Python TFLite int8 output
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
    p = argparse.ArgumentParser(description="Run quantized cross-platform parity pipeline")
    p.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    p.add_argument(
        "--ncs-workspace",
        default=str(DEFAULT_NCS_WORKSPACE),
        help="NCS west workspace path. Defaults to $NCS_WORKSPACE or ~/ncs.",
    )
    p.add_argument("--west", default=None, help="Optional west executable override")
    p.add_argument("--build-dir", default=str(DEFAULT_WORKSPACE / "build_qref"))
    p.add_argument("--skip-flash", action="store_true")
    p.add_argument("--port", default=None)
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--capture-timeout", type=float, default=60.0)
    p.add_argument("--uart-log", default="training/outputs/qref_uart.log")
    p.add_argument("--qref-json", default="training/outputs/qref_tensors.json")
    p.add_argument("--summary-out", default="training/outputs/quantized_cross_platform_summary.json")
    p.add_argument("--model", default="training/outputs/model.tflite")
    p.add_argument(
        "--acceptance-mode",
        choices=["exact", "behavior", "threshold"],
        default="behavior",
        help="Acceptance policy passed to verify_quantized_cross_platform.py",
    )
    p.add_argument(
        "--target-class-index",
        type=int,
        default=2,
        help="Target class index for threshold decision checks",
    )
    p.add_argument(
        "--decision-threshold",
        type=float,
        default=0.75,
        help="Detection threshold for behavior/threshold acceptance",
    )
    p.add_argument(
        "--max-abs-diff-int8",
        type=int,
        default=1,
        help="Maximum allowed int8 logit drift for non-exact acceptance",
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
    qref_json = str((workspace / args.qref_json).resolve())
    summary_out = str((workspace / args.summary_out).resolve())
    model = str((workspace / args.model).resolve())

    try:
        if not args.skip_flash:
            run([west, "flash", "-d", str(build_dir)], cwd=ncs)

        capture_cmd = [
            python,
            str(workspace / "training/scripts/capture_quantized_reference_uart.py"),
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
                str(workspace / "training/scripts/parse_quantized_reference_uart.py"),
                "--input",
                uart_log,
                "--output",
                qref_json,
            ],
            cwd=workspace,
        )

        run(
            [
                python,
                str(workspace / "training/scripts/verify_quantized_cross_platform.py"),
                "--qref-json",
                qref_json,
                "--model",
                model,
                "--summary-out",
                summary_out,
                "--acceptance-mode",
                args.acceptance_mode,
                "--target-class-index",
                str(args.target_class_index),
                "--decision-threshold",
                str(args.decision_threshold),
                "--max-abs-diff-int8",
                str(args.max_abs_diff_int8),
            ],
            cwd=workspace,
        )
    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed at step with exit code {e.returncode}")
        return e.returncode

    print("Quantized parity pipeline completed.")
    print(f"Summary: {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
