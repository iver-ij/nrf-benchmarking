#!/usr/bin/env python3
"""Verify inference runtime constraint claims from source and UART records."""

from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ARENA_RE = re.compile(r"const\s+int\s+kTensorArenaSize\s*=\s*([^;]+);")
ARENA_USED_RE = re.compile(r"Tensor arena used:\s*(\d+)\s*/\s*(\d+)\s*bytes")
ARENA_KB_RE = re.compile(r"Tensor arena:\s*(\d+)\s*KB")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify inference runtime constraints")
    p.add_argument("--inference-source", default="firmware/src/inference.cpp")
    p.add_argument("--uart-log", default="benchmarks/results/dual_core_runtime_uart.log")
    p.add_argument(
        "--build-dir",
        default=None,
        help="Optional Zephyr build directory used to verify total data+bss RAM footprint",
    )
    p.add_argument(
        "--elf",
        default=None,
        help="Optional explicit ELF path (overrides --build-dir auto-detection)",
    )
    p.add_argument("--max-arena-kb", type=int, default=256)
    p.add_argument("--max-total-ram-kb", type=int, default=256)
    p.add_argument("--size-tool", default="arm-zephyr-eabi-size")
    p.add_argument(
        "--require-total-ram",
        action="store_true",
        help="Fail if total firmware RAM (data+bss) cannot be verified",
    )
    p.add_argument(
        "--summary-out",
        default="benchmarks/results/runtime_constraints_summary.json",
        help="Output summary JSON",
    )
    return p.parse_args()


def safe_eval_int(expr: str) -> int | None:
    cleaned = expr.strip()
    if not re.fullmatch(r"[0-9\s\(\)\+\-\*/]+", cleaned):
        return None
    try:
        value = eval(cleaned, {"__builtins__": {}}, {})
    except Exception:
        return None
    if not isinstance(value, (int, float)):
        return None
    return int(value)


def parse_runtime_uart(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {
            "uart_log_present": False,
            "logged_arena_used_bytes": None,
            "logged_arena_total_bytes": None,
            "logged_arena_kb": None,
        }

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    used_matches = ARENA_USED_RE.findall(text)
    kb_matches = ARENA_KB_RE.findall(text)

    used_bytes = int(used_matches[-1][0]) if used_matches else None
    total_bytes = int(used_matches[-1][1]) if used_matches else None
    logged_kb = int(kb_matches[-1]) if kb_matches else None

    return {
        "uart_log_present": True,
        "logged_arena_used_bytes": used_bytes,
        "logged_arena_total_bytes": total_bytes,
        "logged_arena_kb": logged_kb,
    }


def resolve_size_tool(size_tool: str) -> str | None:
    if Path(size_tool).exists():
        return size_tool

    found = shutil.which(size_tool)
    if found:
        return found

    sdk_root = Path("/opt/nordic/ncs/toolchains")
    pattern = sdk_root / "*" / "opt/zephyr-sdk/arm-zephyr-eabi/bin/arm-zephyr-eabi-size"
    matches = sorted(glob.glob(str(pattern)))
    if matches:
        return matches[-1]

    return None


def resolve_elf_path(build_dir: str | None, elf_override: str | None) -> Path | None:
    if elf_override:
        return Path(elf_override)
    if not build_dir:
        return None

    root = Path(build_dir)
    candidates = [
        root / "zephyr" / "zephyr.elf",
        root / "firmware" / "zephyr" / "zephyr.elf",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_size_summary(size_tool: str, elf_path: Path | None) -> dict[str, Any]:
    resolved_tool = resolve_size_tool(size_tool)

    if elf_path is None:
        return {
            "elf_path": None,
            "size_tool": size_tool,
            "size_tool_resolved": resolved_tool,
            "size_present": False,
            "data_bytes": None,
            "bss_bytes": None,
            "total_ram_bytes": None,
            "size_error": None,
        }

    if not elf_path.exists():
        return {
            "elf_path": str(elf_path),
            "size_tool": size_tool,
            "size_tool_resolved": resolved_tool,
            "size_present": False,
            "data_bytes": None,
            "bss_bytes": None,
            "total_ram_bytes": None,
            "size_error": "elf_not_found",
        }

    if resolved_tool is None:
        return {
            "elf_path": str(elf_path),
            "size_tool": size_tool,
            "size_tool_resolved": None,
            "size_present": False,
            "data_bytes": None,
            "bss_bytes": None,
            "total_ram_bytes": None,
            "size_error": f"size_tool_not_found:{size_tool}",
        }

    try:
        proc = subprocess.run(
            [resolved_tool, str(elf_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {
            "elf_path": str(elf_path),
            "size_tool": size_tool,
            "size_tool_resolved": resolved_tool,
            "size_present": False,
            "data_bytes": None,
            "bss_bytes": None,
            "total_ram_bytes": None,
            "size_error": str(exc),
        }

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    data_bytes = None
    bss_bytes = None
    for line in reversed(lines):
        parts = line.split()
        if len(parts) >= 6 and all(part.isdigit() for part in parts[:4]):
            data_bytes = int(parts[1])
            bss_bytes = int(parts[2])
            break

    total_ram_bytes = (
        (data_bytes + bss_bytes)
        if data_bytes is not None and bss_bytes is not None
        else None
    )

    return {
        "elf_path": str(elf_path),
        "size_tool": size_tool,
        "size_tool_resolved": resolved_tool,
        "size_present": total_ram_bytes is not None,
        "data_bytes": data_bytes,
        "bss_bytes": bss_bytes,
        "total_ram_bytes": total_ram_bytes,
        "size_error": None if total_ram_bytes is not None else "unparsed_size_output",
    }


def main() -> int:
    args = parse_args()
    source_path = Path(args.inference_source)
    if not source_path.exists():
        raise SystemExit(f"Missing inference source: {source_path}")

    source = source_path.read_text(encoding="utf-8", errors="ignore")

    arena_match = ARENA_RE.search(source)
    arena_expr = arena_match.group(1).strip() if arena_match else None
    arena_bytes = safe_eval_int(arena_expr) if arena_expr else None

    static_arena_decl = bool(
        re.search(r"static\s+uint8_t\s+tensor_arena\s*\[\s*kTensorArenaSize\s*\]", source)
    )
    static_interpreter_decl = bool(
        re.search(r"static\s+tflite::MicroInterpreter\s+static_interpreter\b", source)
    )
    allocate_tensors_called = "AllocateTensors()" in source

    malloc_calls = len(re.findall(r"\bmalloc\s*\(", source))
    new_calls = len(re.findall(r"\bnew\b", source))
    dynamic_alloc_calls = malloc_calls + new_calls

    runtime = parse_runtime_uart(Path(args.uart_log))
    size_summary = parse_size_summary(
        args.size_tool,
        resolve_elf_path(args.build_dir, args.elf),
    )
    max_arena_bytes = args.max_arena_kb * 1024
    max_total_ram_bytes = args.max_total_ram_kb * 1024

    arena_size_pass = arena_bytes is not None and arena_bytes <= max_arena_bytes
    static_alloc_pass = static_arena_decl and static_interpreter_decl and allocate_tensors_called
    no_dynamic_alloc_pass = dynamic_alloc_calls == 0

    uart_total_bytes = runtime["logged_arena_total_bytes"]
    runtime_arena_pass = uart_total_bytes is None or uart_total_bytes <= max_arena_bytes

    total_ram_bytes = size_summary["total_ram_bytes"]
    total_ram_pass = (
        total_ram_bytes is not None and total_ram_bytes <= max_total_ram_bytes
    )
    total_ram_gate = total_ram_pass if total_ram_bytes is not None else (not args.require_total_ram)

    overall_pass = (
        arena_size_pass
        and static_alloc_pass
        and no_dynamic_alloc_pass
        and runtime_arena_pass
        and total_ram_gate
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "inference_source": str(source_path),
            "uart_log": args.uart_log,
            "max_arena_kb": args.max_arena_kb,
            "build_dir": args.build_dir,
            "elf": args.elf,
            "max_total_ram_kb": args.max_total_ram_kb,
            "size_tool": args.size_tool,
            "require_total_ram": args.require_total_ram,
        },
        "source_checks": {
            "arena_expression": arena_expr,
            "arena_size_bytes": arena_bytes,
            "arena_size_pass": arena_size_pass,
            "static_arena_decl": static_arena_decl,
            "static_interpreter_decl": static_interpreter_decl,
            "allocate_tensors_called": allocate_tensors_called,
            "static_alloc_pass": static_alloc_pass,
            "malloc_calls": malloc_calls,
            "new_calls": new_calls,
            "no_dynamic_alloc_pass": no_dynamic_alloc_pass,
        },
        "runtime_checks": {
            **runtime,
            "runtime_arena_pass": runtime_arena_pass,
        },
        "binary_checks": {
            **size_summary,
            "max_total_ram_bytes": max_total_ram_bytes,
            "total_ram_pass": total_ram_pass if total_ram_bytes is not None else None,
        },
        "overall_pass": overall_pass,
        "notes": [
            "This verifies inference runtime memory constraints and, when a build/ELF is supplied, total firmware RAM via data+bss.",
        ],
    }

    out_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"arena_size_bytes={arena_bytes}")
    print(f"static_alloc_pass={static_alloc_pass}")
    print(f"no_dynamic_alloc_pass={no_dynamic_alloc_pass}")
    print(f"runtime_arena_pass={runtime_arena_pass}")
    print(f"total_ram_bytes={total_ram_bytes}")
    print(f"total_ram_pass={total_ram_pass if total_ram_bytes is not None else 'n/a'}")
    print(f"overall_pass={overall_pass}")
    print(f"summary={out_path}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
