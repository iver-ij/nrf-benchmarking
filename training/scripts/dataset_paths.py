#!/usr/bin/env python3
"""Shared dataset path resolution for the training pipeline."""

from __future__ import annotations

import os
import platform
from pathlib import Path

DATASET_NAME = "speech_commands_v0.02"
CACHE_NAMESPACE = "nrf-embedded-ml-benchmarks"


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_dataset_cache_dir() -> Path:
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Caches" / CACHE_NAMESPACE / "datasets" / DATASET_NAME

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser() / CACHE_NAMESPACE / "datasets" / DATASET_NAME

    return Path.home() / ".cache" / CACHE_NAMESPACE / "datasets" / DATASET_NAME


def candidate_dataset_dirs(explicit: str | Path | None = None) -> list[Path]:
    candidates: list[Path] = []

    if explicit:
        candidates.append(Path(explicit).expanduser())

    for env_name in ("NRF_EMBEDDED_ML_DATASET_DIR", "SPECTRAL_DATASET_DIR", "SPEECH_COMMANDS_DATASET"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value).expanduser())

    candidates.append(workspace_root() / "training" / "datasets" / DATASET_NAME)
    candidates.append(default_dataset_cache_dir())

    out: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
    return out


def resolve_dataset_dir(explicit: str | Path | None = None, *, must_exist: bool = True) -> Path:
    candidates = candidate_dataset_dirs(explicit)
    if not must_exist:
        return candidates[0]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    joined = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Speech Commands dataset not found. Looked in:\n"
        f"{joined}\n"
        "Set NRF_EMBEDDED_ML_DATASET_DIR or run training/scripts/fetch_speech_commands_dataset.py."
    )
