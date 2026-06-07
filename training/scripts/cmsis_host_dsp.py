#!/usr/bin/env python3
"""
Host-side CMSIS-DSP frontend bridge.

This module builds a small C shared library that reuses the firmware DSP
frontend logic (FFT + sparse mel + log) and exposes it to Python.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
NUM_SAMPLES = 16000
SPECTROGRAM_ROWS = 49
NUM_MEL_BINS = 40
SPECTROGRAM_SIZE = SPECTROGRAM_ROWS * NUM_MEL_BINS


def _workspace() -> Path:
    return Path(__file__).resolve().parents[2]


def _ncs_workspace() -> Path:
    return Path(os.environ.get("NCS_WORKSPACE", "~/ncs")).expanduser()


def _shared_lib_path() -> Path:
    ext = ".dylib" if os.uname().sysname == "Darwin" else ".so"
    return _workspace() / "training" / "outputs" / f"libtinyml_host_dsp{ext}"


def _sources() -> list[Path]:
    ws = _workspace()
    cmsis = _ncs_workspace() / "modules/lib/cmsis-dsp"
    return [
        ws / "training" / "c_dsp" / "host_dsp.c",
        ws / "firmware" / "src" / "mel_constants.c",
        cmsis / "Source/TransformFunctions/arm_rfft_fast_f32.c",
        cmsis / "Source/TransformFunctions/arm_rfft_fast_init_f32.c",
        cmsis / "Source/TransformFunctions/arm_cfft_f32.c",
        cmsis / "Source/TransformFunctions/arm_cfft_init_f32.c",
        cmsis / "Source/TransformFunctions/arm_cfft_radix8_f32.c",
        cmsis / "Source/TransformFunctions/arm_bitreversal2.c",
        cmsis / "Source/CommonTables/arm_const_structs.c",
        cmsis / "Source/CommonTables/arm_common_tables.c",
        cmsis / "Source/ComplexMathFunctions/arm_cmplx_mag_f32.c",
        cmsis / "Source/FastMathFunctions/arm_cos_f32.c",
        cmsis / "Source/FastMathFunctions/arm_sin_f32.c",
        cmsis / "Source/MatrixFunctions/arm_mat_vec_mult_f32.c",
    ]


def _build_host_library(lib_path: Path) -> None:
    ws = _workspace()
    ncs = _ncs_workspace()
    cmsis_include = ncs / "modules/lib/cmsis-dsp/Include"
    cmsis_core_include = ncs / "modules/hal/cmsis/CMSIS/Core/Include"
    firmware_include = ws / "firmware" / "include"
    srcs = _sources()

    missing = [str(p) for p in srcs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing CMSIS host build sources: {missing}")

    compiler = os.environ.get("CC") or shutil.which("clang") or shutil.which("gcc")
    if compiler is None:
        raise FileNotFoundError("No C compiler found. Set CC or install clang/gcc.")

    lib_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        compiler,
        "-O3",
        "-std=c11",
        "-shared",
        "-fPIC",
        "-ffp-contract=off",
        "-fno-fast-math",
        f"-I{firmware_include}",
        f"-I{cmsis_include}",
        f"-I{cmsis_core_include}",
        *[str(p) for p in srcs],
        "-lm",
        "-o",
        str(lib_path),
    ]
    subprocess.run(cmd, check=True)


def _needs_rebuild(lib_path: Path, srcs: list[Path]) -> bool:
    if not lib_path.exists():
        return True
    lib_mtime = lib_path.stat().st_mtime
    return any(p.stat().st_mtime > lib_mtime for p in srcs)


class _HostDSP:
    def __init__(self) -> None:
        self.lib = None

    def ensure_loaded(self) -> None:
        if self.lib is not None:
            return

        lib_path = _shared_lib_path()
        srcs = _sources()
        if _needs_rebuild(lib_path, srcs):
            _build_host_library(lib_path)

        lib = ctypes.CDLL(str(lib_path))
        lib.tinyml_host_dsp_init.argtypes = []
        lib.tinyml_host_dsp_init.restype = ctypes.c_int
        lib.tinyml_generate_spectrogram_int16.argtypes = [
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.tinyml_generate_spectrogram_int16.restype = ctypes.c_int
        lib.tinyml_generate_reference_multitone_int16.argtypes = [
            ctypes.POINTER(ctypes.c_int16),
        ]
        lib.tinyml_generate_reference_multitone_int16.restype = ctypes.c_int

        ret = lib.tinyml_host_dsp_init()
        if ret != 0:
            raise RuntimeError(f"tinyml_host_dsp_init failed: {ret}")
        self.lib = lib

    def extract(self, audio_float: np.ndarray) -> np.ndarray:
        self.ensure_loaded()
        if audio_float.ndim != 1:
            raise ValueError("audio_float must be 1-D")
        if audio_float.shape[0] != NUM_SAMPLES:
            raise ValueError(f"audio_float must have {NUM_SAMPLES} samples")

        audio_f32 = np.ascontiguousarray(audio_float.astype(np.float32))
        audio_i16 = np.clip(np.round(audio_f32 * 32768.0), -32768, 32767).astype(np.int16)
        out = np.zeros((SPECTROGRAM_SIZE,), dtype=np.float32)

        ret = self.lib.tinyml_generate_spectrogram_int16(
            audio_i16.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if ret != 0:
            raise RuntimeError(f"tinyml_generate_spectrogram_int16 failed: {ret}")
        return out.reshape(SPECTROGRAM_ROWS, NUM_MEL_BINS)

    def generate_reference_multitone_signal(self) -> np.ndarray:
        self.ensure_loaded()
        audio_i16 = np.zeros((NUM_SAMPLES,), dtype=np.int16)
        ret = self.lib.tinyml_generate_reference_multitone_int16(
            audio_i16.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        )
        if ret != 0:
            raise RuntimeError(f"tinyml_generate_reference_multitone_int16 failed: {ret}")
        return (audio_i16.astype(np.float32) / np.float32(32768.0)).astype(np.float32)


_HOST = _HostDSP()


def extract_logmel_cmsis(audio_float: np.ndarray, add_channel: bool = True) -> np.ndarray:
    spec = _HOST.extract(audio_float)
    if add_channel:
        return np.expand_dims(spec, axis=-1).astype(np.float32)
    return spec.astype(np.float32)


def generate_reference_multitone_signal() -> np.ndarray:
    return _HOST.generate_reference_multitone_signal()
