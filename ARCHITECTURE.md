# Architecture

This document describes the implemented architecture of the repository.

## 1) System Overview

The project is split into host-side tooling and target-side firmware.

- Host side (Python): training, quantization, summary validation, UART parsing, benchmark aggregation
- Target side (C/C++ on Zephyr): DSP frontend, TFLM inference, WiFi telemetry, optional dual-core DSP offload

High-level data flow:

```text
Signal (synthetic/test audio)
  -> frame segmentation (49 frames, 512 samples each, 20 ms hop)
  -> Hann window + RFFT + magnitude
  -> dense CMSIS matrix-vector mel projection (40 bins)
  -> log compression
  -> int8 quantized TFLM inference
  -> threshold decision
  -> optional UDP telemetry event
```

## 2) Firmware Runtime Modes

Controlled by CMake compile definitions in `firmware/CMakeLists.txt`:

- `TINYML_WIFI_RUNTIME` (ON/OFF)
- `TINYML_TELEMETRY_SEND_STARTUP` (ON/OFF)
- `TINYML_TELEMETRY_SEND_TARGET` (ON/OFF)
- `TINYML_TELEMETRY_SEND_PER_FRAME` (ON/OFF)
- `TINYML_INCLUDE_BENCHMARK_SOURCES` (ON/OFF)
- `TEST_MODE` (enabled with `CONFIG_TEST_MODE=y`)
- `TINYML_DUAL_CORE_DSP_OFFLOAD` (enabled by dual-core build options)

Runtime paths in `firmware/src/main.c`:

- Reference export mode:
  - `EXPORT_C_REFERENCE_SPECTROGRAM` or `EXPORT_QUANTIZED_REFERENCE`
- Test mode:
  - runs benchmarks then canned validation loop
- Production mode:
  - deterministic synthetic 1 kHz tone loop (self-generated input source)
  - emits `PHASE_TIMING dsp_us=... nn_us=... total_us=...` markers for stage-aware UART/PPK2 analysis

## 3) DSP Frontend

Files:
- `firmware/src/dsp.c`
- `firmware/include/data.h`
- `firmware/include/mel_constants.h`
- `firmware/src/mel_constants.c`
- `training/scripts/gen_mel_header.py`

Parameters:
- Sample rate: 16 kHz
- Clip length: 1 second (`NUM_SAMPLES=16000`)
- FFT size: 512
- FFT bins: 257
- Mel bins: 40
- Frames: 49 (`WINDOW_STEP=320`)

Mel projection implementation:
- FFT magnitudes are multiplied by a generated dense `[40 x 257]` mel matrix.
- The runtime uses `arm_mat_vec_mult_f32()` for the mel stage.
- The dense constants are generated from TensorFlow's mel-weight matrix so host and firmware share the same coefficients.

## 4) Inference Runtime

Files:
- `firmware/src/inference.cpp`
- `firmware/include/inference.h`
- `firmware/src/model.cpp`

Key points:
- Tensor arena is static allocation (deterministic startup behavior)
- A constrained verification profile exists in `firmware/prj_inference_only.conf`
- Inference API:
  - `int inference_setup(void)`
  - `int run_inference(float* spectrogram_data, float* output_probability)`
- Model input/output are int8 TFLM tensors
- Output confidence used for threshold detection in `main.c`

## 5) Optional Dual-Core DSP Offload

CPUAPP side:
- `firmware/src/dual_core_ipc.c`

CPUNET side service:
- DSP worker image for offloaded frame processing

IPC protocol:
- `firmware/include/dual_core_protocol.h`
- Endpoint name: `tinyml_dsp_ep`
- Commands:
  - `FRAME_START`
  - `FRAME_CHUNK`
  - `FRAME_END`
  - `FRAME_RESULT`

Sysbuild wiring:
- `firmware/Kconfig.sysbuild`
- `firmware/sysbuild.cmake`
- `firmware/prj_dualcore.conf`

## 6) WiFi + UDP Telemetry

Files:
- `firmware/src/main.c`
- `firmware/prj.conf`
- `firmware/include/wifi_credentials.h`

Implemented logic:
- WiFi ready callback wait
- Stored credential connect attempt
- Explicit credential fallback (PSK/2.4GHz then WPA_AUTO/ANY)
- Reconnect thread for deferred association retry
- UDP socket setup with `connect()` + `send()`
- Runtime telemetry gating based on connection and socket readiness

Tracked defaults:
- WiFi credentials: placeholder values in `firmware/prj.conf`, intended for local override
- IPv4 assignment: DHCP
- Telemetry target: placeholder destination in `firmware/include/wifi_credentials.h`, intended for local override

## 7) Benchmarks and Result Pipelines

Firmware-side benchmark code:
- `firmware/src/benchmark.c`

Host-side scripts:
- `benchmarks/capture_inference_uart.py`
- `benchmarks/validate_latency_claim.py`
- `benchmarks/analyze_power.py`

Training/validation scripts:
- `training/scripts/evaluate_quantization.py`
- `training/scripts/verify_numerical_equivalence.py`
- `training/scripts/run_bit_exactness_pipeline.py`
- `training/scripts/run_quantized_parity_pipeline.py`

## 8) Results-Driven Validation Model

The repo treats generated outputs as validation records for claims.

Primary result files:
- Quantization accuracy summary JSON
- Latency claim validation JSON
- Numerical equivalence JSON (DSP)
- Quantized cross-platform summary JSON
- Dual-core runtime summary JSON

This keeps claim verification reproducible and scriptable.
Retained UART logs, traces, and benchmark result JSON files serve as historical hardware records.

## 9) Practical Boundaries

- Default production loop uses a synthetic deterministic signal source by design.
- WiFi stability depends on environment/AP behavior and board/host state.
- Power characterization uses exported CSV analysis; quality depends on capture protocol correctness.
- Strict same-run DSP/NN power attribution uses the native USB CDC console profile rather than the iMCU VCOM path.
