# Claims and Results Map

This file maps project claims to concrete files and generated results in the repository.

## 1) End-to-End ML Pipeline

Claim:
- Training -> quantization -> firmware deployment pipeline exists

Sources:
- `training/scripts/train_audio_model.py`
- `training/scripts/quantize.py`
- `firmware/src/inference.cpp`
- `firmware/src/model.cpp`

Status: Implemented

## 2) Quantization Drop (<2 pp)

Claim:
- Post-training int8 with <2 percentage-point accuracy drop

Records:
- `training/outputs/quantization_accuracy_summary.json`

Current values:
- FP32: `91.62380602498163%`
- INT8: `92.21160911094783%`
- Drop: `-0.5878030859662005 pp`
- Gate: pass (`< 2.0 pp`)

Status: Pass

## 3) Inference Optimization Claim

Claim:
- 32s -> sub-110ms range and ~300x class speedup, captured with sub-millisecond UART precision

Records:
- `benchmarks/results/latency_claim_validation.json`

Current values:
- Baseline mean: `33565.625 ms`
- Current mean: `108.0 ms`
- Speedup: `310.7928240740741x`
- Reduction: `99.67824224932501%`

Status: Pass

## 4) DSP Numerical Equivalence

Claims:
- Cross-platform DSP parity for the CMSIS FFT + matrix-projection frontend
- Bit-level parity in reference replay mode

Records:
- `training/outputs/numerical_equivalence_cross_impl_summary.json`

Current values:
- `bit_exact=true`
- `tolerance_pass=true`
- `max_abs_diff=0.0`

Status: Pass for captured reference workflow

## 5) Quantized Cross-Platform Inference Parity

Claim:
- Model behavior equivalence across Python and firmware for captured quantized tensors

Records:
- `training/outputs/quantized_cross_platform_summary.json`

Current values:
- `exact_int8_output_match=true`
- `behavior_equivalent=true`
- Max abs int8 diff: `0`
- Same argmax: yes

Status: Exact int8 parity pass

## 6) Dual-Core DSP Offload

Claim:
- CPUAPP/CPUNET split with IPC-based DSP offload

Records:
- `firmware/src/dual_core_ipc.c`
- `benchmarks/results/dual_core_runtime_summary.json`

Status: Implemented

## 7) Wireless Telemetry Stack

Claim:
- UDP telemetry over WiFi on nRF7002, including hotspot/DHCP deployment

Sources and records:
- `firmware/src/main.c`
- `firmware/prj.conf`
- `firmware/include/wifi_credentials.h`
- `benchmarks/verify_wifi_telemetry.py`
- `benchmarks/results/wifi_telemetry_summary.json`

Status: Implemented

Operational stability depends on hotspot/AP environment and runtime association conditions.

## 8) Constrained Inference Runtime

Claim:
- Tensor arena is statically allocated once at init, with constrained inference build RAM footprint <256KB.

Sources and records:
- `firmware/src/inference.cpp`
- `firmware/prj_inference_only.conf`
- `benchmarks/verify_runtime_constraints.py`
- `benchmarks/results/runtime_constraints_summary.json`

Status:
- Static inference allocation is verified from source and UART records.
- Total firmware RAM is verified only when `verify_runtime_constraints.py` is run against a constrained build ELF.

## 9) Power Characterization

Claim:
- PPK2-based energy characterization

Sources and records:
- `benchmarks/analyze_power.py`
- `firmware/src/main.c`
- `benchmarks/traces/README.md`
- `benchmarks/results/power_trace_manifest.json`
- `benchmarks/results/isolated_stage_power_summary.json`
- `docs/assets/ppk2-matched-runtime-traces.png`
- `docs/assets/ppk2-isolated-stage-traces.png`

Status:
- Tooling implemented
- Firmware emits per-frame `PHASE_TIMING` markers for DSP/NN attribution during UART + PPK2 capture
- Native USB CDC console profile is implemented for same-run PPK2 + timing capture without iMCU VCOM power coupling.
- Software-only isolated power modes are implemented for `DSP-only` and `NN-only` captures when any live USB logging path perturbs PPK2 measurement.
- Matched `AI-only` vs `AI + WiFi` 60 s trace summaries, visualizations, and raw CSV checksums are checked in.
- Isolated `DSP-only` vs `NN-only` 60 s trace summaries, visualizations, and raw CSV checksums are checked in and show a baseline-subtracted dynamic split of `46.51%` DSP vs `53.49%` NN.
- The full raw CSV exports are intentionally stored outside Git because each capture is about 166 MB.
