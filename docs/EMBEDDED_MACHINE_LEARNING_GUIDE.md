# Technical Walkthrough of the nRF Embedded ML Benchmark Pipeline

This repository implements a complete host-to-target embedded ML benchmark on Nordic nRF5340/nRF7002DK hardware. The current workload is a compact three-class spectral classifier: the host side trains it, quantizes it to int8, exports it into firmware, and validates it against the deployed target. The target side reconstructs the spectral frontend in C with CMSIS-DSP, runs the quantized model with TensorFlow Lite Micro and CMSIS-NN, emits timing markers for profiling, and can transmit runtime state over WiFi by UDP.

The runtime input path is deterministic and reproducible: synthetic digital input in the default production loop, or canned fixtures in `TEST_MODE`. That choice keeps the implementation focused on spectral preprocessing, constrained inference, numerical verification, runtime behavior, and measurement.

## 1. System Boundary

The implemented system has four major layers.

1. Host-side training and export in `training/`
2. Firmware build and runtime in `firmware/`
3. Validation and measurement tooling in `benchmarks/`
4. Tracked outputs in `training/outputs/` and `benchmarks/results/`

The pipeline can be summarized as:

1. Resolve dataset paths and load one-second audio clips
2. Convert clips into `49 x 40` log-mel features on the host
3. Train a compact CNN on the host
4. Quantize the model to int8 and export it as both `.tflite` and C++ source
5. Generate dense mel filterbank constants for firmware
6. Recompute the same frontend on the target with CMSIS-DSP
7. Quantize firmware features into the model input tensor and run TFLM inference
8. Measure latency, RAM usage, power, and cross-platform parity

## 2. Repository Structure

The implementation is concentrated in the following paths.

- `training/scripts/train_audio_model.py`
  - host-side dataset assembly, feature extraction, training, and model save
- `training/scripts/quantize.py`
  - post-training int8 conversion and C++ model export
- `training/scripts/gen_mel_header.py`
  - generation of dense mel filterbank constants for firmware
- `training/scripts/run_bit_exactness_pipeline.py`
  - host-to-firmware DSP equivalence pipeline
- `training/scripts/run_quantized_parity_pipeline.py`
  - host-to-firmware quantized inference parity pipeline
- `firmware/src/dsp.c`
  - target-side spectral frontend
- `firmware/src/inference.cpp`
  - TFLM runtime, tensor arena, tensor quantization, output dequantization
- `firmware/src/main.c`
  - application sequencing, timing markers, thresholding, WiFi, telemetry, power isolation modes
- `firmware/src/dual_core_ipc.c`
  - optional CPUAPP to CPUNET offload transport for DSP frames
- CPUNET DSP worker image
- `benchmarks/capture_inference_uart.py`
  - UART capture of inference and stage timing markers
- `benchmarks/analyze_power.py`
  - offline PPK2 CSV analysis and event extraction

## 3. Host-Side Training Pipeline

### 3.1 Dataset Resolution and Label Scheme

Dataset lookup is centralized in `training/scripts/dataset_paths.py`. The actual download path can be resolved into a cache-backed directory rather than being committed into the repository. `training/scripts/fetch_speech_commands_dataset.py` exists to populate that path when needed.

The current training task is a three-class classifier:

- class `0`: silence or background
- class `1`: unknown word
- class `2`: target class `"on"`

The training script is `training/scripts/train_audio_model.py`. The important fixed parameters are:

- sample rate: `16000 Hz`
- clip length: `16000` samples, exactly one second
- FFT size: `512`
- frame step: `320`
- mel bins: `40`
- input shape: `(49, 40, 1)`
- batch size: `64`
- epochs: `30`
- seed: `42`

The script balances the training set with explicit caps:

- unknown-to-target ratio: `2.0`
- silence-to-target ratio: `1.0`

That produces a controlled three-class problem rather than an unconstrained full-dataset classifier.

### 3.2 Host Feature Extraction

The host pipeline computes the same representation that the firmware will later compute:

1. load one second of audio
2. convert to mono float32 in `[-1, 1]`
3. apply STFT with `frame_length=512`, `frame_step=320`, `fft_length=512`
4. take magnitude
5. project to a `40`-bin mel scale
6. apply `log(mel + 1e-6)`
7. add a channel dimension to obtain `(49, 40, 1)`

There are two host implementations:

- TensorFlow reference path inside `train_audio_model.py`
- CMSIS-compatible path through `training/scripts/cmsis_host_dsp.py`

The default host backend is controlled by `DSP_BACKEND`, and the repository uses the CMSIS-compatible host path by default so the training/export side and the firmware side stay aligned.

### 3.3 Training Model Architecture

The trained network is a compact convolutional classifier defined inline in `train_audio_model.py`:

1. `Conv2D(16, 3x3, stride 2x2, relu, same)`
2. `BatchNormalization`
3. `DepthwiseConv2D(3x3, relu, same)`
4. `Conv2D(32, 1x1, relu)`
5. `MaxPooling2D(2x2)`
6. `DepthwiseConv2D(3x3, relu, same)`
7. `Conv2D(64, 1x1, relu)`
8. `GlobalAveragePooling2D`
9. `Dense(64, relu)`
10. `Dropout(0.2)`
11. `Dense(3, softmax)`

Training uses:

- optimizer: `adam`
- loss: `sparse_categorical_crossentropy`
- metric: `accuracy`
- callbacks:
  - `ReduceLROnPlateau`
  - `EarlyStopping(restore_best_weights=True)`

The resulting model is stored as `training/outputs/model.keras`.

## 4. Quantization and Model Export

Post-training quantization is implemented in `training/scripts/quantize.py`.

Important implementation details:

- the script exports a SavedModel from the Keras model for Keras 3 compatibility
- the converter uses `tf.lite.Optimize.DEFAULT`
- supported ops are restricted to `TFLITE_BUILTINS_INT8`
- both inference input and output types are forced to `int8`
- representative calibration data is class-balanced and deterministic

The current representative dataset configuration is:

- target samples: `450`
- unknown samples: `450`
- silence samples: `100`

The quantizer writes:

- `training/outputs/model.tflite`
- `firmware/src/model.cpp`

`firmware/src/model.cpp` is generated by serializing the `.tflite` flatbuffer into a C++ byte array named `g_model`. That is the model blob compiled into the firmware image.

Quantization quality is evaluated by `training/scripts/evaluate_quantization.py`, which compares:

- host FP32 Keras inference
- host INT8 TFLite inference

The current tracked summary is `training/outputs/quantization_accuracy_summary.json`.

## 5. Generated DSP Constants

The firmware does not build the mel filterbank at runtime. `training/scripts/gen_mel_header.py` generates a dense float32 matrix and emits:

- `firmware/include/mel_constants.h`
- `firmware/src/mel_constants.c`

The generator uses TensorFlow to build the mel weight matrix, then transposes it from TensorFlow layout `[fft_bin][mel_bin]` into runtime layout `[mel_bin][fft_bin]`. The firmware stores this dense matrix so it can call `arm_mat_vec_mult_f32()` directly during feature generation.

The runtime matrix dimensions are:

- rows: `40` mel bins
- columns: `257` FFT magnitude bins
- total dense elements: `40 * 257 = 10280`

This is a deliberate flash-for-simplicity tradeoff. The dense representation makes the runtime projection simple and deterministic.

## 6. Firmware Build System and Compile-Time Modes

The firmware build is driven by `firmware/CMakeLists.txt`. Run `west` from an active nRF Connect SDK toolchain shell and set `ZEPHYR_BASE` to the SDK checkout, for example `export ZEPHYR_BASE="$HOME/ncs/v2.9.3/zephyr"` followed by `nrfutil toolchain-manager launch --ncs-version v2.9.3 --shell`, so the bundled Python, CMake, SDK, and linker paths are all present.

The build integrates:

- Zephyr
- TensorFlow Lite Micro
- CMSIS-DSP
- CMSIS-NN
- CMSIS Core headers

### 6.1 TFLM Source Selection

`CMakeLists.txt` collects TFLM sources recursively and filters out:

- tests
- mocks
- examples
- tools
- non-target architectures

When `CONFIG_CMSIS_NN=y`, generic TFLM kernels are excluded in favor of CMSIS-NN kernels for supported operators such as:

- `conv`
- `depthwise_conv`
- `fully_connected`
- `pooling`
- `mul`
- `add`
- `svdf`
- `transpose_conv`
- `unidirectional_sequence_lstm`

If `CONFIG_CMSIS_NN_SOFTMAX` is disabled, generic softmax is retained.

### 6.2 Build-Time Feature Flags

The main compile-time switches are exposed as CMake options and then translated into compile definitions:

- `ENABLE_DUAL_CORE_DSP`
- `ENABLE_TEST_MODE`
- `TINYML_INCLUDE_BENCHMARK_SOURCES`
- `TINYML_EXPORT_C_REFERENCE`
- `TINYML_EXPORT_QUANTIZED_REFERENCE`
- `TINYML_WIFI_RUNTIME`
- `TINYML_TELEMETRY_SEND_STARTUP`
- `TINYML_TELEMETRY_SEND_TARGET`
- `TINYML_TELEMETRY_SEND_PER_FRAME`
- `TINYML_PRINT_STAGE_TIMINGS`
- `TINYML_POWER_DSP_ONLY`
- `TINYML_POWER_NN_ONLY`

`TINYML_POWER_DSP_ONLY` and `TINYML_POWER_NN_ONLY` are mutually exclusive. The build fails if both are enabled.

### 6.3 Firmware Configurations

The repository keeps several configuration overlays for different tasks:

- `firmware/prj.conf`
  - base image, WiFi-enabled runtime
- `firmware/prj_inference_only.conf`
  - constrained inference and power-focused build
- `firmware/prj_dualcore.conf`
  - dual-core/OpenAMP build
- `firmware/prj_c_ref.conf`
  - C-reference spectrogram export
- `firmware/prj_qref.conf`
  - quantized tensor export
- `firmware/prj_usb_console.conf`
  - USB CDC console path for logging-oriented builds
- `firmware/usb_console.overlay`
  - devicetree overlay for USB console selection

## 7. Target-Side Spectral Frontend

The target-side frontend is implemented in `firmware/src/dsp.c`.

### 7.1 Buffers and Alignment

The DSP path uses statically allocated float buffers aligned to 16 bytes:

- `fft_input_buffer[FFT_SIZE]`
- `fft_output_buffer[FFT_SIZE]`
- `fft_magnitude_buffer[NUM_FFT_BINS]`
- `window_func[FFT_SIZE]`

The alignment is intentional. The frontend is designed for predictable access patterns and CMSIS-DSP execution, not dynamic allocation.

### 7.2 Initialization

`dsp_init()` performs one-time initialization:

- `arm_rfft_fast_init_f32(&fft_instance, FFT_SIZE)`
- matrix shape setup for the dense mel filterbank
- Hann window generation using `arm_cos_f32`

The mel filterbank matrix is bound directly to the generated `mel_filterbank_dense` data from `mel_constants.c`.

### 7.3 Feature Generation

`generate_spectrogram()` performs the full frontend:

1. iterate over `SPECTROGRAM_ROWS`
2. copy a frame from the int16 input buffer
3. normalize by `32768.0f`
4. apply the Hann window
5. run `arm_rfft_fast_f32`
6. reconstruct magnitude bins:
   - DC from `fft_output_buffer[0]`
   - Nyquist from `fft_output_buffer[1]`
   - remaining complex magnitudes with `arm_cmplx_mag_f32`
7. compute mel bins with `arm_mat_vec_mult_f32`
8. apply `logf(mel + 1e-6f)`
9. canonicalize the float bits before storing

The canonicalization step masks off lower float bits:

- `LOGMEL_CANON_MASK = 0xFFFF8000u`

That step reduces cross-platform drift in stored log-mel values and supports the numerical equivalence pipeline.

The output tensor shape is `49 x 40`, flattened into `SPECTROGRAM_SIZE`.

## 8. Inference Runtime

The model runtime is implemented in `firmware/src/inference.cpp`.

### 8.1 Tensor Arena

The TFLM allocator is a statically allocated tensor arena:

- `kTensorArenaSize = 50 * 1024`
- `static uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)))`

This avoids dynamic allocation during inference and fixes the memory layout at compile time.

### 8.2 Operator Registration

The runtime registers only the operators required by the model through `MicroMutableOpResolver<12>`:

- `Conv2D`
- `DepthwiseConv2D`
- `MaxPool2D`
- `FullyConnected`
- `Softmax`
- `Reshape`
- `AveragePool2D`
- `Mul`
- `Add`
- `Mean`
- `Quantize`
- `Dequantize`

This is a memory-reduction choice. The runtime does not register unused kernels.

### 8.3 Input and Output Handling

`inference_setup()`:

- reads the compiled-in model from `g_model`
- verifies the TFLite schema version
- allocates tensors
- records the actual arena usage
- retrieves input and output tensor handles
- requires both tensors to be `kTfLiteInt8`

`run_inference()`:

1. quantizes the float32 spectrogram into the input int8 tensor using tensor scale and zero-point
2. invokes the interpreter
3. copies quantized input and output tensors into static buffers for parity tooling
4. dequantizes output index `2`

Class index `2` is the target class and is the probability used in the application logic.

The runtime also exposes helper functions for:

- tensor arena size
- input/output tensor byte counts
- last quantized input copy
- last quantized output copy

Those functions are consumed by validation tooling.

## 9. Application Runtime

The application orchestration is implemented in `firmware/src/main.c`.

### 9.1 Initialization Sequence

`main()` performs the following sequence:

1. enable nRF cache with `nrf_cache_enable`
2. initialize console transport
3. configure GPIO
4. print the firmware banner and configuration summary
5. initialize TFLM inference
6. optionally prepare cached features for `NN-only` mode
7. optionally initialize dual-core DSP IPC
8. initialize WiFi after inference and IPC setup
9. enter the main production or test loop

The WiFi initialization is intentionally delayed until after inference setup so network association cannot block deterministic inference startup.

### 9.2 Production and Test Modes

The default production loop is deterministic:

- it synthesizes a `1000 Hz` tone at amplitude `0.5`
- processes one one-second frame
- sleeps for one second

`ENABLE_TEST_MODE` defines `TEST_MODE` and replaces that path with a validation loop that iterates through canned fixtures:

- silence
- interference sample
- target sample

### 9.3 `process_audio()`

`process_audio()` is the central runtime path.

When DSP is active, it:

1. fills or copies the input audio buffer
2. invalidates cache for CPU access if needed
3. computes the spectrogram locally or through dual-core offload
4. flushes the feature buffer for potential device-side consumers
5. computes the peak feature magnitude
6. skips inference if the peak is below `SIGNAL_NOISE_FLOOR = 2.0f`

When the signal exceeds the noise floor, it:

1. runs inference
2. measures DSP, NN, and end-to-end durations
3. prints `PHASE_TIMING dsp_us=... nn_us=... total_us=...`
4. prints the inference duration and confidence
5. compares the target probability to `CONFIDENCE_THRESHOLD = 0.75f`
6. drives the target LED and optional telemetry

The timing markers are consumed by `benchmarks/capture_inference_uart.py`.

### 9.4 Power Isolation Modes

Two compile-time power modes exist in `main.c`:

- `TINYML_POWER_DSP_ONLY`
  - run the frontend and return before inference
- `TINYML_POWER_NN_ONLY`
  - precompute one cached spectrogram and repeatedly run only inference

These modes were added so DSP and NN stage power could be measured separately with PPK2 without relying on same-run UART phase alignment.

## 10. WiFi and UDP Telemetry

The networking path also lives in `firmware/src/main.c`, with destination parameters in `firmware/include/wifi_credentials.h`.

### 10.1 Enable Conditions

Telemetry is compiled only when both of the following Zephyr capabilities are present:

- `CONFIG_WIFI`
- `CONFIG_NET_SOCKETS`

The application-level runtime switch is `TINYML_WIFI_RUNTIME`.

### 10.2 Connection Strategy

The connection logic uses:

- a WiFi-ready wait path
- an optional pre-connect scan
- stored explicit credentials from `CONFIG_WIFI_CREDENTIALS_STATIC_*`
- a reconnect thread driven by semaphores
- socket refresh on transient send failures

The current repository state uses DHCP hotspot deployment, not static IP configuration. Credentials are injected through Zephyr configuration, and the telemetry destination is fixed in `wifi_credentials.h`.

### 10.3 Telemetry Semantics

The implemented event messages are:

- `WIFI_READY`
  - sent on startup or reconnect, with retry burst logic
- `TARGET_DETECTED`
  - sent only when the target probability exceeds `0.75`
- `INFERENCE_FRAME`
  - optional per-frame event, disabled by default

The code does not send a packet for every correct inference. It sends only the configured runtime events.

## 11. Optional Dual-Core DSP Offload

The optional offload path splits DSP across the two nRF5340 cores.

### 11.1 CPUAPP Side

`firmware/src/dual_core_ipc.c`:

- opens the `ipc0` instance
- registers an endpoint
- sends each `512`-sample frame as:
  - `FRAME_START`
  - one or more `FRAME_CHUNK` packets
  - `FRAME_END`
- waits for a `FRAME_RESULT` payload containing one `40`-value mel frame

The application core assembles the full `49 x 40` spectrogram from these returned frame results.

### 11.2 CPUNET Side

CPUNET worker:

- initializes the same FFT and dense mel matrix backend
- reconstructs frame samples from incoming IPC chunks
- computes the log-mel frame
- returns the result to CPUAPP

### 11.3 Runtime Fallback

`main.c` wraps the offload path in `compute_spectrogram_with_runtime_fallback()`.

If dual-core DSP returns an error, the firmware prints a warning and falls back to local DSP on CPUAPP. The offload path is therefore optional at runtime even when compiled in.

## 12. Validation and Measurement Tooling

### 12.1 DSP Equivalence

`training/scripts/run_bit_exactness_pipeline.py` and `training/scripts/verify_numerical_equivalence.py` validate the spectral frontend across Python and firmware.

The tracked summary is:

- `training/outputs/numerical_equivalence_cross_impl_summary.json`

This summary stores:

- feature-generation configuration
- mel filterbank sanity checks
- exact and tolerance-based equivalence metrics

### 12.2 Quantized Cross-Platform Parity

`training/scripts/run_quantized_parity_pipeline.py` and `training/scripts/verify_quantized_cross_platform.py` validate the quantized inference path using identical int8 inputs on host and target.

The tracked summary is:

- `training/outputs/quantized_cross_platform_summary.json`

This summary checks:

- exact int8 output match
- argmax equivalence
- threshold-decision equivalence
- bounded quantized drift acceptance rules

### 12.3 Latency Capture

`benchmarks/capture_inference_uart.py` reads the UART console, extracts:

- `Inference: ...`
- `PHASE_TIMING dsp_us=... nn_us=... total_us=...`

and writes both a raw log and a JSON summary. `benchmarks/validate_latency_claim.py` turns those results into the tracked latency summary file.

### 12.4 Runtime Constraints

`benchmarks/verify_runtime_constraints.py` verifies:

- static tensor arena declaration
- absence of dynamic allocation in the inference path
- runtime-logged tensor arena usage
- final ELF `data + bss` RAM size

### 12.5 Power Analysis

`benchmarks/analyze_power.py` operates offline on PPK2 CSV exports.
`benchmarks/generate_power_summaries.py` converts retained oversized CSV exports into tracked checksum and SVG summaries.

It supports:

- whole-trace average current and power
- event extraction using:
  - `burst`
  - `steady_inference`
- optional phase timing alignment when logs are available

The repository now uses two power analysis strategies:

1. matched whole-system traces:
   - `AI-only`
   - `AI + WiFi`
2. isolated stage traces:
   - `DSP-only`
   - `NN-only`

The isolated stage method is the repository's current mechanism for estimating DSP-vs-NN energy distribution without requiring same-run UART logging during PPK2 capture.

## 13. Current Tracked Results

The tracked outputs currently support the following technical state.

### 13.1 Quantization

Source:

- `training/outputs/quantization_accuracy_summary.json`

Current values:

- FP32 accuracy: `91.62380602498163%`
- INT8 accuracy: `92.21160911094783%`
- accuracy delta: `-0.5878030859662005` percentage points
- pass threshold: `< 2.0` percentage points

### 13.2 Latency

Source:

- `benchmarks/results/latency_claim_validation.json`

Current values:

- baseline mean: `33565.625 ms`
- current mean: `108.0 ms`
- speedup: `310.7928240740741x`
- reduction: `99.67824224932501%`

### 13.3 Runtime Memory

Source:

- `benchmarks/results/runtime_constraints_summary.json`

Current values:

- tensor arena: `51200` bytes
- runtime-logged arena used: `27204` bytes
- total firmware RAM (`data + bss`): `146434` bytes
- total RAM pass threshold: `< 256 KB`

### 13.4 DSP Equivalence

Source:

- `training/outputs/numerical_equivalence_cross_impl_summary.json`

Current values:

- `bit_exact = true`
- `tolerance_pass = true`
- `max_abs_diff = 0.0`
- `max_rel_diff = 0.0`

### 13.5 Quantized Parity

Source:

- `training/outputs/quantized_cross_platform_summary.json`

Current values:

- `exact_int8_output_match = true`
- `behavior_equivalent = true`
- `threshold_decision_equivalent = true`
- `max_abs_diff_int8 = 0`

### 13.6 Whole-System Power

Source:

- `benchmarks/results/power_comparison_summary.md`
- `benchmarks/results/power_trace_manifest.json`
- `docs/assets/ppk2-matched-runtime-traces.png`

Current values from matched `60 s` PPK2 captures at `3.6 V`:

- AI-only mean current: `21.606 mA`
- AI + WiFi mean current: `22.675 mA`
- WiFi overhead: `+1.069 mA`
- WiFi overhead: `+4.95%`
- WiFi overhead power: `+3.848 mW`

### 13.7 Isolated DSP and NN Power

Source:

- `benchmarks/results/isolated_stage_power_summary.md`
- `benchmarks/results/power_trace_manifest.json`
- `docs/assets/ppk2-isolated-stage-traces.png`

Current values from separate `60 s` stage-isolated captures at `3.6 V`:

- DSP dynamic current: `2.935 mA`
- NN dynamic current: `3.375 mA`
- DSP share: `46.51%`
- NN share: `53.49%`

## 14. Recommended Reading Order

For a direct technical reading of the repository, the shortest useful path is:

1. `README.md`
2. `ARCHITECTURE.md`
3. `firmware/CMakeLists.txt`
4. `firmware/src/dsp.c`
5. `firmware/src/inference.cpp`
6. `firmware/src/main.c`
7. `training/scripts/train_audio_model.py`
8. `training/scripts/quantize.py`
9. `training/scripts/gen_mel_header.py`
10. `training/scripts/run_bit_exactness_pipeline.py`
11. `training/scripts/run_quantized_parity_pipeline.py`
12. `benchmarks/capture_inference_uart.py`
13. `benchmarks/analyze_power.py`
14. `docs/CLAIMS_AND_RESULTS.md`

That order follows the actual dependency structure of the project:

- what is built
- how the frontend works
- how inference is executed
- how host outputs are generated
- how the deployed behavior is measured and verified
