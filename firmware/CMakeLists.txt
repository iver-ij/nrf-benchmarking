cmake_minimum_required(VERSION 3.20.0)
# Work around legacy CMakeLists in transitive OpenAMP/libmetal modules.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(firmware)

set(TFLM_ROOT_CANDIDATES
    "${ZEPHYR_BASE}/../modules/tflite-micro"
    "${ZEPHYR_BASE}/../modules/lib/tflite-micro"
    "${ZEPHYR_BASE}/../optional/modules/lib/tflite-micro"
)
foreach(candidate IN LISTS TFLM_ROOT_CANDIDATES)
    if(EXISTS "${candidate}/tensorflow/lite/micro")
        set(TFLM_ROOT "${candidate}")
        break()
    endif()
endforeach()
if(NOT TFLM_ROOT)
    message(FATAL_ERROR "TensorFlow Lite Micro module not found. Checked: ${TFLM_ROOT_CANDIDATES}")
endif()
set(CMSIS_ROOT "${ZEPHYR_BASE}/../modules/lib/cmsis-dsp")
set(CMSIS_NN_ROOT "${ZEPHYR_BASE}/../modules/lib/cmsis-nn")
set(CMSIS_CORE "${ZEPHYR_BASE}/../modules/hal/cmsis")

target_include_directories(app PRIVATE
    include
    "${TFLM_ROOT}"
    "${TFLM_ROOT}/third_party_static/flatbuffers/include"
    "${TFLM_ROOT}/third_party_static/gemmlowp"
    "${TFLM_ROOT}/third_party_static/kissfft"
    "${TFLM_ROOT}/third_party_static/ruy"
    "${TFLM_ROOT}/third_party_static/ruy/include"
    "${CMSIS_ROOT}/Include"
    "${CMSIS_NN_ROOT}"
    "${CMSIS_NN_ROOT}/Include"
    "${CMSIS_CORE}/CMSIS/Core/Include"
    "${CMSIS_CORE}/CMSIS/DSP/Include"
)

file(GLOB_RECURSE TFLM_SOURCES
    "${TFLM_ROOT}/tensorflow/lite/micro/*.cc"
    "${TFLM_ROOT}/tensorflow/lite/micro/*.c"
    "${TFLM_ROOT}/tensorflow/lite/kernels/internal/*.cc"
    "${TFLM_ROOT}/tensorflow/lite/kernels/internal/*.c"
    "${TFLM_ROOT}/tensorflow/lite/core/*.cc"
    "${TFLM_ROOT}/tensorflow/lite/core/*.c"
    "${TFLM_ROOT}/tensorflow/lite/schema/*.cc"
)

set(TFLM_EXTRA_SOURCES
    "${TFLM_ROOT}/tensorflow/lite/kernels/kernel_util.cc"
)

set(TFLM_EXCLUDE_PATTERNS
    "test.cc$"
    "mock"
    "bench"
    "_test_"
    "integration_tests"
    "examples"
    "test_data_generation"
    "/tools/"
    "/python/"
    "/downloads/"
    "/ethos_u/"
    "/hexagon/"
    "/xtensa/"
    "/arc_"
    "/ceva/"
    "bluepill"
    "chre"
    "riscv"
    "apollo3"
    "stm32"
    "himax"
    "esp"
    "cortex_m_corstone_300"
    "cortex_m_generic"
)

foreach(pattern IN LISTS TFLM_EXCLUDE_PATTERNS)
    list(FILTER TFLM_SOURCES EXCLUDE REGEX "${pattern}")
endforeach()

# Use CMSIS-NN kernels where available and exclude duplicate generic kernels.
# For strict parity capture builds we may disable CONFIG_CMSIS_NN to force
# generic/reference TFLM kernels on firmware.
if(CONFIG_CMSIS_NN)
    set(TFLM_KERNELS_TO_REPLACE_WITH_CMSIS_NN
        add
        conv
        depthwise_conv
        fully_connected
        mul
        pooling
        svdf
        transpose_conv
        unidirectional_sequence_lstm
    )

    # Softmax replacement is conditional: keep generic softmax when
    # CONFIG_CMSIS_NN_SOFTMAX is disabled.
    if(CONFIG_CMSIS_NN_SOFTMAX)
        list(APPEND TFLM_KERNELS_TO_REPLACE_WITH_CMSIS_NN softmax)
    else()
        list(FILTER TFLM_SOURCES EXCLUDE REGEX "tensorflow/lite/micro/kernels/cmsis_nn/softmax\.cc$")
    endif()

    foreach(kernel IN LISTS TFLM_KERNELS_TO_REPLACE_WITH_CMSIS_NN)
        list(FILTER TFLM_SOURCES EXCLUDE REGEX "tensorflow/lite/micro/kernels/${kernel}\.cc$")
    endforeach()
else()
    # Reference path: keep generic kernels, drop CMSIS-NN kernel sources.
    list(FILTER TFLM_SOURCES EXCLUDE REGEX "tensorflow/lite/micro/kernels/cmsis_nn/.*\.cc$")
endif()

target_sources(app PRIVATE
    src/main.c
    src/dsp.c
    src/inference.cpp
    src/model.cpp
    src/mel_constants.c
    ${TFLM_SOURCES}
    ${TFLM_EXTRA_SOURCES}
)

function(tinyml_option_from_sysbuild option_name option_help option_default)
    # Resolve sysbuild-propagated values (global and image-local) before defining options.
    zephyr_get(_sysbuild_value VAR ${option_name})
    if(DEFINED _sysbuild_value AND NOT "${_sysbuild_value}" STREQUAL "")
        set(_resolved_default "${_sysbuild_value}")
    else()
        set(_resolved_default "${option_default}")
    endif()
    option(${option_name} "${option_help}" ${_resolved_default})
endfunction()

tinyml_option_from_sysbuild(
    ENABLE_DUAL_CORE_DSP
    "Enable dual-core DSP offload over IPC (cpuapp <-> cpunet)"
    OFF
)
# Sysbuild does not always forward image-local CMake options; when dual-core
# IPC transport is configured (OpenAMP enabled), compile the offload path too.
if(ENABLE_DUAL_CORE_DSP OR CONFIG_OPENAMP)
    target_sources(app PRIVATE src/dual_core_ipc.c)
    target_compile_definitions(app PRIVATE TINYML_DUAL_CORE_DSP_OFFLOAD)
endif()

tinyml_option_from_sysbuild(
    ENABLE_TEST_MODE
    "Enable test-mode sources without Kconfig symbol"
    OFF
)
if(CONFIG_TEST_MODE OR ENABLE_TEST_MODE)
    target_sources(app PRIVATE
        src/data.c
    )
endif()

if(CONFIG_TEST_MODE OR ENABLE_TEST_MODE)
    target_compile_definitions(app PRIVATE TEST_MODE)
endif()

tinyml_option_from_sysbuild(
    TINYML_INCLUDE_BENCHMARK_SOURCES
    "Compile benchmark/export helper sources into the firmware image"
    ON
)
if(TINYML_INCLUDE_BENCHMARK_SOURCES)
    target_sources(app PRIVATE
        src/benchmark.c
        src/test_signals.c
    )
endif()

tinyml_option_from_sysbuild(
    TINYML_EXPORT_C_REFERENCE
    "Build firmware in repeating C reference spectrogram export mode"
    OFF
)
if(TINYML_EXPORT_C_REFERENCE)
    target_compile_definitions(app PRIVATE EXPORT_C_REFERENCE_SPECTROGRAM)
endif()

tinyml_option_from_sysbuild(
    TINYML_EXPORT_QUANTIZED_REFERENCE
    "Build firmware in repeating quantized tensor export mode"
    OFF
)
if(TINYML_EXPORT_QUANTIZED_REFERENCE)
    target_compile_definitions(app PRIVATE EXPORT_QUANTIZED_REFERENCE)
endif()

tinyml_option_from_sysbuild(
    TINYML_WIFI_RUNTIME
    "Enable WiFi initialization at runtime"
    ON
)
tinyml_option_from_sysbuild(
    TINYML_TELEMETRY_SEND_STARTUP
    "Send WIFI_READY telemetry packet at startup"
    ON
)
tinyml_option_from_sysbuild(
    TINYML_TELEMETRY_SEND_TARGET
    "Send target-detection telemetry packets"
    ON
)
tinyml_option_from_sysbuild(
    TINYML_TELEMETRY_SEND_PER_FRAME
    "Send telemetry packet every inference frame"
    OFF
)
tinyml_option_from_sysbuild(
    TINYML_PRINT_STAGE_TIMINGS
    "Print per-frame DSP/NN/end-to-end timing markers"
    ON
)
tinyml_option_from_sysbuild(
    TINYML_POWER_DSP_ONLY
    "Run DSP frontend only in the production loop"
    OFF
)
tinyml_option_from_sysbuild(
    TINYML_POWER_NN_ONLY
    "Run NN inference only in the production loop using a cached spectrogram"
    OFF
)

if(TINYML_WIFI_RUNTIME)
    target_compile_definitions(app PRIVATE TINYML_WIFI_RUNTIME=1)
else()
    target_compile_definitions(app PRIVATE TINYML_WIFI_RUNTIME=0)
endif()

if(TINYML_TELEMETRY_SEND_STARTUP)
    target_compile_definitions(app PRIVATE TINYML_TELEMETRY_SEND_STARTUP=1)
else()
    target_compile_definitions(app PRIVATE TINYML_TELEMETRY_SEND_STARTUP=0)
endif()

if(TINYML_TELEMETRY_SEND_TARGET)
    target_compile_definitions(app PRIVATE TINYML_TELEMETRY_SEND_TARGET=1)
else()
    target_compile_definitions(app PRIVATE TINYML_TELEMETRY_SEND_TARGET=0)
endif()

if(TINYML_TELEMETRY_SEND_PER_FRAME)
    target_compile_definitions(app PRIVATE TINYML_TELEMETRY_SEND_PER_FRAME=1)
else()
    target_compile_definitions(app PRIVATE TINYML_TELEMETRY_SEND_PER_FRAME=0)
endif()

if(TINYML_PRINT_STAGE_TIMINGS)
    target_compile_definitions(app PRIVATE TINYML_PRINT_STAGE_TIMINGS=1)
else()
    target_compile_definitions(app PRIVATE TINYML_PRINT_STAGE_TIMINGS=0)
endif()

if(TINYML_POWER_DSP_ONLY AND TINYML_POWER_NN_ONLY)
    message(FATAL_ERROR "TINYML_POWER_DSP_ONLY and TINYML_POWER_NN_ONLY cannot both be enabled")
endif()

if(TINYML_POWER_DSP_ONLY)
    target_compile_definitions(app PRIVATE TINYML_POWER_DSP_ONLY=1)
else()
    target_compile_definitions(app PRIVATE TINYML_POWER_DSP_ONLY=0)
endif()

if(TINYML_POWER_NN_ONLY)
    target_compile_definitions(app PRIVATE TINYML_POWER_NN_ONLY=1)
else()
    target_compile_definitions(app PRIVATE TINYML_POWER_NN_ONLY=0)
endif()

target_compile_definitions(app PRIVATE TF_LITE_STATIC_MEMORY)
if(CONFIG_CMSIS_NN)
    target_compile_definitions(app PRIVATE CMSIS_NN)
endif()
