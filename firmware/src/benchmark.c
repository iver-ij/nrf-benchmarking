#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <stdint.h>
#include "dsp.h"
#include "inference.h"
#include "data.h"
#include "test_signals.h"
#include "reference_signal.h"

#define WARMUP_RUNS 10
#define BENCHMARK_RUNS 100

#ifndef BENCHMARK_BASELINE_MS
#define BENCHMARK_BASELINE_MS 0
#endif

static int16_t test_audio[NUM_SAMPLES];
static float features[SPECTROGRAM_SIZE];
#ifdef EXPORT_QUANTIZED_REFERENCE
static int8_t qref_input[SPECTROGRAM_SIZE];
static int8_t qref_output[16];
#endif

enum benchmark_signal_kind {
    SIGNAL_SINE,
    SIGNAL_WHITE_NOISE,
    SIGNAL_CHIRP,
    SIGNAL_SILENCE,
};

struct benchmark_signal_case {
    const char* name;
    enum benchmark_signal_kind kind;
    float freq1;
    float freq2;
    float amplitude;
};

static void fill_benchmark_signal(const struct benchmark_signal_case *test)
{
    switch (test->kind) {
    case SIGNAL_SINE:
        generate_sine_wave(test_audio, test->freq1, test->amplitude);
        break;
    case SIGNAL_WHITE_NOISE:
        generate_white_noise(test_audio, test->amplitude);
        break;
    case SIGNAL_CHIRP:
        generate_chirp(test_audio, test->freq1, test->freq2, test->amplitude);
        break;
    case SIGNAL_SILENCE:
        generate_silence(test_audio);
        break;
    }
}

static uint32_t measure_spectrogram_us(void)
{
    for (int i = 0; i < WARMUP_RUNS; i++) {
        generate_spectrogram(test_audio, features);
    }

    uint32_t start = k_cycle_get_32();
    for (int i = 0; i < BENCHMARK_RUNS; i++) {
        generate_spectrogram(test_audio, features);
    }
    uint32_t end = k_cycle_get_32();

    return k_cyc_to_us_near32((end - start) / BENCHMARK_RUNS);
}

static double rate_per_second(uint32_t avg_us)
{
    return avg_us > 0U ? 1000000.0 / (double)avg_us : 0.0;
}

#ifdef EXPORT_C_REFERENCE_SPECTROGRAM
static void dump_c_reference_spectrogram(void) {
    generate_bit_exact_reference_signal(test_audio);

    if (generate_spectrogram(test_audio, features) != 0) {
        printk("C_REF_ERROR spectrogram_generation_failed\n");
        return;
    }

    printk("C_REF_BEGIN rows=%d cols=%d\n", SPECTROGRAM_ROWS, NUM_MEL_BINS);
    for (int i = 0; i < SPECTROGRAM_SIZE; i++) {
        union {
            float f;
            uint32_t u;
        } bits = { .f = features[i] };
        printk("C_REF %d %.9f 0x%08x\n", i, (double)features[i], bits.u);
        // Throttle UART output to avoid dropped log lines that can
        // truncate the reference block (especially C_REF_BEGIN).
        if ((i & 0x0F) == 0x0F) {
            k_msleep(1);
        }
    }
    printk("C_REF_END\n");
}
#endif

static void benchmark_dsp_variety(void) {
    printk("\n=== DSP Benchmark (Various Signals) ===\n");

    const struct benchmark_signal_case test_signals[] = {
        {"Sine 1kHz", SIGNAL_SINE, 1000.0f, 0.0f, 0.5f},
        {"Sine 3kHz", SIGNAL_SINE, 3000.0f, 0.0f, 0.5f},
        {"White Noise", SIGNAL_WHITE_NOISE, 0.0f, 0.0f, 0.5f},
        {"Chirp 100-4k", SIGNAL_CHIRP, 100.0f, 4000.0f, 0.5f},
        {"Silence", SIGNAL_SILENCE, 0.0f, 0.0f, 0.0f},
    };

    const int num_tests = sizeof(test_signals) / sizeof(test_signals[0]);

    for (int t = 0; t < num_tests; t++) {
        fill_benchmark_signal(&test_signals[t]);
        uint32_t avg_us = measure_spectrogram_us();

        printk("  %-15s: %5u us (%.3f ms)\n",
               test_signals[t].name, avg_us, (double)((float)avg_us / 1000.0f));
    }
}

void benchmark_dsp(void) {
    printk("\n=== DSP Benchmark ===\n");

    float frequencies[] = {500.0f, 1500.0f, 3000.0f};
    generate_multi_tone(test_audio, frequencies, 3, 0.3f);

    for (int i = 0; i < WARMUP_RUNS; i++) {
        generate_spectrogram(test_audio, features);
    }

    uint32_t start = k_cycle_get_32();
    for (int i = 0; i < BENCHMARK_RUNS; i++) {
        generate_spectrogram(test_audio, features);
    }
    uint32_t end = k_cycle_get_32();

    uint32_t total_cycles = end - start;
    uint32_t avg_us = k_cyc_to_us_near32(total_cycles / BENCHMARK_RUNS);

    printk("DSP Frontend (Mel Spectrogram):\n");
    printk("  Runs: %d\n", BENCHMARK_RUNS);
    printk("  Total: %u cycles\n", total_cycles);
    printk("  Average: %u us (%.3f ms)\n", avg_us, (double)((float)avg_us / 1000.0f));
    printk("  Throughput: %.1f fps\n", rate_per_second(avg_us));

    printk("  Per-frame (49 FFTs):\n");
    printk("    Total per frame: %u us\n", avg_us);
    printk("    Per FFT: ~%u us\n", avg_us / 49);
}

void benchmark_inference(void) {
    printk("\n=== Inference Benchmark ===\n");

    for (int i = 0; i < SPECTROGRAM_SIZE; i++) {
        features[i] = 0.0f;
    }

    float probability;

    for (int i = 0; i < WARMUP_RUNS; i++) {
        run_inference(features, &probability);
    }

    uint32_t start = k_cycle_get_32();
    for (int i = 0; i < BENCHMARK_RUNS; i++) {
        run_inference(features, &probability);
    }
    uint32_t end = k_cycle_get_32();

    uint32_t total_cycles = end - start;
    uint32_t avg_us = k_cyc_to_us_near32(total_cycles / BENCHMARK_RUNS);
    float avg_ms = (float)avg_us / 1000.0f;

    printk("Neural Network Inference:\n");
    printk("  Runs: %d\n", BENCHMARK_RUNS);
    printk("  Total: %u cycles\n", total_cycles);
    printk("  Average: %u us (%.3f ms)\n", avg_us, (double)avg_ms);
    printk("  Throughput: %.1f inferences/sec\n", rate_per_second(avg_us));

    printk("\nOptimization Summary:\n");
    printk("  Current: ~%.3f ms\n", (double)avg_ms);
    if (BENCHMARK_BASELINE_MS > 0 && avg_ms > 0.0f) {
        double speedup = (double)BENCHMARK_BASELINE_MS / (double)avg_ms;
        double reduction_percent =
            (1.0 - ((double)avg_ms / (double)BENCHMARK_BASELINE_MS)) * 100.0;

        printk("  Baseline (configured): ~%d ms\n", BENCHMARK_BASELINE_MS);
        printk("  Speedup: %.1fx\n", speedup);
        printk("  Reduction: %.2f%%\n", reduction_percent);
    } else {
        printk("  Baseline: not configured (set BENCHMARK_BASELINE_MS to compare)\n");
    }
}

void benchmark_end_to_end(void) {
    printk("\n=== End-to-End Benchmark ===\n");

    float frequencies[] = {500.0f, 1500.0f, 3000.0f};
    generate_multi_tone(test_audio, frequencies, 3, 0.3f);

    float probability;

    for (int i = 0; i < WARMUP_RUNS; i++) {
        generate_spectrogram(test_audio, features);
        run_inference(features, &probability);
    }

    uint32_t start = k_cycle_get_32();
    for (int i = 0; i < BENCHMARK_RUNS; i++) {
        generate_spectrogram(test_audio, features);
        run_inference(features, &probability);
    }
    uint32_t end = k_cycle_get_32();

    uint32_t total_cycles = end - start;
    uint32_t avg_us = k_cyc_to_us_near32(total_cycles / BENCHMARK_RUNS);

    printk("Full Pipeline (DSP + Inference):\n");
    printk("  Average latency: %u us (%.3f ms)\n", avg_us, (double)((float)avg_us / 1000.0f));
    printk("  Real-time factor: %.2fx\n", rate_per_second(avg_us));
}

void run_benchmarks(void) {
    printk("\n");
    printk("===============================================================\n");
    printk("   Performance Benchmarking Framework\n");
    printk("===============================================================\n");
    printk("Platform: nRF5340 @ %d MHz\n", SystemCoreClock / 1000000);
    printk("Compiler: GCC %s\n", __VERSION__);
    printk("Optimization: %s\n",
        #if defined(__OPTIMIZE_SIZE__)
            "Size (-Os)"
        #elif defined(__OPTIMIZE__)
            "Speed (-O2/O3)"
        #else
            "Debug (-O0)"
        #endif
    );

    benchmark_dsp();
    benchmark_dsp_variety();
    benchmark_inference();
    benchmark_end_to_end();

#ifdef EXPORT_C_REFERENCE_SPECTROGRAM
    printk("\n=== C Reference Spectrogram Dump ===\n");
    dump_c_reference_spectrogram();
#endif

#ifdef EXPORT_QUANTIZED_REFERENCE
    printk("\n=== Quantized Tensor Reference Dump ===\n");
    export_quantized_reference_only();
#endif

    printk("\n===============================================================\n");
    printk("Benchmark complete. Compare results to documented claims.\n");
    printk("===============================================================\n\n");
}

void export_c_reference_only(void) {
#ifdef EXPORT_C_REFERENCE_SPECTROGRAM
    printk("\n=== C Reference Spectrogram Dump (DSP only) ===\n");
    dump_c_reference_spectrogram();
#else
    printk("C reference export disabled (build without EXPORT_C_REFERENCE_SPECTROGRAM)\n");
#endif
}

void export_quantized_reference_only(void) {
#ifdef EXPORT_QUANTIZED_REFERENCE
    generate_bit_exact_reference_signal(test_audio);

    if (generate_spectrogram(test_audio, features) != 0) {
        printk("QREF_ERROR spectrogram_generation_failed\n");
        return;
    }

    float probability = 0.0f;
    if (run_inference(features, &probability) != 0) {
        printk("QREF_ERROR inference_failed\n");
        return;
    }

    int input_bytes = inference_copy_last_quantized_input(qref_input, sizeof(qref_input));
    int output_bytes = inference_copy_last_quantized_output(qref_output, sizeof(qref_output));
    if (input_bytes <= 0 || output_bytes <= 0) {
        printk("QREF_ERROR tensor_capture_failed in=%d out=%d\n", input_bytes, output_bytes);
        return;
    }

    printk("QREF_BEGIN input_bytes=%d output_bytes=%d prob=%.9f\n",
           input_bytes, output_bytes, (double)probability);

    for (int i = 0; i < input_bytes; i++) {
        printk("QREF_IN %d %d\n", i, (int)qref_input[i]);
        if ((i & 0x1F) == 0x1F) {
            k_msleep(1);
        }
    }

    for (int i = 0; i < output_bytes; i++) {
        printk("QREF_OUT %d %d\n", i, (int)qref_output[i]);
    }

    printk("QREF_END\n");
#else
    printk("Quantized reference export disabled (build without EXPORT_QUANTIZED_REFERENCE)\n");
#endif
}
