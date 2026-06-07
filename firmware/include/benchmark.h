#ifndef BENCHMARK_H
#define BENCHMARK_H

/**
 * Performance Benchmarking Framework
 *
 * Provides accurate timing measurements for DSP and inference operations
 * to verify optimization claims.
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Run all benchmarks and print results to console
 *
 * Call this during initialization to verify performance claims:
 * - DSP Frontend latency
 * - Neural network inference latency
 * - End-to-end pipeline latency
 */
void run_benchmarks(void);

/**
 * Benchmark DSP frontend (mel spectrogram computation)
 */
void benchmark_dsp(void);

/**
 * Benchmark neural network inference only
 */
void benchmark_inference(void);

/**
 * Benchmark complete pipeline (DSP + Inference)
 */
void benchmark_end_to_end(void);

/**
 * Export only the C reference spectrogram dump.
 *
 * This path intentionally skips inference setup/benchmarking to keep
 * cross-platform DSP equivalence capture deterministic and lightweight.
 */
void export_c_reference_only(void);

/**
 * Export quantized inference tensors for cross-platform int8 parity checks.
 *
 * Emits UART markers:
 * - QREF_BEGIN input_bytes=<n> output_bytes=<m>
 * - QREF_IN <idx> <int8>
 * - QREF_OUT <idx> <int8>
 * - QREF_END
 */
void export_quantized_reference_only(void);

#ifdef __cplusplus
}
#endif

#endif
