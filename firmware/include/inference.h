#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize TensorFlow Lite Micro inference engine
 * @return 0 on success, -1 on failure
 */
int inference_setup(void);

/**
 * Run inference on spectrogram data
 *
 * @param spectrogram_data Input spectrogram (49x40 float array)
 * @param output_probability Output parameter - probability of "ON" class (0.0-1.0)
 * @return 0 on completed inference, -1 on runtime failure.
 *
 * A low-confidence result is still a completed inference; thresholding belongs to the caller.
 */
int run_inference(const float* spectrogram_data, float* output_probability);

/**
 * Query configured tensor arena size for logging/diagnostics.
 *
 * @return Tensor arena size in bytes.
 */
uint32_t inference_tensor_arena_size(void);

/**
 * Copy latest quantized model input tensor produced by run_inference().
 *
 * @param out Destination buffer.
 * @param out_len Destination capacity in bytes.
 * @return Number of bytes copied, or -1 on invalid arguments / unavailable data.
 */
int inference_copy_last_quantized_input(int8_t *out, size_t out_len);

/**
 * Copy latest quantized model output tensor produced by run_inference().
 *
 * @param out Destination buffer.
 * @param out_len Destination capacity in bytes.
 * @return Number of bytes copied, or -1 on invalid arguments / unavailable data.
 */
int inference_copy_last_quantized_output(int8_t *out, size_t out_len);

#ifdef __cplusplus
}
#endif

#endif
