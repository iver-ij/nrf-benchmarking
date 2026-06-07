#include "dsp.h"
#include "data.h"
#include "mel_constants.h"

#include <arm_math.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>

static float fft_input_buffer[FFT_SIZE] __attribute__((aligned(16)));
static float fft_output_buffer[FFT_SIZE] __attribute__((aligned(16)));
static float fft_magnitude_buffer[NUM_FFT_BINS] __attribute__((aligned(16)));

static float window_func[FFT_SIZE] __attribute__((aligned(16)));
static arm_rfft_fast_instance_f32 fft_instance;
static arm_matrix_instance_f32 mel_filterbank_matrix;
static bool is_initialized = false;

#define LOGMEL_CANON_MASK 0xFFFF8000u

static inline float canonicalize_logmel(float value) {
    union {
        float f;
        uint32_t u;
    } bits = { .f = value };
    bits.u &= LOGMEL_CANON_MASK;
    return bits.f;
}

void dsp_init(void) {
    if (!is_initialized) {
        arm_rfft_fast_init_f32(&fft_instance, FFT_SIZE);
        mel_filterbank_matrix.numRows = NUM_MEL_BINS;
        mel_filterbank_matrix.numCols = NUM_FFT_BINS;
        mel_filterbank_matrix.pData = (float32_t *)mel_filterbank_dense;
        for (int i = 0; i < FFT_SIZE; i++) {
            window_func[i] = 0.5f - 0.5f * arm_cos_f32(2.0f * PI * (float)i / (float)FFT_SIZE);
        }
        is_initialized = true;
    }
}

/*
 * Mel frontend uses CMSIS-DSP FFT + magnitude + matrix-vector projection.
 * The dense mel matrix is generated from the Python training pipeline so the
 * runtime projection matches the host-side reference constants exactly.
 */
int generate_spectrogram(const int16_t* audio_in, float* features_out) {
    if (audio_in == NULL || features_out == NULL) {
        return -1;
    }

    if (!is_initialized) {
        dsp_init();
    }

    for (int frame = 0; frame < SPECTROGRAM_ROWS; frame++) {
        int read_index = frame * WINDOW_STEP;
        int write_index = frame * NUM_MEL_BINS;

        for (int j = 0; j < FFT_SIZE; j++) {
            if (read_index + j < NUM_SAMPLES) {
                float normalized = (float)audio_in[read_index + j] / 32768.0f;
                fft_input_buffer[j] = normalized * window_func[j];
            } else {
                fft_input_buffer[j] = 0.0f;
            }
        }

        arm_rfft_fast_f32(&fft_instance, fft_input_buffer, fft_output_buffer, 0);

        fft_magnitude_buffer[0] = fabsf(fft_output_buffer[0]);
        fft_magnitude_buffer[NUM_FFT_BINS - 1] = fabsf(fft_output_buffer[1]);

        arm_cmplx_mag_f32(fft_output_buffer + 2, fft_magnitude_buffer + 1, NUM_FFT_BINS - 2);

        float mel_bins[NUM_MEL_BINS] __attribute__((aligned(16)));
        arm_mat_vec_mult_f32(&mel_filterbank_matrix, fft_magnitude_buffer, mel_bins);

        for (int m = 0; m < NUM_MEL_BINS; m++) {
            float logmel = logf(mel_bins[m] + 1e-6f);
            features_out[write_index + m] = canonicalize_logmel(logmel);
        }
    }

    return 0;
}
