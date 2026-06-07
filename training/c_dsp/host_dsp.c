#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include "arm_math.h"
#include "data.h"
#include "reference_signal.h"
#include "mel_constants.h"

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

static float fft_input_buffer[FFT_SIZE];
static float fft_output_buffer[FFT_SIZE];
static float fft_magnitude_buffer[NUM_FFT_BINS];
static float window_func[FFT_SIZE];

int tinyml_generate_reference_multitone_int16(int16_t *buffer) {
    if (buffer == 0) {
        return -1;
    }

    generate_bit_exact_reference_signal(buffer);
    return 0;
}

int tinyml_host_dsp_init(void) {
    if (is_initialized) {
        return 0;
    }

    if (arm_rfft_fast_init_f32(&fft_instance, FFT_SIZE) != ARM_MATH_SUCCESS) {
        return -1;
    }
    mel_filterbank_matrix.numRows = NUM_MEL_BINS;
    mel_filterbank_matrix.numCols = NUM_FFT_BINS;
    mel_filterbank_matrix.pData = (float32_t *)mel_filterbank_dense;

    for (int i = 0; i < FFT_SIZE; i++) {
        window_func[i] = 0.5f - 0.5f * arm_cos_f32(2.0f * PI * (float)i / (float)FFT_SIZE);
    }

    is_initialized = true;
    return 0;
}

int tinyml_generate_spectrogram_int16(const int16_t *audio_in, float *features_out) {
    if (audio_in == 0 || features_out == 0) {
        return -1;
    }

    if (!is_initialized && tinyml_host_dsp_init() != 0) {
        return -2;
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

        float mel_bins[NUM_MEL_BINS];
        arm_mat_vec_mult_f32(&mel_filterbank_matrix, fft_magnitude_buffer, mel_bins);

        for (int m = 0; m < NUM_MEL_BINS; m++) {
            float logmel = logf(mel_bins[m] + 1e-6f);
            features_out[write_index + m] = canonicalize_logmel(logmel);
        }
    }

    return 0;
}
