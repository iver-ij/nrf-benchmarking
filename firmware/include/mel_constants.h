#ifndef MEL_CONSTANTS_H
#define MEL_CONSTANTS_H

#include "data.h"

/*
 * Dense log-mel projection matrix for CMSIS-DSP matrix operations.
 *
 * Layout is row-major [mel_bin][fft_bin] so it can be passed directly
 * to arm_mat_vec_mult_f32() with the FFT magnitude vector.
 */
#define MEL_FILTERBANK_ROWS NUM_MEL_BINS
#define MEL_FILTERBANK_COLS NUM_FFT_BINS
#define MEL_FILTERBANK_DENSE_SIZE (MEL_FILTERBANK_ROWS * MEL_FILTERBANK_COLS)

extern const float mel_filterbank_dense[MEL_FILTERBANK_DENSE_SIZE];

#endif
