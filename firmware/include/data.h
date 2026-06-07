#ifndef DATA_H
#define DATA_H

#include <stdint.h>

#define SAMPLE_RATE     16000
#define NUM_SAMPLES     16000

#define FFT_SIZE        512
#define NUM_FFT_BINS    257
#define NUM_MEL_BINS    40
#define WINDOW_STEP     320
#define SPECTROGRAM_ROWS 49
#define SPECTROGRAM_SIZE (SPECTROGRAM_ROWS * NUM_MEL_BINS)

#ifndef PI
#define PI 3.14159265358979323846f
#endif

#ifdef TEST_MODE
extern const int16_t audio_cat2[NUM_SAMPLES];
extern const int16_t audio_on2[NUM_SAMPLES];
#endif

#endif
