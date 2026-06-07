#ifndef DSP_H
#define DSP_H
#include <stdint.h>

int generate_spectrogram(const int16_t* waveform, float* spectrogram);

#endif