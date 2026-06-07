#ifndef TEST_SIGNALS_H
#define TEST_SIGNALS_H

#include <stdint.h>
#include "data.h"

#ifdef __cplusplus
extern "C" {
#endif

void generate_sine_wave(int16_t* buffer, float frequency_hz, float amplitude);
void generate_white_noise(int16_t* buffer, float amplitude);
void generate_silence(int16_t* buffer);
void generate_chirp(int16_t* buffer, float f_start_hz, float f_end_hz, float amplitude);
void generate_multi_tone(int16_t* buffer, const float* frequencies, int num_tones, float amplitude);

#ifdef __cplusplus
}
#endif

#endif
