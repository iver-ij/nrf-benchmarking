#include "test_signals.h"
#include <math.h>
#include <stddef.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

void generate_sine_wave(int16_t* buffer, float frequency_hz, float amplitude) {
    if (buffer == NULL) {
        return;
    }

    float max_amplitude = 32767.0f * amplitude;
    float phase_increment = 2.0f * M_PI * frequency_hz / SAMPLE_RATE;

    for (int i = 0; i < NUM_SAMPLES; i++) {
        float phase = phase_increment * i;
        buffer[i] = (int16_t)(max_amplitude * sinf(phase));
    }
}

void generate_white_noise(int16_t* buffer, float amplitude) {
    if (buffer == NULL) {
        return;
    }

    float max_amplitude = 32767.0f * amplitude;
    unsigned int seed = 42;

    for (int i = 0; i < NUM_SAMPLES; i++) {
        seed = (1103515245 * seed + 12345) & 0x7fffffff;
        float random_val = (float)seed / 0x7fffffff;
        random_val = 2.0f * random_val - 1.0f;
        buffer[i] = (int16_t)(max_amplitude * random_val);
    }
}

void generate_silence(int16_t* buffer) {
    if (buffer == NULL) {
        return;
    }

    for (int i = 0; i < NUM_SAMPLES; i++) {
        buffer[i] = 0;
    }
}

void generate_chirp(int16_t* buffer, float f_start_hz, float f_end_hz, float amplitude) {
    if (buffer == NULL) {
        return;
    }

    float max_amplitude = 32767.0f * amplitude;
    float duration_sec = (float)NUM_SAMPLES / SAMPLE_RATE;

    for (int i = 0; i < NUM_SAMPLES; i++) {
        float t = (float)i / SAMPLE_RATE;
        float phase = 2.0f * M_PI * (f_start_hz * t +
                     (f_end_hz - f_start_hz) * t * t / (2.0f * duration_sec));
        buffer[i] = (int16_t)(max_amplitude * sinf(phase));
    }
}

void generate_multi_tone(int16_t* buffer, const float* frequencies, int num_tones, float amplitude) {
    if (buffer == NULL || frequencies == NULL || num_tones <= 0) {
        return;
    }

    float max_amplitude = 32767.0f * amplitude / num_tones;

    for (int i = 0; i < NUM_SAMPLES; i++) {
        buffer[i] = 0;
    }

    for (int tone = 0; tone < num_tones; tone++) {
        float phase_increment = 2.0f * M_PI * frequencies[tone] / SAMPLE_RATE;
        for (int i = 0; i < NUM_SAMPLES; i++) {
            float phase = phase_increment * i;
            buffer[i] += (int16_t)(max_amplitude * sinf(phase));
        }
    }
}
