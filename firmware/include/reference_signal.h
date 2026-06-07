#ifndef REFERENCE_SIGNAL_H
#define REFERENCE_SIGNAL_H

#include <stdint.h>
#include "data.h"

/*
 * Deterministic integer-only reference signal used by both firmware and
 * host-side DSP verification. Integer arithmetic avoids libm differences
 * (for example sinf/logf implementation drift) in the stimulus path.
 */
static inline void generate_bit_exact_reference_signal(int16_t *buffer) {
    uint32_t lcg = 0xC001D00Du;
    uint32_t xorshift = 0x1BADB002u;

    for (int i = 0; i < NUM_SAMPLES; i++) {
        lcg = lcg * 1664525u + 1013904223u;

        xorshift ^= xorshift << 13;
        xorshift ^= xorshift >> 17;
        xorshift ^= xorshift << 5;

        int32_t a = (int32_t)((lcg >> 16) & 0x7FFu) - 1024;
        int32_t b = (int32_t)(xorshift & 0x3FFu) - 512;

        int32_t sample = a * 24 + b * 12;
        buffer[i] = (int16_t)sample;
    }
}

#endif
