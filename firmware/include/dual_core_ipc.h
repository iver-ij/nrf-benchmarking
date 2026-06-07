#ifndef DUAL_CORE_IPC_H
#define DUAL_CORE_IPC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int dual_core_dsp_init(void);
int dual_core_generate_spectrogram(const int16_t *audio_in, float *features_out);
int dual_core_dsp_ready(void);

#ifdef __cplusplus
}
#endif

#endif
