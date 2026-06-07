#ifndef DUAL_CORE_PROTOCOL_H
#define DUAL_CORE_PROTOCOL_H

#include <stdint.h>
#include "data.h"

#define TINYML_DSP_IPC_EP_NAME "tinyml_dsp_ep"

#define TINYML_DSP_IPC_MAX_CHUNK_SAMPLES 240U

enum tinyml_dsp_ipc_cmd {
    TINYML_DSP_CMD_FRAME_START = 1,
    TINYML_DSP_CMD_FRAME_CHUNK = 2,
    TINYML_DSP_CMD_FRAME_END = 3,
    TINYML_DSP_CMD_FRAME_RESULT = 4
};

struct tinyml_dsp_ipc_header {
    uint8_t cmd;
    uint8_t frame_idx;
    uint16_t payload_len;
} __attribute__((packed));

struct tinyml_dsp_ipc_chunk_prefix {
    uint16_t offset;
    uint16_t count;
} __attribute__((packed));

struct tinyml_dsp_ipc_result_payload {
    float mel[NUM_MEL_BINS];
} __attribute__((packed));

#endif
