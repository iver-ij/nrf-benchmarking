#include <zephyr/device.h>
#include <zephyr/ipc/ipc_service.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include <arm_math.h>
#include <math.h>
#include <errno.h>
#include <stdint.h>
#include <string.h>

#include "data.h"
#include "mel_constants.h"
#include "dual_core_protocol.h"

static struct ipc_ept dsp_ept;
static K_SEM_DEFINE(bound_sem, 0, 1);

static int16_t frame_buffer[FFT_SIZE];
static float window_func[FFT_SIZE];
static float fft_input_buffer[FFT_SIZE];
static float fft_output_buffer[FFT_SIZE];
static float fft_magnitude_buffer[NUM_FFT_BINS];

static arm_rfft_fast_instance_f32 fft_instance;
static arm_matrix_instance_f32 mel_filterbank_matrix;
static bool dsp_ready;
static bool frame_started;

#define LOGMEL_CANON_MASK 0xFFFF8000u

static inline float canonicalize_logmel(float value)
{
    union {
        float f;
        uint32_t u;
    } bits = { .f = value };
    bits.u &= LOGMEL_CANON_MASK;
    return bits.f;
}

static void dsp_frame_init(void)
{
    if (dsp_ready) {
        return;
    }

    arm_rfft_fast_init_f32(&fft_instance, FFT_SIZE);
    mel_filterbank_matrix.numRows = NUM_MEL_BINS;
    mel_filterbank_matrix.numCols = NUM_FFT_BINS;
    mel_filterbank_matrix.pData = (float32_t *)mel_filterbank_dense;
    for (int i = 0; i < FFT_SIZE; i++) {
        window_func[i] = 0.5f - 0.5f * arm_cos_f32(2.0f * PI * (float)i / (float)FFT_SIZE);
    }
    dsp_ready = true;
}

static void compute_mel_frame(const int16_t *samples, float *mel_out)
{
    for (int i = 0; i < FFT_SIZE; i++) {
        float normalized = (float)samples[i] / 32768.0f;
        fft_input_buffer[i] = normalized * window_func[i];
    }

    arm_rfft_fast_f32(&fft_instance, fft_input_buffer, fft_output_buffer, 0);

    fft_magnitude_buffer[0] = fabsf(fft_output_buffer[0]);
    fft_magnitude_buffer[NUM_FFT_BINS - 1] = fabsf(fft_output_buffer[1]);
    arm_cmplx_mag_f32(fft_output_buffer + 2, fft_magnitude_buffer + 1, NUM_FFT_BINS - 2);

    arm_mat_vec_mult_f32(&mel_filterbank_matrix, fft_magnitude_buffer, mel_out);

    for (int m = 0; m < NUM_MEL_BINS; m++) {
        mel_out[m] = canonicalize_logmel(logf(mel_out[m] + 1e-6f));
    }
}

static int ipc_send_blocking(const void *buf, size_t len)
{
    int ret;
    while (true) {
        ret = ipc_service_send(&dsp_ept, buf, len);
        if (ret == -ENOMEM || ret == -EBUSY) {
            k_busy_wait(200);
            continue;
        }
        return ret;
    }
}

static int send_result(uint8_t frame_idx)
{
    uint8_t txbuf[sizeof(struct tinyml_dsp_ipc_header) +
                  sizeof(struct tinyml_dsp_ipc_result_payload)];
    struct tinyml_dsp_ipc_header *hdr = (struct tinyml_dsp_ipc_header *)txbuf;
    struct tinyml_dsp_ipc_result_payload *payload =
        (struct tinyml_dsp_ipc_result_payload *)(txbuf + sizeof(*hdr));

    hdr->cmd = TINYML_DSP_CMD_FRAME_RESULT;
    hdr->frame_idx = frame_idx;
    hdr->payload_len = sizeof(*payload);
    compute_mel_frame(frame_buffer, payload->mel);

    return ipc_send_blocking(txbuf, sizeof(txbuf));
}

static void ep_bound(void *priv)
{
    ARG_UNUSED(priv);
    k_sem_give(&bound_sem);
}

static void ep_recv(const void *data, size_t len, void *priv)
{
    ARG_UNUSED(priv);
    if (len < sizeof(struct tinyml_dsp_ipc_header)) {
        return;
    }

    const struct tinyml_dsp_ipc_header *hdr = (const struct tinyml_dsp_ipc_header *)data;
    const uint8_t *payload = (const uint8_t *)data + sizeof(*hdr);
    size_t payload_available = len - sizeof(*hdr);

    switch (hdr->cmd) {
    case TINYML_DSP_CMD_FRAME_START:
        memset(frame_buffer, 0, sizeof(frame_buffer));
        frame_started = true;
        break;

    case TINYML_DSP_CMD_FRAME_CHUNK: {
        if (!frame_started) {
            break;
        }
        if (payload_available < sizeof(struct tinyml_dsp_ipc_chunk_prefix)) {
            break;
        }

        const struct tinyml_dsp_ipc_chunk_prefix *chunk =
            (const struct tinyml_dsp_ipc_chunk_prefix *)payload;
        size_t sample_bytes = (size_t)chunk->count * sizeof(int16_t);
        if (chunk->count > TINYML_DSP_IPC_MAX_CHUNK_SAMPLES ||
            (size_t)chunk->offset + (size_t)chunk->count > FFT_SIZE ||
            payload_available < sizeof(*chunk) + sample_bytes) {
            break;
        }

        const int16_t *samples = (const int16_t *)(payload + sizeof(*chunk));
        memcpy(&frame_buffer[chunk->offset], samples, sample_bytes);
        break;
    }

    case TINYML_DSP_CMD_FRAME_END:
        if (!frame_started) {
            break;
        }
        frame_started = false;
        (void)send_result(hdr->frame_idx);
        break;

    default:
        break;
    }
}

static struct ipc_ept_cfg ep_cfg = {
    .name = TINYML_DSP_IPC_EP_NAME,
    .cb = {
        .bound = ep_bound,
        .received = ep_recv,
    },
};

int main(void)
{
    dsp_frame_init();

    const struct device *ipc_instance = DEVICE_DT_GET(DT_NODELABEL(ipc0));
    if (!device_is_ready(ipc_instance)) {
        printk("CPUNET ERROR: ipc0 not ready\n");
        return 0;
    }

    int ret = ipc_service_open_instance(ipc_instance);
    if (ret < 0 && ret != -EALREADY) {
        printk("CPUNET ERROR: ipc_service_open_instance failed (%d)\n", ret);
        return 0;
    }

    ret = ipc_service_register_endpoint(ipc_instance, &dsp_ept, &ep_cfg);
    if (ret < 0) {
        printk("CPUNET ERROR: ipc_service_register_endpoint failed (%d)\n", ret);
        return 0;
    }

    if (k_sem_take(&bound_sem, K_SECONDS(10)) != 0) {
        printk("CPUNET ERROR: endpoint bind timeout\n");
        return 0;
    }

    printk("CPUNET DSP service ready (%s)\n", TINYML_DSP_IPC_EP_NAME);

    while (1) {
        k_msleep(1000);
    }
}
