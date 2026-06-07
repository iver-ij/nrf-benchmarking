#include "dual_core_ipc.h"
#include "dual_core_protocol.h"
#include "data.h"

#include <zephyr/device.h>
#include <zephyr/ipc/ipc_service.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/cache.h>

#include <stdbool.h>
#include <errno.h>
#include <string.h>

static const struct device *ipc_instance;
static struct ipc_ept dsp_ept;
static bool ipc_initialized;
static bool ipc_bound;

static K_SEM_DEFINE(bound_sem, 0, 1);
static K_SEM_DEFINE(result_sem, 0, 1);
static K_MUTEX_DEFINE(ipc_mutex);

static float frame_result[NUM_MEL_BINS];
static uint8_t awaited_frame_idx;

#define TINYML_DSP_IPC_SEND_RETRY_US 200
#define TINYML_DSP_IPC_RESULT_TIMEOUT_MS 200
#define TINYML_DSP_IPC_FRAME_ATTEMPTS 1

static inline void ipc_cache_prepare_for_cpu(void *addr, size_t size)
{
#if defined(CONFIG_CACHE_MANAGEMENT) && defined(CONFIG_DCACHE)
	(void)sys_cache_data_invd_range(addr, size);
#else
	ARG_UNUSED(addr);
	ARG_UNUSED(size);
#endif
}

static inline void ipc_cache_prepare_for_device(void *addr, size_t size)
{
#if defined(CONFIG_CACHE_MANAGEMENT) && defined(CONFIG_DCACHE)
	(void)sys_cache_data_flush_range(addr, size);
#else
	ARG_UNUSED(addr);
	ARG_UNUSED(size);
#endif
}

static bool ipc_send_error_is_transient(int ret)
{
    return (ret == -ENOMEM || ret == -EBUSY);
}

static void ipc_ep_bound(void *priv)
{
    ARG_UNUSED(priv);
    ipc_bound = true;
    k_sem_give(&bound_sem);
}

static void ipc_ep_received(const void *data, size_t len, void *priv)
{
    ARG_UNUSED(priv);
    if (len < sizeof(struct tinyml_dsp_ipc_header)) {
        return;
    }

    const struct tinyml_dsp_ipc_header *hdr = (const struct tinyml_dsp_ipc_header *)data;
    ipc_cache_prepare_for_cpu((void *)data, len);
    const uint8_t *payload = (const uint8_t *)data + sizeof(*hdr);
    size_t payload_available = len - sizeof(*hdr);

    if (hdr->cmd == TINYML_DSP_CMD_FRAME_RESULT) {
        if (hdr->payload_len != sizeof(struct tinyml_dsp_ipc_result_payload) ||
            payload_available < sizeof(struct tinyml_dsp_ipc_result_payload)) {
            return;
        }

        const struct tinyml_dsp_ipc_result_payload *res =
            (const struct tinyml_dsp_ipc_result_payload *)payload;

        if (hdr->frame_idx == awaited_frame_idx) {
            memcpy(frame_result, res->mel, sizeof(frame_result));
            k_sem_give(&result_sem);
        }
    }
}

static struct ipc_ept_cfg dsp_ept_cfg = {
    .name = TINYML_DSP_IPC_EP_NAME,
    .cb = {
        .bound = ipc_ep_bound,
        .received = ipc_ep_received,
    },
};

static int ipc_send_blocking(const void *buf, size_t len)
{
    int ret;
    ipc_cache_prepare_for_device((void *)buf, len);
    while (true) {
        ret = ipc_service_send(&dsp_ept, buf, len);
        if (ipc_send_error_is_transient(ret)) {
            k_busy_wait(TINYML_DSP_IPC_SEND_RETRY_US);
            continue;
        }
        return ret;
    }
}

static int offload_frame_once(const int16_t *frame_samples, uint8_t frame_idx, float *mel_out)
{
    uint8_t txbuf[sizeof(struct tinyml_dsp_ipc_header) +
                  sizeof(struct tinyml_dsp_ipc_chunk_prefix) +
                  (TINYML_DSP_IPC_MAX_CHUNK_SAMPLES * sizeof(int16_t))];
    struct tinyml_dsp_ipc_header *hdr = (struct tinyml_dsp_ipc_header *)txbuf;

    hdr->cmd = TINYML_DSP_CMD_FRAME_START;
    hdr->frame_idx = frame_idx;
    hdr->payload_len = 0U;
    if (ipc_send_blocking(txbuf, sizeof(*hdr)) < 0) {
        return -1;
    }

    uint16_t offset = 0U;
    while (offset < FFT_SIZE) {
        uint16_t count = (uint16_t)(FFT_SIZE - offset);
        if (count > TINYML_DSP_IPC_MAX_CHUNK_SAMPLES) {
            count = TINYML_DSP_IPC_MAX_CHUNK_SAMPLES;
        }

        hdr->cmd = TINYML_DSP_CMD_FRAME_CHUNK;
        hdr->frame_idx = frame_idx;
        hdr->payload_len = (uint16_t)(sizeof(struct tinyml_dsp_ipc_chunk_prefix) + count * sizeof(int16_t));

        struct tinyml_dsp_ipc_chunk_prefix *chunk =
            (struct tinyml_dsp_ipc_chunk_prefix *)(txbuf + sizeof(*hdr));
        chunk->offset = offset;
        chunk->count = count;
        memcpy((uint8_t *)chunk + sizeof(*chunk), frame_samples + offset, count * sizeof(int16_t));

        size_t tx_len = sizeof(*hdr) + hdr->payload_len;
        if (ipc_send_blocking(txbuf, tx_len) < 0) {
            return -1;
        }

        offset = (uint16_t)(offset + count);
    }

    awaited_frame_idx = frame_idx;
    k_sem_reset(&result_sem);

    hdr->cmd = TINYML_DSP_CMD_FRAME_END;
    hdr->frame_idx = frame_idx;
    hdr->payload_len = 0U;
    if (ipc_send_blocking(txbuf, sizeof(*hdr)) < 0) {
        return -1;
    }

    if (k_sem_take(&result_sem, K_MSEC(TINYML_DSP_IPC_RESULT_TIMEOUT_MS)) != 0) {
        return -1;
    }

    memcpy(mel_out, frame_result, sizeof(frame_result));
    return 0;
}

static int offload_frame(const int16_t *frame_samples, uint8_t frame_idx, float *mel_out)
{
    int ret = -1;

    for (int attempt = 0; attempt < TINYML_DSP_IPC_FRAME_ATTEMPTS; attempt++) {
        ret = offload_frame_once(frame_samples, frame_idx, mel_out);
        if (ret == 0) {
            return 0;
        }

        if (attempt + 1 < TINYML_DSP_IPC_FRAME_ATTEMPTS) {
            printk("WARN: Dual-core DSP frame %u retry %d/%d\n",
                   frame_idx,
                   attempt + 1,
                   TINYML_DSP_IPC_FRAME_ATTEMPTS - 1);
            k_msleep(5);
        }
    }

    printk("WARN: Dual-core DSP frame %u failed after %d attempts\n",
           frame_idx,
           TINYML_DSP_IPC_FRAME_ATTEMPTS);
    return ret;
}

int dual_core_dsp_init(void)
{
    if (ipc_initialized) {
        return 0;
    }

    ipc_instance = DEVICE_DT_GET(DT_NODELABEL(ipc0));
    if (!device_is_ready(ipc_instance)) {
        printk("ERROR: IPC instance not ready\n");
        return -1;
    }

    int ret = ipc_service_open_instance(ipc_instance);
    if (ret < 0 && ret != -EALREADY) {
        printk("ERROR: ipc_service_open_instance failed (%d)\n", ret);
        return -1;
    }

    ret = ipc_service_register_endpoint(ipc_instance, &dsp_ept, &dsp_ept_cfg);
    if (ret < 0) {
        printk("ERROR: ipc_service_register_endpoint failed (%d)\n", ret);
        return -1;
    }

    if (!ipc_bound && k_sem_take(&bound_sem, K_SECONDS(10)) != 0) {
        printk("ERROR: IPC endpoint bind timeout\n");
        return -1;
    }

    ipc_initialized = true;
    return 0;
}

int dual_core_dsp_ready(void)
{
    return (ipc_initialized && ipc_bound) ? 1 : 0;
}

int dual_core_generate_spectrogram(const int16_t *audio_in, float *features_out)
{
    if (audio_in == NULL || features_out == NULL) {
        return -1;
    }

    if (dual_core_dsp_init() != 0) {
        return -1;
    }

    int ret = 0;
    int read_index = 0;
    int write_index = 0;
    int16_t frame_samples[FFT_SIZE];

    k_mutex_lock(&ipc_mutex, K_FOREVER);
    for (uint8_t frame = 0; frame < SPECTROGRAM_ROWS; frame++) {
        memset(frame_samples, 0, sizeof(frame_samples));
        for (int i = 0; i < FFT_SIZE; i++) {
            int src = read_index + i;
            if (src >= 0 && src < NUM_SAMPLES) {
                frame_samples[i] = audio_in[src];
            }
        }

        ret = offload_frame(frame_samples, frame, &features_out[write_index]);
        if (ret != 0) {
            break;
        }

        write_index += NUM_MEL_BINS;
        read_index += WINDOW_STEP;
    }
    k_mutex_unlock(&ipc_mutex);

    return ret;
}
