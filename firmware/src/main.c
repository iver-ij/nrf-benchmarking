#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/devicetree.h>
#include <zephyr/device.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/logging/log.h>
#include <zephyr/cache.h>
#include <zephyr/usb/usb_device.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>

#include "data.h"
#include "dsp.h"
#include "inference.h"
#include <hal/nrf_cache.h>
#ifdef TINYML_DUAL_CORE_DSP_OFFLOAD
#include "dual_core_ipc.h"
#endif

#if defined(TEST_MODE) || defined(EXPORT_C_REFERENCE_SPECTROGRAM) || defined(EXPORT_QUANTIZED_REFERENCE)
#include "benchmark.h"
#endif

#if defined(CONFIG_WIFI) && defined(CONFIG_NET_SOCKETS)
#define WIFI_TELEMETRY_ENABLED
#endif

LOG_MODULE_REGISTER(main, CONFIG_LOG_DEFAULT_LEVEL);

#define CONFIDENCE_THRESHOLD 0.75f
#define SIGNAL_NOISE_FLOOR   2.0f

#ifndef TINYML_WIFI_RUNTIME
#define TINYML_WIFI_RUNTIME 1
#endif

#ifndef TINYML_TELEMETRY_SEND_STARTUP
#define TINYML_TELEMETRY_SEND_STARTUP 1
#endif

#ifndef TINYML_TELEMETRY_SEND_TARGET
#define TINYML_TELEMETRY_SEND_TARGET 1
#endif

#ifndef TINYML_TELEMETRY_SEND_PER_FRAME
#define TINYML_TELEMETRY_SEND_PER_FRAME 0
#endif

#ifndef TINYML_PRINT_STAGE_TIMINGS
#define TINYML_PRINT_STAGE_TIMINGS 1
#endif

#ifndef TINYML_POWER_DSP_ONLY
#define TINYML_POWER_DSP_ONLY 0
#endif

#ifndef TINYML_POWER_NN_ONLY
#define TINYML_POWER_NN_ONLY 0
#endif

#ifndef TINYML_WIFI_DIAGNOSTIC_SCAN
#define TINYML_WIFI_DIAGNOSTIC_SCAN 0
#endif

/*
 * Keep a pre-connect scan enabled by default. On iPhone hotspots this avoids
 * early association timeouts (-116) seen when connect is requested immediately
 * after backend-ready.
 */
#ifndef TINYML_WIFI_PRECONNECT_SCAN
#define TINYML_WIFI_PRECONNECT_SCAN 1
#endif

#define TELEMETRY_CONNECT_RETRY_MS 750
#define TELEMETRY_SOCKET_REFRESH_MS 4000
#define TELEMETRY_TRANSIENT_STREAK_REFRESH 8
#define TELEMETRY_TRANSIENT_STREAK_WIFI_RECOVER 3
#define TELEMETRY_WIFI_RECOVER_COOLDOWN_MS 10000
#define TELEMETRY_WIFI_RECOVER_MIN_OUTAGE_MS 12000
#define TELEMETRY_STARTUP_RETRY_COUNT 4
#define TELEMETRY_STARTUP_RETRY_INTERVAL_MS 1000
#define USB_CONSOLE_DTR_WAIT_MS 5000

#ifdef WIFI_TELEMETRY_ENABLED
    #include <zephyr/net/net_if.h>
    #include <zephyr/net/wifi_mgmt.h>
    #include <zephyr/net/net_event.h>
    #include <net/wifi_mgmt_ext.h>
    #if defined(CONFIG_WIFI_READY_LIB)
    #include <net/wifi_ready.h>
    #endif
    #include <zephyr/net/socket.h>
    #include "wifi_credentials.h"
#endif

static const struct gpio_dt_spec led_target = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);
static const struct gpio_dt_spec pin_timing = GPIO_DT_SPEC_GET(DT_ALIAS(led1), gpios);

static int16_t audio_buffer[NUM_SAMPLES] __attribute__((aligned(16)));
static float features_buffer[SPECTROGRAM_SIZE] __attribute__((aligned(16)));
static bool nn_only_features_ready;

static int init_console_transport(void)
{
#if defined(CONFIG_USB_DEVICE_STACK) && \
    DT_NODE_HAS_COMPAT(DT_CHOSEN(zephyr_console), zephyr_cdc_acm_uart)
    const struct device *console = DEVICE_DT_GET(DT_CHOSEN(zephyr_console));
    int ret = usb_enable(NULL);

    if (ret && ret != -EALREADY) {
        return ret;
    }

    if (!device_is_ready(console)) {
        return -ENODEV;
    }

#if defined(CONFIG_UART_LINE_CTRL)
    int64_t deadline = k_uptime_get() + USB_CONSOLE_DTR_WAIT_MS;

    while (k_uptime_get() < deadline) {
        uint32_t dtr = 0;

        if (uart_line_ctrl_get(console, UART_LINE_CTRL_DTR, &dtr) == 0 && dtr) {
            (void)uart_line_ctrl_set(console, UART_LINE_CTRL_DCD, 1);
            (void)uart_line_ctrl_set(console, UART_LINE_CTRL_DSR, 1);
            break;
        }

        k_sleep(K_MSEC(50));
    }
#endif
#endif

    return 0;
}

static inline void dma_cache_prepare_for_cpu(void *addr, size_t size) {
#if defined(CONFIG_CACHE_MANAGEMENT) && defined(CONFIG_DCACHE)
    (void)sys_cache_data_invd_range(addr, size);
#else
    ARG_UNUSED(addr);
    ARG_UNUSED(size);
#endif
}

static inline void dma_cache_prepare_for_device(void *addr, size_t size) {
#if defined(CONFIG_CACHE_MANAGEMENT) && defined(CONFIG_DCACHE)
    (void)sys_cache_data_flush_range(addr, size);
#else
    ARG_UNUSED(addr);
    ARG_UNUSED(size);
#endif
}

#ifdef WIFI_TELEMETRY_ENABLED
#define WIFI_RECONNECT_THREAD_STACK 8192
#define WIFI_RECONNECT_THREAD_PRIO 7
#define WIFI_READY_WAIT_SECONDS 5
#define WIFI_CONNECT_RESULT_WAIT_SECONDS 20
#define WIFI_RECONNECT_FAST_RETRY_SECONDS 5
#define WIFI_RECONNECT_SLOW_RETRY_SECONDS 15
#define WIFI_RECONNECT_RESET_STREAK 3

#if TINYML_WIFI_RUNTIME == 1
#if defined(CONFIG_WIFI_CREDENTIALS_STATIC)
    static const uint8_t wifi_static_ssid[] = CONFIG_WIFI_CREDENTIALS_STATIC_SSID;
    static const uint8_t wifi_static_psk[] = CONFIG_WIFI_CREDENTIALS_STATIC_PASSWORD;
    /*
     * Keep connect params in static storage. WiFi mgmt requests can complete
     * asynchronously; stack-backed params can become invalid before driver use.
     */
    static struct wifi_connect_req_params wifi_static_connect_params = {
        .ssid = wifi_static_ssid,
        .ssid_length = sizeof(wifi_static_ssid) - 1,
        .psk = wifi_static_psk,
        .psk_length = sizeof(wifi_static_psk) - 1,
        .sae_password = wifi_static_psk,
        .sae_password_length = sizeof(wifi_static_psk) - 1,
        .security = (sizeof(wifi_static_psk) > 1) ? WIFI_SECURITY_TYPE_PSK :
                                                    WIFI_SECURITY_TYPE_NONE,
        .channel = WIFI_CHANNEL_ANY,
        .band = WIFI_FREQ_BAND_2_4_GHZ,
        .mfp = WIFI_MFP_OPTIONAL,
        .timeout = 30,
    };
#endif

    static struct net_mgmt_event_callback wifi_mgmt_cb;
    static K_SEM_DEFINE(wifi_connect_result_sem, 0, 1);
    static K_SEM_DEFINE(wifi_scan_done_sem, 0, 1);
    static K_SEM_DEFINE(wifi_reconnect_sem, 0, 1);
    static K_SEM_DEFINE(wifi_ready_sem, 0, 1);
    static bool wifi_ready_status;
    static bool wifi_is_connected;
    static bool wifi_runtime_active;
    static bool wifi_connect_request_active;
    static bool wifi_scan_target_seen;
    static uint32_t wifi_reconnect_failures;
    static int wifi_last_connect_status;
    static int telemetry_sock = -1;
    static struct sockaddr_in telemetry_dest;
    static bool telemetry_dest_ready;
    static bool telemetry_peer_connected;
    static uint32_t telemetry_transient_failures;
    static int64_t telemetry_last_connect_attempt_ms;
    static int64_t telemetry_last_socket_refresh_ms;
    static int64_t telemetry_last_success_ms;
    static int64_t telemetry_last_wifi_recover_ms;
    static uint8_t telemetry_startup_retries_left;
    static void send_telemetry_event(const char *msg);
    static void telemetry_schedule_startup_burst(void);
    static void telemetry_startup_work_handler(struct k_work *work);
    static void telemetry_reset_socket_state(void);
    static bool telemetry_maybe_connect_peer(void);
    static void telemetry_trigger_wifi_reconnect(int err, const char *reason);
    static void setup_udp_socket(void);
    static void wifi_reconnect_thread(void *a, void *b, void *c);
    K_WORK_DELAYABLE_DEFINE(telemetry_startup_work, telemetry_startup_work_handler);
    K_THREAD_DEFINE(wifi_reconnect_tid,
                    WIFI_RECONNECT_THREAD_STACK,
                    wifi_reconnect_thread,
                    NULL,
                    NULL,
                    NULL,
                    WIFI_RECONNECT_THREAD_PRIO,
                    0,
                    0);

    static void drain_connect_result_sem(void)
    {
        while (k_sem_take(&wifi_connect_result_sem, K_NO_WAIT) == 0) {
        }
    }

    static int ensure_wifi_iface_up(struct net_if *iface, const char *tag)
    {
        int ret;

        if (!iface) {
            return -EINVAL;
        }

        if (net_if_is_admin_up(iface)) {
            return 0;
        }

        ret = net_if_up(iface);
        if (ret < 0 && ret != -EALREADY) {
            printk("[WiFi] %s iface up failed: %d\n", tag, ret);
            return ret;
        }

        printk("[WiFi] %s iface up\n", tag);
        return 0;
    }

    static void print_wifi_iface_status(struct net_if *iface, const char *tag)
    {
        struct wifi_iface_status status = {0};

        if (!iface) {
            return;
        }

        if (net_mgmt(NET_REQUEST_WIFI_IFACE_STATUS,
                     iface,
                     &status,
                     sizeof(status)) != 0) {
            printk("[WiFi] %s iface status query failed\n", tag);
            return;
        }

        printk("[WiFi] %s state=%s ssid=%.*s band=%s ch=%u sec=%s rssi=%d\n",
               tag,
               wifi_state_txt(status.state),
               (int)status.ssid_len,
               status.ssid,
               wifi_band_txt(status.band),
               status.channel,
               wifi_security_txt(status.security),
               status.rssi);
    }

    static void schedule_wifi_reconnect(k_timeout_t delay)
    {
        ARG_UNUSED(delay);

        if (!wifi_runtime_active) {
            return;
        }

        if (k_sem_count_get(&wifi_reconnect_sem) == 0) {
            (void)k_sem_give(&wifi_reconnect_sem);
        }
    }

    static int issue_wifi_connect_request(struct net_if *iface, bool use_stored)
    {
        int ret;

        wifi_last_connect_status = -EINPROGRESS;

        if (use_stored) {
            printk("[WiFi] Connect request: STORED credentials\n");
            ret = net_mgmt(NET_REQUEST_WIFI_CONNECT_STORED, iface, NULL, 0);
        } else {
#if defined(CONFIG_WIFI_CREDENTIALS_STATIC)
            printk("[WiFi] Connect request: EXPLICIT credentials (SSID=%s)\n",
                   CONFIG_WIFI_CREDENTIALS_STATIC_SSID);
            printk("[WiFi] Explicit profile: sec=%s band=%s ch=%u\n",
                   wifi_security_txt(wifi_static_connect_params.security),
                   wifi_band_txt(wifi_static_connect_params.band),
                   wifi_static_connect_params.channel);
            ret = net_mgmt(NET_REQUEST_WIFI_CONNECT,
                           iface,
                           (void *)&wifi_static_connect_params,
                           sizeof(wifi_static_connect_params));
#else
            ARG_UNUSED(iface);
            printk("[WiFi] Explicit connect unavailable (CONFIG_WIFI_CREDENTIALS_STATIC=n)\n");
            return -ENOTSUP;
#endif
        }

        if (ret) {
            printk("[WiFi] Connect request failed immediately: %d\n", ret);
            return ret;
        }

        return 0;
    }

    static int request_wifi_connect(struct net_if *iface, bool use_stored)
    {
        int ret;

        ret = ensure_wifi_iface_up(iface, "Connect");
        if (ret != 0) {
            return ret;
        }

        drain_connect_result_sem();
        wifi_connect_request_active = true;
        ret = issue_wifi_connect_request(iface, use_stored);
        if (ret != 0) {
            wifi_connect_request_active = false;
            return ret;
        }

        if (k_sem_take(&wifi_connect_result_sem, K_SECONDS(WIFI_CONNECT_RESULT_WAIT_SECONDS)) != 0) {
            wifi_connect_request_active = false;
            printk("[WiFi] Connect result timeout\n");
            return -ETIMEDOUT;
        }

        wifi_connect_request_active = false;
        return wifi_last_connect_status;
    }

#if defined(CONFIG_WIFI_CREDENTIALS_STATIC)
    static void set_explicit_connect_profile(uint8_t security, uint8_t band, uint8_t channel)
    {
        wifi_static_connect_params.security = security;
        wifi_static_connect_params.band = band;
        wifi_static_connect_params.channel = channel;
    }
#endif

    static int request_wifi_connect_with_fallback(struct net_if *iface)
    {
        int ret = request_wifi_connect(iface, true);

#if defined(CONFIG_WIFI_CREDENTIALS_STATIC)
        if (ret != 0) {
            printk("[WiFi] Stored connect failed (%d), trying explicit PSK/2.4GHz\n", ret);
            set_explicit_connect_profile(WIFI_SECURITY_TYPE_PSK,
                                         WIFI_FREQ_BAND_2_4_GHZ,
                                         WIFI_CHANNEL_ANY);
            ret = request_wifi_connect(iface, false);
            if (ret != 0) {
                printk("[WiFi] Explicit PSK/2.4GHz failed (%d), trying WPA_AUTO/ANY\n", ret);
                set_explicit_connect_profile(WIFI_SECURITY_TYPE_WPA_AUTO_PERSONAL,
                                             WIFI_FREQ_BAND_UNKNOWN,
                                             WIFI_CHANNEL_ANY);
                ret = request_wifi_connect(iface, false);
            }
        }
#endif

        return ret;
    }

    static bool wifi_iface_connected_now(struct net_if *iface)
    {
        struct wifi_iface_status status = {0};

        if (!iface) {
            return false;
        }

        if (net_mgmt(NET_REQUEST_WIFI_IFACE_STATUS, iface, &status, sizeof(status)) != 0) {
            return false;
        }

        return status.state == WIFI_STATE_COMPLETED;
    }

    #if TINYML_WIFI_DIAGNOSTIC_SCAN || TINYML_WIFI_PRECONNECT_SCAN
    static int run_wifi_scan(struct net_if *iface)
    {
        int ret;

        ret = ensure_wifi_iface_up(iface, "Scan");
        if (ret != 0) {
            return ret;
        }

        while (k_sem_take(&wifi_scan_done_sem, K_NO_WAIT) == 0) {
        }
        wifi_scan_target_seen = false;

        ret = net_mgmt(NET_REQUEST_WIFI_SCAN, iface, NULL, 0);
        if (ret) {
            printk("[WiFi][Scan] request failed: %d\n", ret);
            return ret;
        }

        printk("[WiFi][Scan] pre-connect scan requested\n");
        if (k_sem_take(&wifi_scan_done_sem, K_SECONDS(15)) != 0) {
            printk("[WiFi][Scan] timeout waiting for completion\n");
            return -ETIMEDOUT;
        }

#if defined(CONFIG_WIFI_CREDENTIALS_STATIC)
        if (wifi_scan_target_seen) {
            printk("[WiFi][Scan] target SSID '%s' visible\n",
                   CONFIG_WIFI_CREDENTIALS_STATIC_SSID);
        } else {
            printk("[WiFi][Scan] target SSID '%s' NOT visible\n",
                   CONFIG_WIFI_CREDENTIALS_STATIC_SSID);
        }
#endif
        return 0;
    }
    #endif

    static void wifi_reconnect_thread(void *a, void *b, void *c)
    {
        ARG_UNUSED(a);
        ARG_UNUSED(b);
        ARG_UNUSED(c);

        while (1) {
            struct net_if *iface;
            int ret;
            uint32_t retry_delay_s;

            (void)k_sem_take(&wifi_reconnect_sem, K_FOREVER);
            if (!wifi_runtime_active || wifi_is_connected) {
                continue;
            }

            if (wifi_connect_request_active) {
                continue;
            }

            if (!wifi_ready_status) {
                printk("[WiFi] Backend not ready yet, retry in 5s\n");
                k_sleep(K_SECONDS(5));
                if (!wifi_is_connected && wifi_runtime_active) {
                    (void)k_sem_give(&wifi_reconnect_sem);
                }
                continue;
            }

            iface = net_if_get_first_wifi();
            if (!iface) {
                k_sleep(K_SECONDS(2));
                (void)k_sem_give(&wifi_reconnect_sem);
                continue;
            }

            if (wifi_iface_connected_now(iface)) {
                wifi_is_connected = true;
                wifi_reconnect_failures = 0;
                if (telemetry_sock < 0) {
                    setup_udp_socket();
#if TINYML_TELEMETRY_SEND_STARTUP
                    if (telemetry_sock >= 0 && telemetry_dest_ready) {
                        telemetry_schedule_startup_burst();
                    }
#endif
                }
                continue;
            }

            printk("[WiFi] Reconnect attempt...\n");
#if TINYML_WIFI_DIAGNOSTIC_SCAN || TINYML_WIFI_PRECONNECT_SCAN
            (void)run_wifi_scan(iface);
#endif
            if (wifi_reconnect_failures >= WIFI_RECONNECT_RESET_STREAK) {
                (void)net_mgmt(NET_REQUEST_WIFI_DISCONNECT, iface, NULL, 0);
                k_sleep(K_MSEC(250));
            }
            ret = request_wifi_connect_with_fallback(iface);

            if (ret == 0) {
                printk("[WiFi] Reconnect success\n");
                wifi_reconnect_failures = 0;
                /* Allow link-layer neighbor state to settle before UDP setup. */
                k_sleep(K_SECONDS(2));
                if (telemetry_sock < 0) {
                    setup_udp_socket();
#if TINYML_TELEMETRY_SEND_STARTUP
                    if (telemetry_sock >= 0 && telemetry_dest_ready) {
                        telemetry_schedule_startup_burst();
                    }
#endif
                }
            } else {
                wifi_reconnect_failures++;
                retry_delay_s = (ret == -ETIMEDOUT || ret == -116) ?
                    WIFI_RECONNECT_FAST_RETRY_SECONDS : WIFI_RECONNECT_SLOW_RETRY_SECONDS;
                if (wifi_reconnect_failures >= WIFI_RECONNECT_RESET_STREAK) {
                    retry_delay_s = WIFI_RECONNECT_SLOW_RETRY_SECONDS;
                }
                printk("[WiFi] Reconnect failed: %d (streak=%u, retry in %us)\n",
                       ret, wifi_reconnect_failures, retry_delay_s);
                k_sleep(K_SECONDS(retry_delay_s));
                if (!wifi_is_connected && wifi_runtime_active) {
                    (void)k_sem_give(&wifi_reconnect_sem);
                }
            }
        }
    }

    static void wifi_mgmt_event_handler(struct net_mgmt_event_callback *cb,
                                        uint32_t mgmt_event, struct net_if *iface)
    {
        switch (mgmt_event) {
        case NET_EVENT_WIFI_CONNECT_RESULT: {
            const struct wifi_status *status = (const struct wifi_status *)cb->info;
            bool request_active = wifi_connect_request_active;
            bool connected_now = wifi_iface_connected_now(iface);
            if (!status) {
                LOG_WRN("WiFi connect result event with null status");
                break;
            }
            wifi_last_connect_status = status->status;
            if (status->status) {
                if (!request_active) {
                    if (connected_now || wifi_is_connected) {
                        /* Ignore transient background failures while link is up. */
                        LOG_WRN("Ignoring late connect failure event: %d", status->status);
                        break;
                    }

                    /* Background attempt failed while link is down: drive recovery. */
                    wifi_is_connected = false;
                    telemetry_reset_socket_state();
                    LOG_WRN("Background WiFi connect failed: %d (retry scheduled)",
                            status->status);
                    schedule_wifi_reconnect(K_SECONDS(2));
                    break;
                }
                LOG_ERR("WiFi connection failed: %d", status->status);
                printk("[WiFi] Connect failed: %d\n", status->status);
                /*
                 * Do not schedule reconnect from this event. The caller path
                 * (init or reconnect thread) owns retry cadence to avoid
                 * overlapping connect storms.
                 */
            } else {
                bool state_changed = !wifi_is_connected;
                wifi_is_connected = true;
                wifi_reconnect_failures = 0;
                LOG_INF("WiFi connected!");
                if (state_changed) {
                    printk("[WiFi] Connected\n");
                }
            }
            if (request_active) {
                k_sem_give(&wifi_connect_result_sem);
            }
            break;
        }
        case NET_EVENT_WIFI_DISCONNECT_RESULT:
            wifi_is_connected = false;
            LOG_INF("WiFi disconnected");
            printk("[WiFi] Disconnected\n");
            telemetry_reset_socket_state();
            schedule_wifi_reconnect(K_SECONDS(1));
            break;
        case NET_EVENT_WIFI_SCAN_RESULT: {
            const struct wifi_scan_result *entry = (const struct wifi_scan_result *)cb->info;
#if TINYML_WIFI_DIAGNOSTIC_SCAN
            printk("[WiFi][Scan] SSID=%.*s ch=%u rssi=%d sec=%s\n",
                   (int)entry->ssid_length,
                   entry->ssid,
                   entry->channel,
                   entry->rssi,
                   wifi_security_txt(entry->security));
#endif
#if defined(CONFIG_WIFI_CREDENTIALS_STATIC)
            size_t target_len = strlen(CONFIG_WIFI_CREDENTIALS_STATIC_SSID);
            if (entry->ssid_length == target_len &&
                memcmp(entry->ssid, CONFIG_WIFI_CREDENTIALS_STATIC_SSID, target_len) == 0) {
                wifi_scan_target_seen = true;
            }
#endif
            break;
        }
        case NET_EVENT_WIFI_SCAN_DONE:
#if TINYML_WIFI_DIAGNOSTIC_SCAN
            printk("[WiFi][Scan] done\n");
#endif
            k_sem_give(&wifi_scan_done_sem);
            break;
        default:
            break;
        }
    }

    static void telemetry_reset_socket_state(void)
    {
        (void)k_work_cancel_delayable(&telemetry_startup_work);
        telemetry_startup_retries_left = 0;
        if (telemetry_sock >= 0) {
            zsock_close(telemetry_sock);
        }
        telemetry_sock = -1;
        telemetry_dest_ready = false;
        telemetry_peer_connected = false;
        telemetry_transient_failures = 0;
    }

    static bool telemetry_maybe_connect_peer(void)
    {
        int ret;
        int saved_errno = 0;
        int64_t now_ms = k_uptime_get();

        if (telemetry_sock < 0 || !telemetry_dest_ready) {
            return false;
        }

        if ((now_ms - telemetry_last_connect_attempt_ms) < TELEMETRY_CONNECT_RETRY_MS) {
            return false;
        }

        telemetry_last_connect_attempt_ms = now_ms;
        ret = zsock_connect(telemetry_sock,
                            (struct sockaddr *)&telemetry_dest,
                            sizeof(telemetry_dest));
        saved_errno = errno;

        if (ret == 0 || saved_errno == EISCONN) {
            telemetry_peer_connected = true;
            return true;
        }

        if (saved_errno == EINPROGRESS || saved_errno == EALREADY) {
            return false;
        }

        LOG_WRN("UDP connect retry failed: errno=%d", saved_errno);
        return false;
    }

    static void telemetry_trigger_wifi_reconnect(int err, const char *reason)
    {
        int64_t now_ms = k_uptime_get();
        struct net_if *iface = net_if_get_first_wifi();

        if ((now_ms - telemetry_last_wifi_recover_ms) < TELEMETRY_WIFI_RECOVER_COOLDOWN_MS) {
            return;
        }
        telemetry_last_wifi_recover_ms = now_ms;

        if (iface && wifi_iface_connected_now(iface)) {
            (void)net_mgmt(NET_REQUEST_WIFI_DISCONNECT, iface, NULL, 0);
        }
        wifi_is_connected = false;
        telemetry_reset_socket_state();
        LOG_WRN("Forcing WiFi reconnect (%s, errno=%d)", reason, err);
        printk("[WiFi] Force reconnect (%s errno=%d)\n", reason, err);
        schedule_wifi_reconnect(K_NO_WAIT);
    }

    static void setup_udp_socket(void) {
        int ret;
        int saved_errno = 0;

        telemetry_reset_socket_state();
        telemetry_last_socket_refresh_ms = k_uptime_get();

        telemetry_sock = zsock_socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (telemetry_sock < 0) {
            LOG_ERR("Socket creation failed");
            printk("[UDP] Socket create failed errno=%d\n", errno);
            telemetry_peer_connected = false;
            return;
        }

        telemetry_dest = (struct sockaddr_in){
            .sin_family = AF_INET,
            .sin_port = htons(TELEMETRY_PORT)
        };
        if (zsock_inet_pton(AF_INET, TELEMETRY_IP, &telemetry_dest.sin_addr) <= 0) {
            LOG_ERR("Invalid telemetry IP: %s", TELEMETRY_IP);
            printk("[UDP] Invalid TELEMETRY_IP: %s\n", TELEMETRY_IP);
            zsock_close(telemetry_sock);
            telemetry_sock = -1;
            telemetry_dest_ready = false;
            telemetry_peer_connected = false;
            return;
        }

        /*
         * Connect UDP socket to force route/neighbor resolution state early
         * in static-IP mode.
         */
        ret = zsock_connect(telemetry_sock,
                            (struct sockaddr *)&telemetry_dest,
                            sizeof(telemetry_dest));
        saved_errno = errno;
        telemetry_last_connect_attempt_ms = k_uptime_get();

        if (ret < 0 && saved_errno != EINPROGRESS &&
            saved_errno != EALREADY && saved_errno != EISCONN) {
            LOG_ERR("UDP connect failed: errno=%d", saved_errno);
            printk("[UDP] connect() failed errno=%d\n", saved_errno);
            telemetry_reset_socket_state();
            return;
        }

        telemetry_peer_connected = (ret == 0 || saved_errno == EISCONN);
        telemetry_dest_ready = true;
        if (telemetry_peer_connected) {
        LOG_INF("UDP socket ready for %s:%d", TELEMETRY_IP, TELEMETRY_PORT);
        printk("[UDP] Socket connected to %s:%d\n", TELEMETRY_IP, TELEMETRY_PORT);
        } else {
            LOG_INF("UDP socket pending connect for %s:%d (errno=%d)",
                    TELEMETRY_IP, TELEMETRY_PORT, saved_errno);
            printk("[UDP] Socket pending connect errno=%d\n", saved_errno);
        }
    }

    static void telemetry_startup_work_handler(struct k_work *work)
    {
        ARG_UNUSED(work);

        if (!wifi_is_connected || telemetry_sock < 0 || !telemetry_dest_ready) {
            return;
        }

        send_telemetry_event("WIFI_READY");

        if (telemetry_startup_retries_left > 0) {
            telemetry_startup_retries_left--;
        }

        if (telemetry_startup_retries_left > 0) {
            (void)k_work_reschedule(&telemetry_startup_work,
                                    K_MSEC(TELEMETRY_STARTUP_RETRY_INTERVAL_MS));
        }
    }

    static void telemetry_schedule_startup_burst(void)
    {
        if (!wifi_is_connected) {
            return;
        }

        if (telemetry_sock < 0 || !telemetry_dest_ready) {
            setup_udp_socket();
        }

        if (telemetry_sock < 0 || !telemetry_dest_ready) {
            return;
        }

        telemetry_startup_retries_left = TELEMETRY_STARTUP_RETRY_COUNT;
        (void)k_work_reschedule(&telemetry_startup_work, K_NO_WAIT);
    }

#if defined(CONFIG_WIFI_READY_LIB)
    static void wifi_ready_state_cb(bool wifi_ready)
    {
        wifi_ready_status = wifi_ready;
        k_sem_give(&wifi_ready_sem);
    }

    static int wait_for_wifi_ready(struct net_if *iface)
    {
        wifi_ready_callback_t cb = {
            .wifi_ready_cb = wifi_ready_state_cb,
        };
        int ret;

        ret = register_wifi_ready_callback(cb, iface);
        if (ret) {
            LOG_ERR("Failed to register WiFi ready callback: %d", ret);
            return ret;
        }

        LOG_INF("Waiting for WiFi ready event...");
        printk("[WiFi] Waiting ready event...\n");
        ret = k_sem_take(&wifi_ready_sem, K_SECONDS(WIFI_READY_WAIT_SECONDS));
        if (ret) {
            LOG_WRN("WiFi ready event timeout");
            printk("[WiFi] Ready timeout, deferring connect\n");
            return -ETIMEDOUT;
        }

        if (!wifi_ready_status) {
            LOG_ERR("WiFi logged not ready");
            return -1;
        }

        LOG_INF("WiFi ready");
        printk("[WiFi] Ready\n");
        return 0;
    }
#endif

    static void send_telemetry_event(const char* msg) {
        int ret;
        int err;
        int64_t now_ms;
        struct net_if *iface;
        bool prolonged_outage;

        if (!wifi_is_connected) {
            printk("[UDP] Drop (wifi not connected): %s\n", msg);
            return;
        }

        if (telemetry_sock < 0 || !telemetry_dest_ready) {
            setup_udp_socket();
        }
        if (telemetry_sock < 0 || !telemetry_dest_ready) {
            printk("[UDP] Drop (socket not ready): %s\n", msg);
            return;
        }

        ret = zsock_send(telemetry_sock, msg, strlen(msg), 0);
        if (ret > 0) {
            telemetry_peer_connected = true;
            telemetry_transient_failures = 0;
            telemetry_last_success_ms = k_uptime_get();
            LOG_INF("Telemetry sent: %s (%d bytes)", msg, ret);
            printk("[UDP] Sent: %s (%d bytes)\n", msg, ret);
            return;
        }

        err = errno;

        if (err == EINPROGRESS || err == EALREADY ||
            err == EAGAIN || err == EWOULDBLOCK || err == ENETDOWN ||
            err == ENOTCONN) {
            telemetry_transient_failures++;
            now_ms = k_uptime_get();

            if (telemetry_transient_failures == 1U ||
                (telemetry_transient_failures % 10U) == 0U) {
                LOG_WRN("Telemetry transient send failure: errno=%d streak=%u",
                        err, telemetry_transient_failures);
                printk("[UDP] Pending send errno=%d streak=%u\n",
                       err, telemetry_transient_failures);
            }

            (void)telemetry_maybe_connect_peer();
            if ((err == ENETDOWN || err == ENOTCONN) &&
                telemetry_transient_failures >= TELEMETRY_TRANSIENT_STREAK_WIFI_RECOVER) {
                prolonged_outage =
                    telemetry_last_success_ms <= 0 ||
                    (now_ms - telemetry_last_success_ms) >=
                        TELEMETRY_WIFI_RECOVER_MIN_OUTAGE_MS;
                if (prolonged_outage) {
                    telemetry_trigger_wifi_reconnect(err, "transient-send");
                    return;
                }
            }
            if (telemetry_transient_failures >= TELEMETRY_TRANSIENT_STREAK_WIFI_RECOVER) {
                iface = net_if_get_first_wifi();
                if (!wifi_iface_connected_now(iface)) {
                    telemetry_trigger_wifi_reconnect(err, "iface-status");
                    return;
                }
            }

            if (telemetry_transient_failures >= TELEMETRY_TRANSIENT_STREAK_REFRESH &&
                (now_ms - telemetry_last_socket_refresh_ms) >= TELEMETRY_SOCKET_REFRESH_MS) {
                LOG_WRN("Refreshing UDP socket after transient failure streak (%u)",
                        telemetry_transient_failures);
                setup_udp_socket();
            }
            return;
        }

        LOG_ERR("Telemetry send failed: %d (errno=%d)", ret, err);
        printk("[UDP] Send failed ret=%d errno=%d\n", ret, err);

        if (err == ENETUNREACH || err == EHOSTUNREACH) {
            /*
             * Peer may be temporarily unreachable (e.g., hotspot screen off).
             * Keep socket state and let retry path recover.
             */
            return;
        }

        if (err == EPIPE || err == ECONNRESET || err == ENOTSOCK) {
            telemetry_reset_socket_state();
            /* Keep WiFi reconnect decisions in NET_EVENT_WIFI_DISCONNECT_RESULT only. */
            if (wifi_is_connected) {
                setup_udp_socket();
            }
        }
    }

    static int init_network_stack(void) {
        uint32_t wifi_events = NET_EVENT_WIFI_CONNECT_RESULT |
                               NET_EVENT_WIFI_DISCONNECT_RESULT;

        LOG_INF("=== WiFi Init ===");
        printk("[WiFi] Init start\n");
        /*
         * Keep runtime active from init start so reconnect can continue in
         * background even if hotspot association is delayed at boot.
         */
        wifi_runtime_active = true;
        wifi_connect_request_active = false;

#if TINYML_WIFI_DIAGNOSTIC_SCAN || TINYML_WIFI_PRECONNECT_SCAN
        wifi_events |= NET_EVENT_WIFI_SCAN_RESULT | NET_EVENT_WIFI_SCAN_DONE;
#endif

        net_mgmt_init_event_callback(&wifi_mgmt_cb, wifi_mgmt_event_handler, wifi_events);
        net_mgmt_add_event_callback(&wifi_mgmt_cb);

        struct net_if *iface = net_if_get_first_wifi();
        if (!iface) {
            LOG_ERR("No WiFi interface");
            printk("[WiFi] No interface\n");
            return -1;
        }

        if (ensure_wifi_iface_up(iface, "Init") != 0) {
            return -1;
        }

        LOG_INF("WiFi interface found");
        print_wifi_iface_status(iface, "Initial");
#if defined(CONFIG_WIFI_READY_LIB)
        if (wait_for_wifi_ready(iface) != 0) {
            LOG_WRN("WiFi backend not ready during init; deferring connect");
            schedule_wifi_reconnect(K_SECONDS(5));
            return 0;
        }
#else
        k_msleep(1000);
#endif
#if TINYML_WIFI_DIAGNOSTIC_SCAN || TINYML_WIFI_PRECONNECT_SCAN
        (void)run_wifi_scan(iface);
#endif

        if (!wifi_iface_connected_now(iface)) {
            int ret = request_wifi_connect_with_fallback(iface);

            if (ret != 0) {
                LOG_WRN("Initial WiFi connect failed; deferring to background reconnect");
                printk("[WiFi] Initial connect failed: %d, deferring reconnect\n", ret);
                schedule_wifi_reconnect(K_NO_WAIT);
                return 0;
            }
        }

        if (!wifi_iface_connected_now(iface)) {
            LOG_INF("WiFi connect still pending; deferring to background reconnect thread");
            printk("[WiFi] Initial connect still pending, deferring reconnect\n");
            schedule_wifi_reconnect(K_NO_WAIT);
            return 0;
        }

        wifi_is_connected = true;

        LOG_INF("Connected; initializing UDP telemetry path");
        printk("[WiFi] Connected\n");

        /* L2 stabilization delay (ARP/neighbor readiness) before UDP connect. */
        k_msleep(2000);
        setup_udp_socket();

        #if TINYML_TELEMETRY_SEND_STARTUP
        LOG_INF("Sending test packet...");
        telemetry_schedule_startup_burst();
#else
        LOG_INF("Startup telemetry disabled by TINYML_TELEMETRY_SEND_STARTUP=0");
#endif

        LOG_INF("=== WiFi Ready ===");
        return 0;
    }

#else
    static void send_telemetry_event(const char *msg)
    {
        ARG_UNUSED(msg);
    }
#endif
#endif

static void fill_noise_buffer(int16_t* out, uint32_t seed) {
    for (int i = 0; i < NUM_SAMPLES; i++) {
        seed = seed * 1103515245u + 12345u;
        out[i] = (int16_t)((seed >> 16) & 0x1FF) - 256;
    }
}

#if !defined(TEST_MODE) || TINYML_POWER_NN_ONLY
static void fill_tone_buffer(int16_t* out, float frequency_hz, float amplitude) {
    float scale = 32767.0f * amplitude;
    float step = 2.0f * PI * frequency_hz / (float)SAMPLE_RATE;
    for (int i = 0; i < NUM_SAMPLES; i++) {
        out[i] = (int16_t)(scale * sinf(step * (float)i));
    }
}
#endif

static float max_feature_value(const float *features, size_t count)
{
    float peak = 0.0f;

    for (size_t i = 0; i < count; i++) {
        if (features[i] > peak) {
            peak = features[i];
        }
    }

    return peak;
}

static int compute_spectrogram_with_runtime_fallback(const int16_t *audio_in, float *features_out)
{
#ifdef TINYML_DUAL_CORE_DSP_OFFLOAD
    int ret = dual_core_generate_spectrogram(audio_in, features_out);

    if (ret == 0) {
        return 0;
    }

    printk("WARN: Dual-core DSP offload unavailable, falling back to local DSP\n");
    return generate_spectrogram(audio_in, features_out);
#else
    return generate_spectrogram(audio_in, features_out);
#endif
}

#if TINYML_POWER_NN_ONLY
static int prepare_cached_nn_features(void)
{
    fill_tone_buffer(audio_buffer, 1000.0f, 0.5f);
    dma_cache_prepare_for_cpu(audio_buffer, sizeof(audio_buffer));

    if (compute_spectrogram_with_runtime_fallback(audio_buffer, features_buffer) != 0) {
        return -1;
    }

    dma_cache_prepare_for_device(features_buffer, sizeof(features_buffer));
    nn_only_features_ready = true;
    return 0;
}
#endif

static void process_audio(const char* label, const int16_t* input_data, bool expect_target) {
#ifndef TEST_MODE
    ARG_UNUSED(expect_target);
#endif

    printk("\nTest: %s\n", label != NULL ? label : "Frame");
    uint32_t pipeline_start_cycles = k_cycle_get_32();
    uint32_t dsp_end_cycles = pipeline_start_cycles;

    if (TINYML_POWER_NN_ONLY == 0 && input_data == NULL) {
        // Synthetic low-level noise for deterministic non-target testing.
        fill_noise_buffer(audio_buffer, 42u);
    } else if (TINYML_POWER_NN_ONLY == 0 && input_data != audio_buffer) {
        memcpy(audio_buffer, input_data, sizeof(audio_buffer));
    }

    int ret = 0;

    if (TINYML_POWER_NN_ONLY == 0) {
        dma_cache_prepare_for_cpu(audio_buffer, sizeof(audio_buffer));

        ret = compute_spectrogram_with_runtime_fallback(audio_buffer, features_buffer);
        if (ret != 0) {
            printk("ERROR: Spectrogram computation failed\n");
            return;
        }

        dma_cache_prepare_for_device(features_buffer, sizeof(features_buffer));

        float peak = max_feature_value(features_buffer, SPECTROGRAM_SIZE);
        dsp_end_cycles = k_cycle_get_32();

        if (peak < SIGNAL_NOISE_FLOOR) {
            printk("Below noise floor (peak=%.2f)\n", (double)peak);
#if TINYML_PRINT_STAGE_TIMINGS
            uint32_t dsp_duration_us = k_cyc_to_us_near32(dsp_end_cycles - pipeline_start_cycles);
            printk("PHASE_TIMING dsp_us=%u nn_us=0 total_us=%u skipped_nn=1\n",
                   dsp_duration_us,
                   dsp_duration_us);
#endif
            gpio_pin_set_dt(&led_target, 0);
            return;
        }

        if (TINYML_POWER_DSP_ONLY) {
            uint32_t dsp_duration_us = k_cyc_to_us_near32(dsp_end_cycles - pipeline_start_cycles);
#if TINYML_PRINT_STAGE_TIMINGS
            printk("PHASE_TIMING dsp_us=%u nn_us=0 total_us=%u dsp_only=1\n",
                   dsp_duration_us,
                   dsp_duration_us);
#endif
            printk("DSP only: %.3f ms (%u us)\n",
                   (double)((float)dsp_duration_us / 1000.0f),
                   dsp_duration_us);
            gpio_pin_set_dt(&led_target, 0);
            return;
        }
    } else {
        if (!nn_only_features_ready) {
            printk("ERROR: cached NN-only features not ready\n");
            return;
        }
        dsp_end_cycles = pipeline_start_cycles;
    }

    gpio_pin_set_dt(&pin_timing, 1);
    uint32_t t0 = k_cycle_get_32();

    float probability = 0.0f;
    ret = run_inference(features_buffer, &probability);

    uint32_t t1 = k_cycle_get_32();
    gpio_pin_set_dt(&pin_timing, 0);

    if (ret != 0) {
        printk("ERROR: Inference failed\n");
        return;
    }

    uint32_t dsp_duration_us = k_cyc_to_us_near32(dsp_end_cycles - pipeline_start_cycles);
    uint32_t duration_us = k_cyc_to_us_near32(t1 - t0);
    uint32_t pipeline_duration_us = k_cyc_to_us_near32(t1 - pipeline_start_cycles);
#if TINYML_PRINT_STAGE_TIMINGS
    printk("PHASE_TIMING dsp_us=%u nn_us=%u total_us=%u\n",
           dsp_duration_us,
           duration_us,
           pipeline_duration_us);
#endif
    printk("Inference: %.3f ms (%u us) | Confidence: %.1f%%\n",
           (double)((float)duration_us / 1000.0f),
           duration_us,
           (double)(probability * 100.0f));

#ifdef WIFI_TELEMETRY_ENABLED
    #if TINYML_TELEMETRY_SEND_PER_FRAME
    send_telemetry_event("INFERENCE_FRAME");
    #endif
#endif

    if (probability > CONFIDENCE_THRESHOLD) {
        gpio_pin_set_dt(&led_target, 1);
        #ifdef WIFI_TELEMETRY_ENABLED
        #if TINYML_TELEMETRY_SEND_TARGET
        send_telemetry_event("TARGET_DETECTED");
        #endif
        #endif

        #ifdef TEST_MODE
        if (expect_target) {
            printk(">>> TARGET DETECTED <<<\n");
        } else {
            printk("False Positive\n");
        }
        #else
        printk(">>> TARGET DETECTED <<<\n");
        #endif
    } else {
        gpio_pin_set_dt(&led_target, 0);
        #ifdef TEST_MODE
        if (expect_target) {
            printk("Missed detection\n");
        } else {
            printk("Correctly ignored\n");
        }
        #else
        printk("Below threshold\n");
        #endif
    }
}

#ifdef TEST_MODE
static void run_test_loop(void) {
    printk("=== TEST MODE: Running validation loop ===\n");
    printk("Running deterministic validation fixtures.\n\n");

    while (1) {
        process_audio("Silence", NULL, false);
        k_msleep(2000);

        process_audio("Interference (Cat)", audio_cat2, false);
        k_msleep(2000);

        process_audio("Target (ON)", audio_on2, true);
        k_msleep(2000);
    }
}
#endif

int main(void) {
    nrf_cache_enable(NRF_CACHE);

    (void)init_console_transport();

    if (!device_is_ready(led_target.port) || !device_is_ready(pin_timing.port)) {
        printk("ERROR: GPIO devices not ready\n");
        return 0;
    }

    gpio_pin_configure_dt(&led_target, GPIO_OUTPUT_ACTIVE);
    gpio_pin_configure_dt(&pin_timing, GPIO_OUTPUT_ACTIVE);
    gpio_pin_set_dt(&led_target, 0);
    gpio_pin_set_dt(&pin_timing, 0);

    printk("\nnRF Embedded ML Benchmarks\n");
    printk("Config: FFT=%d Mel=%d Frames=%d TensorArena=%uKB\n",
           NUM_FFT_BINS, NUM_MEL_BINS, SPECTROGRAM_ROWS,
           inference_tensor_arena_size() / 1024U);
#ifdef TINYML_DUAL_CORE_DSP_OFFLOAD
    printk("DSP mode: dual-core offload (cpuapp -> cpunet)\n\n");
#else
    printk("DSP mode: local cpuapp\n\n");
#endif

    #if defined(EXPORT_C_REFERENCE_SPECTROGRAM) || defined(EXPORT_QUANTIZED_REFERENCE)
    printk("Reference export mode: WiFi init skipped\n\n");
    printk("Reference export mode: waiting 2s for UART host sync\n");
    k_msleep(2000);

    #if defined(EXPORT_QUANTIZED_REFERENCE)
    printk("Quantized reference mode: initializing AI...\n");
    if (inference_setup() != 0) {
        printk("ERROR: AI initialization failed in quantized export mode\n");
        return 0;
    }
    #endif

    while (1) {
        #if defined(EXPORT_C_REFERENCE_SPECTROGRAM)
        export_c_reference_only();
        #endif
        #if defined(EXPORT_QUANTIZED_REFERENCE)
        export_quantized_reference_only();
        #endif
        // Keep re-emitting so host capture can attach and parse a full block.
        k_msleep(5000);
    }
    #else
    printk("Initializing AI...\n");
    if (inference_setup() != 0) {
        printk("ERROR: AI initialization failed!\n");
        return 0;
    }
    printk("AI ready!\n\n");

#if TINYML_POWER_NN_ONLY
    if (prepare_cached_nn_features() != 0) {
        printk("ERROR: NN-only feature cache preparation failed\n");
        return 0;
    }
    printk("NN-only feature cache ready!\n\n");
#endif

#ifdef TINYML_DUAL_CORE_DSP_OFFLOAD
    if (dual_core_dsp_init() != 0) {
        printk("ERROR: Dual-core DSP IPC init failed\n");
        return 0;
    }
    printk("Dual-core DSP IPC ready!\n\n");
#endif

    // Keep WiFi association from delaying deterministic inference startup.
    #if defined(WIFI_TELEMETRY_ENABLED) && (TINYML_WIFI_RUNTIME == 1)
    if (init_network_stack() != 0) {
        printk("WiFi failed - continuing without telemetry\n\n");
    }
    #elif defined(WIFI_TELEMETRY_ENABLED) && (TINYML_WIFI_RUNTIME == 0)
    printk("WiFi runtime disabled by TINYML_WIFI_RUNTIME=0\n\n");
    #endif

    #ifdef TEST_MODE
    run_benchmarks();
    run_test_loop();
    #else
    printk("=== PRODUCTION MODE ===\n");
    printk("Running deterministic digital input (1 sec frames, synthetic source)\n");
    printk("Mode: wifi_runtime=%d startup_tx=%d event_tx=%d per_frame_tx=%d\n\n",
           TINYML_WIFI_RUNTIME, TINYML_TELEMETRY_SEND_STARTUP,
           TINYML_TELEMETRY_SEND_TARGET, TINYML_TELEMETRY_SEND_PER_FRAME);
#if TINYML_POWER_DSP_ONLY
    printk("Power profile: DSP-only\n\n");
#elif TINYML_POWER_NN_ONLY
    printk("Power profile: NN-only\n\n");
#endif

    while (1) {
        fill_tone_buffer(audio_buffer, 1000.0f, 0.5f);
        process_audio("Generated Tone", audio_buffer, false);

        k_msleep(1000);
    }
    #endif
    #endif

    return 0;
}
