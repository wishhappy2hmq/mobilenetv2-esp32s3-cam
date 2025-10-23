/*
 * ESP32-S3 MobileNetV2 实时分类系统
 *
 * 功能：
 * - WiFi连接
 * - 实时摄像头采集
 * - HTTP视频流服务器
 * - 每10帧进行一次图像分类推理
 * - Web界面显示分类结果
 *
 * 模型：image_classifier_320x320_int8.espdl (二分类)
 */

#include <string.h>
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_http_server.h"
#include "esp_camera.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "freertos/semphr.h"
#include "imagenet_cls.hpp"
#include "dl_image.hpp"
#include "dl_image_jpeg.hpp"

static const char *TAG = "mobilenetv2_cls";

// ============================================================================
// 配置参数 - 请修改为你的实际配置
// ============================================================================

// WiFi配置
#define WIFI_SSID      "***"      // 修改为你的WiFi名称
#define WIFI_PASSWORD  "***"  // 修改为你的WiFi密码

// 推理配置
#define INFERENCE_INTERVAL 10  // 每10帧推理一次

// 摄像头引脚配置
#define CAM_PIN_PWDN    -1
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK    15
#define CAM_PIN_SIOD    4
#define CAM_PIN_SIOC    5
#define CAM_PIN_D7      16
#define CAM_PIN_D6      17
#define CAM_PIN_D5      18
#define CAM_PIN_D4      12
#define CAM_PIN_D3      10
#define CAM_PIN_D2      8
#define CAM_PIN_D1      9
#define CAM_PIN_D0      11
#define CAM_PIN_VSYNC   6
#define CAM_PIN_HREF    7
#define CAM_PIN_PCLK    13

// ============================================================================
// WiFi管理
// ============================================================================

static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT      BIT1
static int s_retry_num = 0;
#define MAX_RETRY 5

static void wifi_event_handler(void* arg, esp_event_base_t event_base,
                                int32_t event_id, void* event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_retry_num < MAX_RETRY) {
            esp_wifi_connect();
            s_retry_num++;
            ESP_LOGI(TAG, "Retry connecting to WiFi... (%d/%d)", s_retry_num, MAX_RETRY);
        } else {
            xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
            ESP_LOGE(TAG, "Failed to connect to WiFi");
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "Got IP address: " IPSTR, IP2STR(&event->ip_info.ip));
        s_retry_num = 0;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

static esp_err_t wifi_init(void)
{
    // 初始化NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                         ESP_EVENT_ANY_ID,
                                                         &wifi_event_handler,
                                                         NULL,
                                                         &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                         IP_EVENT_STA_GOT_IP,
                                                         &wifi_event_handler,
                                                         NULL,
                                                         &instance_got_ip));

    wifi_config_t wifi_config = {};
    strcpy((char*)wifi_config.sta.ssid, WIFI_SSID);
    strcpy((char*)wifi_config.sta.password, WIFI_PASSWORD);
    wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Connecting to SSID: %s", WIFI_SSID);

    // 等待连接结果
    EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group,
                                            WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
                                            pdFALSE,
                                            pdFALSE,
                                            portMAX_DELAY);

    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "Connected to WiFi successfully");
        return ESP_OK;
    } else if (bits & WIFI_FAIL_BIT) {
        ESP_LOGE(TAG, "Failed to connect to WiFi");
        return ESP_FAIL;
    }

    return ESP_ERR_TIMEOUT;
}

// ============================================================================
// 摄像头管理
// ============================================================================

static esp_err_t camera_init(void)
{
    camera_config_t config = {
        .pin_pwdn = CAM_PIN_PWDN,
        .pin_reset = CAM_PIN_RESET,
        .pin_xclk = CAM_PIN_XCLK,
        .pin_sscb_sda = CAM_PIN_SIOD,
        .pin_sscb_scl = CAM_PIN_SIOC,

        .pin_d7 = CAM_PIN_D7,
        .pin_d6 = CAM_PIN_D6,
        .pin_d5 = CAM_PIN_D5,
        .pin_d4 = CAM_PIN_D4,
        .pin_d3 = CAM_PIN_D3,
        .pin_d2 = CAM_PIN_D2,
        .pin_d1 = CAM_PIN_D1,
        .pin_d0 = CAM_PIN_D0,
        .pin_vsync = CAM_PIN_VSYNC,
        .pin_href = CAM_PIN_HREF,
        .pin_pclk = CAM_PIN_PCLK,

        .xclk_freq_hz = 20000000,
        .ledc_timer = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,

        .pixel_format = PIXFORMAT_JPEG,
        .frame_size = FRAMESIZE_QVGA,     // 320x240
        .jpeg_quality = 12,
        .fb_count = 2,
        .fb_location = CAMERA_FB_IN_PSRAM,
        .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
        .sccb_i2c_port = 1,
    };

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed with error 0x%x", err);
        return err;
    }

    // 调整图像参数
    sensor_t *s = esp_camera_sensor_get();
    if (s != NULL) {
        s->set_brightness(s, 0);
        s->set_contrast(s, 0);
        s->set_saturation(s, 0);
        s->set_special_effect(s, 0);
        s->set_whitebal(s, 1);
        s->set_awb_gain(s, 1);
        s->set_wb_mode(s, 0);
        s->set_exposure_ctrl(s, 1);
        s->set_aec2(s, 0);
        s->set_ae_level(s, 0);
        s->set_aec_value(s, 300);
        s->set_gain_ctrl(s, 1);
        s->set_agc_gain(s, 0);
        s->set_gainceiling(s, (gainceiling_t)0);
        s->set_bpc(s, 0);
        s->set_wpc(s, 1);
        s->set_raw_gma(s, 1);
        s->set_lenc(s, 1);
        s->set_hmirror(s, 0);
        s->set_vflip(s, 0);
        s->set_dcw(s, 1);
        s->set_colorbar(s, 0);
    }

    ESP_LOGI(TAG, "Camera initialized successfully");
    return ESP_OK;
}

// ============================================================================
// HTTP服务器
// ============================================================================

static httpd_handle_t stream_httpd = NULL;

// 详细的分类信息结构
struct ClassificationInfo {
    float none_score;
    float exist_score;
    float inference_time_ms;
    char result_text[64];
};

static ClassificationInfo g_classification_info = {
    .none_score = 0.0f,
    .exist_score = 0.0f,
    .inference_time_ms = 0.0f,
    .result_text = "Initializing..."
};
static SemaphoreHandle_t g_classification_mutex = NULL;

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// 更新详细分类结果
static void update_classification(float none_score, float exist_score, float inference_time_ms, const char *result_text)
{
    if (g_classification_mutex && xSemaphoreTake(g_classification_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
        g_classification_info.none_score = none_score;
        g_classification_info.exist_score = exist_score;
        g_classification_info.inference_time_ms = inference_time_ms;
        strncpy(g_classification_info.result_text, result_text, sizeof(g_classification_info.result_text) - 1);
        g_classification_info.result_text[sizeof(g_classification_info.result_text) - 1] = '\0';
        xSemaphoreGive(g_classification_mutex);
    }
}

// MJPEG流处理器
static esp_err_t jpg_stream_handler(httpd_req_t *req)
{
    camera_fb_t *fb = NULL;
    esp_err_t res = ESP_OK;
    char part_buf[64];

    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (res != ESP_OK) {
        return res;
    }

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    while (true) {
        fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGE(TAG, "Camera capture failed");
            res = ESP_FAIL;
            break;
        }

        if (fb->format != PIXFORMAT_JPEG) {
            ESP_LOGE(TAG, "Non-JPEG format not supported");
            esp_camera_fb_return(fb);
            res = ESP_FAIL;
            break;
        }

        // 发送HTTP multipart头
        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        }
        if (res == ESP_OK) {
            size_t hlen = snprintf(part_buf, 64, _STREAM_PART, fb->len);
            res = httpd_resp_send_chunk(req, part_buf, hlen);
        }
        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
        }

        esp_camera_fb_return(fb);

        if (res != ESP_OK) {
            break;
        }
    }

    return res;
}

// 主页处理器
static esp_err_t index_handler(httpd_req_t *req)
{
    const char html[] = R"html(
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ESP32-S3 MobileNetV2 分类</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            background-color: #f0f0f0;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
        }
        #stream {
            width: 100%;
            max-width: 640px;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 5px;
        }
        .classification-info {
            margin-top: 20px;
            padding: 15px;
            background: #e8f5e9;
            border-radius: 5px;
            font-size: 18px;
        }
        .status {
            margin: 10px 0;
            padding: 10px;
            background: #fff3e0;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 ESP32-S3 MobileNetV2 实时分类</h1>
        <div class="status">
            <strong>状态:</strong> <span id="status">连接中...</span>
        </div>
        <img id="stream" src="/stream">
        <div class="classification-info">
            <div><strong>📊 分类结果:</strong></div>
            <div id="classification" style="font-size: 24px; margin-top: 10px; color: #2e7d32;">等待推理...</div>
        </div>
        <div style="margin-top: 20px; color: #666; font-size: 14px;">
            <p>🔄 每10帧进行一次推理</p>
            <p>📷 图像尺寸: 320x240 → 320x320 (模型输入)</p>
        </div>
    </div>
    <script>
        document.getElementById('stream').onload = function() {
            document.getElementById('status').textContent = '已连接 ✓';
            document.getElementById('status').style.color = 'green';
        };

        setInterval(function() {
            fetch('/classification')
                .then(response => response.text())
                .then(data => {
                    document.getElementById('classification').textContent = data;
                })
                .catch(err => console.error('Error:', err));
        }, 500);
    </script>
</body>
</html>
)html";

    httpd_resp_set_type(req, "text/html");
    httpd_resp_set_hdr(req, "Content-Encoding", "identity");
    return httpd_resp_send(req, html, strlen(html));
}

// 分类结果API
static esp_err_t classification_handler(httpd_req_t *req)
{
    char response[256];

    if (xSemaphoreTake(g_classification_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
        snprintf(response, sizeof(response),
                 "%s | none: %.1f%% | exist: %.1f%% | 耗时: %.0f ms",
                 g_classification_info.result_text,
                 g_classification_info.none_score * 100.0f,
                 g_classification_info.exist_score * 100.0f,
                 g_classification_info.inference_time_ms);
        xSemaphoreGive(g_classification_mutex);
    } else {
        strcpy(response, "Error: Timeout");
    }

    httpd_resp_set_type(req, "text/plain");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, response, strlen(response));
}

static esp_err_t httpd_start(void)
{
    // 创建互斥锁
    if (g_classification_mutex == NULL) {
        g_classification_mutex = xSemaphoreCreateMutex();
        if (g_classification_mutex == NULL) {
            ESP_LOGE(TAG, "Failed to create mutex");
            return ESP_FAIL;
        }
    }

    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;
    config.ctrl_port = 32768;
    config.max_open_sockets = 7;
    config.max_uri_handlers = 8;
    config.stack_size = 8192;

    httpd_uri_t index_uri = {
        .uri = "/",
        .method = HTTP_GET,
        .handler = index_handler,
        .user_ctx = NULL
    };

    httpd_uri_t stream_uri = {
        .uri = "/stream",
        .method = HTTP_GET,
        .handler = jpg_stream_handler,
        .user_ctx = NULL
    };

    httpd_uri_t classification_uri = {
        .uri = "/classification",
        .method = HTTP_GET,
        .handler = classification_handler,
        .user_ctx = NULL
    };

    ESP_LOGI(TAG, "Starting HTTP server on port: '%d'", config.server_port);
    if (httpd_start(&stream_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(stream_httpd, &index_uri);
        httpd_register_uri_handler(stream_httpd, &stream_uri);
        httpd_register_uri_handler(stream_httpd, &classification_uri);
        ESP_LOGI(TAG, "HTTP server started successfully");
        return ESP_OK;
    }

    ESP_LOGE(TAG, "Failed to start HTTP server");
    return ESP_FAIL;
}

// ============================================================================
// 推理任务
// ============================================================================

static ImageNetCls *g_classifier = NULL;
static uint32_t g_frame_count = 0;
static char g_last_classification[128] = "Initializing...";

// JPEG解码为RGB888
static dl::image::img_t decode_jpeg_to_rgb888(camera_fb_t *fb)
{
    dl::image::jpeg_img_t jpeg_img = {
        .data = (void *)fb->buf,
        .data_len = fb->len
    };

    dl::image::img_t img = dl::image::sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);

    if (img.data == NULL) {
        ESP_LOGE(TAG, "JPEG decode failed");
    }

    return img;
}

// 推理任务
static void inference_task(void *pvParameters)
{
    ESP_LOGI(TAG, "Inference task started");

    // 创建分类器
    g_classifier = new ImageNetCls();
    if (!g_classifier) {
        ESP_LOGE(TAG, "Failed to create classifier");
        vTaskDelete(NULL);
        return;
    }

    ESP_LOGI(TAG, "Classifier created successfully, using model: image_classifier_320x320_int8.espdl");

    while (true) {
        // 获取摄像头帧
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGW(TAG, "Failed to get camera frame");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        g_frame_count++;

        // 每10帧进行一次推理
        if (g_frame_count % INFERENCE_INTERVAL == 0) {
            ESP_LOGI(TAG, "Frame %lu: Running inference...", g_frame_count);

            int64_t start_time = esp_timer_get_time();

            // 解码JPEG图像
            dl::image::img_t img = decode_jpeg_to_rgb888(fb);
            if (img.data == NULL) {
                ESP_LOGE(TAG, "JPEG decode failed");
                esp_camera_fb_return(fb);
                continue;
            }

            int64_t decode_time = esp_timer_get_time();
            ESP_LOGI(TAG, "  Decode time: %.2f ms", (decode_time - start_time) / 1000.0);

            // 运行推理
            auto &results = g_classifier->run(img);

            int64_t inference_time = esp_timer_get_time();
            ESP_LOGI(TAG, "  Inference time: %.2f ms", (inference_time - decode_time) / 1000.0);

            // 处理结果
            if (results.size() >= 2) {
                const auto &top_result = results[0];
                ESP_LOGI(TAG, "  🎯 Result: %s (%.2f%%)",
                        top_result.cat_name, top_result.score * 100.0f);

                // 获取两个类别的得分
                float none_score = 0.0f, exist_score = 0.0f;
                for (size_t i = 0; i < results.size(); i++) {
                    if (strcmp(results[i].cat_name, "none") == 0) {
                        none_score = results[i].score;
                    } else if (strcmp(results[i].cat_name, "exist") == 0) {
                        exist_score = results[i].score;
                    }
                }

                // 计算推理耗时
                float inference_time_ms = (inference_time - decode_time) / 1000.0;

                // 更新分类结果 (传递完整信息)
                char result_text[64];
                snprintf(result_text, sizeof(result_text), "%s", top_result.cat_name);
                snprintf(g_last_classification, sizeof(g_last_classification),
                        "%s (%.1f%%)", top_result.cat_name, top_result.score * 100.0f);

                update_classification(none_score, exist_score, inference_time_ms, result_text);

                // 打印所有结果
                for (size_t i = 0; i < results.size(); i++) {
                    ESP_LOGI(TAG, "  [%d] %s: %.2f%%",
                            i, results[i].cat_name, results[i].score * 100.0f);
                }
            }

            int64_t total_time = esp_timer_get_time();
            ESP_LOGI(TAG, "  Total time: %.2f ms\n", (total_time - start_time) / 1000.0);

            // 释放RGB888缓冲区
            heap_caps_free(img.data);
        }

        // 释放摄像头帧
        esp_camera_fb_return(fb);

        // 短暂延时
        vTaskDelay(pdMS_TO_TICKS(33)); // ~30fps
    }

    delete g_classifier;
    vTaskDelete(NULL);
}

// ============================================================================
// 主函数
// ============================================================================

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, " ESP32-S3 MobileNetV2 实时分类系统");
    ESP_LOGI(TAG, " Model: image_classifier_320x320_int8");
    ESP_LOGI(TAG, " Inference: Every %d frames", INFERENCE_INTERVAL);
    ESP_LOGI(TAG, "========================================\n");

    // 1. 初始化WiFi
    ESP_LOGI(TAG, "Step 1: Initializing WiFi...");
    if (wifi_init() != ESP_OK) {
        ESP_LOGE(TAG, "WiFi initialization failed!");
        return;
    }
    ESP_LOGI(TAG, "✓ WiFi connected\n");

    // 2. 初始化摄像头
    ESP_LOGI(TAG, "Step 2: Initializing camera...");
    ESP_LOGI(TAG, "Camera config: QVGA(320x240), JPEG, fb_count=2, PSRAM");

    esp_err_t cam_err = camera_init();
    ESP_LOGI(TAG, "Camera init returned: 0x%x", cam_err);

    if (cam_err != ESP_OK) {
        ESP_LOGE(TAG, "Camera initialization failed! Error code: 0x%x", cam_err);
        ESP_LOGE(TAG, "Please check:");
        ESP_LOGE(TAG, "  1. Camera is properly connected");
        ESP_LOGE(TAG, "  2. Using ESP32-S3-EYE board or compatible");
        ESP_LOGE(TAG, "  3. PSRAM is enabled in sdkconfig");
        return;
    }
    ESP_LOGI(TAG, "✓ Camera initialized\n");

    // 3. 启动HTTP流服务器
    ESP_LOGI(TAG, "Step 3: Starting HTTP server...");
    if (httpd_start() != ESP_OK) {
        ESP_LOGE(TAG, "HTTP server start failed!");
        return;
    }
    ESP_LOGI(TAG, "✓ HTTP server started on port 80\n");

    // 4. 创建推理任务
    ESP_LOGI(TAG, "Step 4: Starting inference task...");
    BaseType_t ret = xTaskCreatePinnedToCore(
        inference_task,
        "inference_task",
        8192,
        NULL,
        5,
        NULL,
        1
    );

    if (ret != pdPASS) {
        ESP_LOGE(TAG, "Failed to create inference task!");
        return;
    }
    ESP_LOGI(TAG, "✓ Inference task started\n");

    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, " System ready!");
    ESP_LOGI(TAG, " Open browser and visit:");
    ESP_LOGI(TAG, "   http://<ESP32_IP_ADDRESS>");
    ESP_LOGI(TAG, "========================================\n");

    // 主任务等待
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(10000));
        ESP_LOGI(TAG, "System running... Frame: %lu, Last: %s",
                g_frame_count, g_last_classification);
    }
}
