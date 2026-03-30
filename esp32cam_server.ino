

#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"

// ── WiFi credentials ─────────────────────────────────────────────────────────
const char* WIFI_SSID = "theinfoflux";
const char* WIFI_PASS = "11122233";

// ── Camera pins (AI Thinker ESP32-CAM) ───────────────────────────────────────
#define CAM_PIN_PWDN    32
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK     0
#define CAM_PIN_SIOD    26
#define CAM_PIN_SIOC    27
#define CAM_PIN_D7      35
#define CAM_PIN_D6      34
#define CAM_PIN_D5      39
#define CAM_PIN_D4      36
#define CAM_PIN_D3      21
#define CAM_PIN_D2      19
#define CAM_PIN_D1      18
#define CAM_PIN_D0       5
#define CAM_PIN_VSYNC   25
#define CAM_PIN_HREF    23
#define CAM_PIN_PCLK    22

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* STREAM_CONTENT_TYPE =
    "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* STREAM_PART =
    "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

httpd_handle_t stream_httpd = NULL;
httpd_handle_t capture_httpd = NULL;

// ── Camera init ───────────────────────────────────────────────────────────────
bool init_camera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = CAM_PIN_D0;
    config.pin_d1       = CAM_PIN_D1;
    config.pin_d2       = CAM_PIN_D2;
    config.pin_d3       = CAM_PIN_D3;
    config.pin_d4       = CAM_PIN_D4;
    config.pin_d5       = CAM_PIN_D5;
    config.pin_d6       = CAM_PIN_D6;
    config.pin_d7       = CAM_PIN_D7;
    config.pin_xclk     = CAM_PIN_XCLK;
    config.pin_pclk     = CAM_PIN_PCLK;
    config.pin_vsync    = CAM_PIN_VSYNC;
    config.pin_href     = CAM_PIN_HREF;
    config.pin_sccb_sda = CAM_PIN_SIOD;
    config.pin_sccb_scl = CAM_PIN_SIOC;
    config.pin_pwdn     = CAM_PIN_PWDN;
    config.pin_reset    = CAM_PIN_RESET;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;

    if (psramFound()) {
        config.frame_size   = FRAMESIZE_VGA;   // 640×480 — good balance
        config.jpeg_quality = 12;              // 0-63 lower = better quality
        config.fb_count     = 2;
    } else {
        config.frame_size   = FRAMESIZE_CIF;
        config.jpeg_quality = 15;
        config.fb_count     = 1;
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init FAILED: 0x%x\n", err);
        return false;
    }

    // Optional: tweak sensor settings
    sensor_t* s = esp_camera_sensor_get();
    s->set_brightness(s, 0);
    s->set_contrast(s, 0);
    s->set_saturation(s, 0);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_exposure_ctrl(s, 1);

    return true;
}

// ── /capture handler ──────────────────────────────────────────────────────────
esp_err_t capture_handler(httpd_req_t* req) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    esp_err_t res = httpd_resp_send(req, (const char*)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    return res;
}

// ── /stream handler ───────────────────────────────────────────────────────────
esp_err_t stream_handler(httpd_req_t* req) {
    httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    char part_buf[64];

    while (true) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) break;

        // boundary
        httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
        // part header
        size_t hlen = snprintf(part_buf, sizeof(part_buf), STREAM_PART, fb->len);
        httpd_resp_send_chunk(req, part_buf, hlen);
        // JPEG data
        esp_err_t res = httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);
        esp_camera_fb_return(fb);

        if (res != ESP_OK) break;   // client disconnected
    }
    return ESP_OK;
}

// ── / status handler ──────────────────────────────────────────────────────────
esp_err_t status_handler(httpd_req_t* req) {
    char buf[256];
    snprintf(buf, sizeof(buf),
        "<h2>TheInfoFlux — ESP32-CAM</h2>"
        "<p>IP: %s</p>"
        "<p><a href='/capture'>Snapshot</a> | <a href='/stream'>MJPEG Stream</a></p>",
        WiFi.localIP().toString().c_str());
    httpd_resp_set_type(req, "text/html");
    httpd_resp_sendstr(req, buf);
    return ESP_OK;
}

void start_servers() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();

    // capture server on port 80
    config.server_port = 80;
    if (httpd_start(&capture_httpd, &config) == ESP_OK) {
        httpd_uri_t cap_uri = { .uri = "/capture", .method = HTTP_GET,
                                .handler = capture_handler, .user_ctx = NULL };
        httpd_register_uri_handler(capture_httpd, &cap_uri);

        httpd_uri_t root_uri = { .uri = "/", .method = HTTP_GET,
                                 .handler = status_handler, .user_ctx = NULL };
        httpd_register_uri_handler(capture_httpd, &root_uri);
    }

    // stream server on port 81
    config.server_port = 81;
    config.ctrl_port   = 32769;
    if (httpd_start(&stream_httpd, &config) == ESP_OK) {
        httpd_uri_t str_uri = { .uri = "/stream", .method = HTTP_GET,
                                .handler = stream_handler, .user_ctx = NULL };
        httpd_register_uri_handler(stream_httpd, &str_uri);
    }
}

// ── Setup & Loop ──────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    Serial.println("\n[TheInfoFlux] ESP32-CAM YOLOv3 feeder starting …");

    if (!init_camera()) { Serial.println("Camera FAIL"); while(1); }

    WiFi.begin(WIFI_SSID, WIFI_PASS);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
    Serial.println();
    Serial.printf("Connected! IP: %s\n", WiFi.localIP().toString().c_str());
    Serial.printf("Snapshot : http://%s/capture\n", WiFi.localIP().toString().c_str());
    Serial.printf("Stream   : http://%s:81/stream\n", WiFi.localIP().toString().c_str());

    start_servers();
}

void loop() {
    delay(10000);   // nothing to do — HTTP server runs on its own task
}
