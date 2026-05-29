#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "tflite_wrapper.h"

static const char* TAG = "MainApp";

void app_main(void)
{
    ESP_LOGI(TAG, "Starting SisFall TFLite Micro Inference App...");

    // Khởi tạo TFLite
    if (tflite_init() == 0) {
        // Chạy thử Inference để đo thời gian
        tflite_run_inference();
    } else {
        ESP_LOGE(TAG, "TFLite Initialization failed!");
    }

    // Vòng lặp chính của FreeRTOS (nếu cần xử lý thêm)
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
