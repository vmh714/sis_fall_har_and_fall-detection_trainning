#include "driver/uart.h"
#include "esp_log.h"
#include "esp_vfs_usb_serial_jtag.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "tflite_wrapper.h"
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

// =========================================================================
// CỜ BUILD DÀNH CHO 2 LOẠI MẠCH (0 = MẠCH CH343 THƯỜNG, 1 = MẠCH NATIVE USB)
// =========================================================================
#define USE_NATIVE_USB 0 // Đã bật thành 1 cho mạch Seeed Studio
// =========================================================================

static const char *TAG = "MainApp";

// Cấu trúc gói tin đẩy qua Queue
typedef struct {
  uint8_t *data;
} inference_msg_t;

static QueueHandle_t inference_queue;
static int expected_bytes = 0;

// ==========================================================
// TASK 1: INFERENCE TASK (Chạy trên Core 1)
// ==========================================================
void inference_task(void *pvParameters) {
  inference_msg_t msg;
  ESP_LOGI("InferenceTask", "Bắt đầu chờ dữ liệu trên Core 1...");

  while (1) {
    // Chờ nhận dữ liệu từ Queue (chờ vô hạn)
    if (xQueueReceive(inference_queue, &msg, portMAX_DELAY) == pdTRUE) {
      // Chạy model
      tflite_run_inference_with_data((float *)msg.data, expected_bytes);

      // Giải phóng bộ nhớ sau khi chạy xong
      free(msg.data);
    }
  }
}

// ==========================================================
// TASK 2: UART/USB RX TASK (Chạy trên Core 0)
// ==========================================================
void rx_task(void *pvParameters) {
  ESP_LOGI("RxTask",
           "Waiting for Python tool via UART/USB... (Handshake: SYNC -> RDY)");

  uint8_t window[4] = {0, 0, 0, 0};

#if USE_NATIVE_USB
  int flags = fcntl(0, F_GETFL, 0);
  fcntl(0, F_SETFL, flags | O_NONBLOCK);
#endif

  while (1) {
    uint8_t b;
    int len = 0;

#if USE_NATIVE_USB
    len = read(0, &b, 1);
#else
    len = uart_read_bytes(UART_NUM_0, &b, 1, pdMS_TO_TICKS(10));
#endif

    if (len > 0) {
      window[0] = window[1];
      window[1] = window[2];
      window[2] = window[3];
      window[3] = b;

      // Nếu 4 byte cuối cùng ghép lại thành 'SYNC'
      if (window[0] == 'S' && window[1] == 'Y' && window[2] == 'N' &&
          window[3] == 'C') {
#if !USE_NATIVE_USB
        uart_flush_input(UART_NUM_0);
#else
        // Flush stdin: Đọc hết rác (ví dụ ký tự '\n' do Python gửi kèm)
        uint8_t dummy;
        while (read(0, &dummy, 1) > 0) {
        }
#endif
        // Phản hồi RDY
        printf("RDY\n");
        fflush(stdout);

        // Cấp phát bộ nhớ cho mẻ dữ liệu mới
        uint8_t *rx_buf = (uint8_t *)malloc(expected_bytes);
        if (!rx_buf) {
          ESP_LOGE("RxTask", "Hết RAM! Không thể cấp phát RX buffer");
          memset(window, 0, sizeof(window));
          continue;
        }

        int total_received = 0;

#if USE_NATIVE_USB
        // GIỮ NGUYÊN CHẾ ĐỘ NON-BLOCKING ĐỂ TIMEOUT HOẠT ĐỘNG!
        int timeout_count = 0;
        while (total_received < expected_bytes && timeout_count < 200) {
          int rxBytes =
              read(0, rx_buf + total_received, expected_bytes - total_received);
          if (rxBytes > 0) {
            total_received += rxBytes;
            timeout_count = 0;
          } else {
            vTaskDelay(pdMS_TO_TICKS(10));
            timeout_count++;
          }
        }
#else
        while (total_received < expected_bytes) {
          int rxBytes = uart_read_bytes(UART_NUM_0, rx_buf + total_received,
                                        expected_bytes - total_received,
                                        pdMS_TO_TICKS(2000));
          if (rxBytes > 0) {
            total_received += rxBytes;
          } else {
            break;
          }
        }
#endif

        if (total_received == expected_bytes) {
          // Đẩy con trỏ dữ liệu sang Inference Task
          inference_msg_t msg = {.data = rx_buf};
          if (xQueueSend(inference_queue, &msg, 0) != pdTRUE) {
            ESP_LOGE("RxTask", "Queue đầy! Rớt gói.");
            free(rx_buf);
          }
        } else {
          ESP_LOGE("RxTask", "Timeout! Expected %d bytes, got %d",
                   expected_bytes, total_received);
          free(rx_buf); // Tránh rò rỉ bộ nhớ
        }

        memset(window, 0, sizeof(window));
      }
      continue;
    }
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

// ==========================================================
// HÀM MAIN
// ==========================================================
void app_main(void) {
  ESP_LOGI(TAG, "Starting SisFall TFLite Micro Inference App...");

#if USE_NATIVE_USB
  // TẮT tính năng tự động chuyển đổi CR/LF của VFS để truyền nhị phân (Binary
  // Float) nguyên vẹn Dùng ESP_LINE_ENDINGS_LF để VFS hiểu rằng không cần
  // convert CRLF sang LF nữa
  esp_vfs_dev_usb_serial_jtag_set_rx_line_endings(ESP_LINE_ENDINGS_LF);
#endif

  // Khởi tạo model
  if (tflite_init() != 0) {
    ESP_LOGE(TAG, "TFLite Initialization failed!");
    return;
  }

  expected_bytes = get_input_bytes();
  if (expected_bytes <= 0) {
    ESP_LOGE(TAG, "Invalid model input size!");
    return;
  }

#if !USE_NATIVE_USB
  uart_driver_install(UART_NUM_0, 8192, 2048, 0, NULL, 0);
  uart_set_pin(UART_NUM_0, 43, 44, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
#endif

  // Tạo Queue chứa tối đa 5 mẻ dữ liệu
  inference_queue = xQueueCreate(5, sizeof(inference_msg_t));

  // Khởi tạo 2 Task phân luồng ra 2 nhân (Core 0 và Core 1)
  xTaskCreatePinnedToCore(rx_task, "RxTask", 8192, NULL, 5, NULL,
                          0); // Chạy trên Core 0
  xTaskCreatePinnedToCore(inference_task, "InferenceTask", 32768, NULL, 5, NULL,
                          1); // Chạy trên Core 1
}
