#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "tflite_wrapper.h"
#include <string.h>
#include "driver/uart.h"
static const char* TAG = "MainApp";

void app_main(void)
{
    ESP_LOGI(TAG, "Starting SisFall TFLite Micro Inference App...");

    if (tflite_init() != 0) {
        ESP_LOGE(TAG, "TFLite Initialization failed!");
        return;
    }

    // Tăng TX buffer lên 2048 để tránh việc printf bị nghẽn khi xuất chuỗi JSON
    uart_driver_install(UART_NUM_0, 8192, 2048, 0, NULL, 0);

    int expected_bytes = get_input_bytes();
    if (expected_bytes <= 0) {
        ESP_LOGE(TAG, "Invalid model input size!");
        return;
    }

    uint8_t *rx_buf = (uint8_t *)malloc(expected_bytes);
    if (!rx_buf) {
        ESP_LOGE(TAG, "Failed to allocate memory for UART RX buffer");
        return;
    }

    ESP_LOGI(TAG, "Waiting for Python tool via UART... (Handshake: SYNC -> RDY)");

    uint8_t window[4] = {0, 0, 0, 0};

    while (1) {
        uint8_t b;
        // Đọc từng byte một để trượt cửa sổ
        int len = uart_read_bytes(UART_NUM_0, &b, 1, pdMS_TO_TICKS(10));
        
        if (len > 0) {
            window[0] = window[1];
            window[1] = window[2];
            window[2] = window[3];
            window[3] = b;
            
            // Nếu 4 byte cuối cùng ghép lại thành 'SYNC'
            if (window[0] == 'S' && window[1] == 'Y' && window[2] == 'N' && window[3] == 'C') {
                // Xóa sạch mọi rác còn sót lại trong buffer
                uart_flush_input(UART_NUM_0);
                
                // Phản hồi RDY
                printf("RDY\n");
                fflush(stdout); 
                
                // Đọc cục data float
                int total_received = 0;
                while (total_received < expected_bytes) {
                    int rxBytes = uart_read_bytes(UART_NUM_0, 
                                                  rx_buf + total_received, 
                                                  expected_bytes - total_received, 
                                                  pdMS_TO_TICKS(2000));
                    if (rxBytes > 0) {
                        total_received += rxBytes;
                    } else {
                        break;
                    }
                }
                
                if (total_received == expected_bytes) {
                    tflite_run_inference_with_data((float*)rx_buf, expected_bytes);
                } else {
                    ESP_LOGE(TAG, "Timeout! Expected %d bytes, got %d", expected_bytes, total_received);
                }
                
                // Reset lại cửa sổ để chờ lần tiếp theo
                memset(window, 0, sizeof(window));
            }
            continue;
        }
        
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
