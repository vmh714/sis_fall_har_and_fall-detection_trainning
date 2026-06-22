#include "tflite_wrapper.h"

#include "esp_heap_caps.h" // Thêm thư viện để cấp phát PSRAM
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <stdio.h>

#include <math.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

static const char *TAG = "TFLiteWrapper";

namespace {
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

// === Vị trí Tensor Arena: 0 = SRAM nội bộ (mặc định — test "arena ở SRAM"); 1 = PSRAM ===
// Khi arena < 64KB (vừa data-cache S3) thì SRAM vs PSRAM tốc độ ~ngang nhau (compute-bound).
#define ARENA_USE_PSRAM 0
// Arena CNN v30/v30_kd2/v32 thực dùng ~12-20KB (theo window); cấp 100KB dư headroom.
constexpr int kTensorArenaSize = 100 * 1024;

// uint8_t tensor_arena[kTensorArenaSize];
// Đổi từ mảng tĩnh sang con trỏ để cấp phát động
uint8_t *tensor_arena = nullptr;

TaskHandle_t g_ram_monitor_task_handle = nullptr;
volatile bool g_is_inferencing = false;
volatile size_t g_max_arena_used = 0;
volatile size_t g_min_free_psram = 0xFFFFFFFF;
volatile size_t g_min_free_sram = 0xFFFFFFFF;

void ram_monitor_task(void *pvParameters) {
  while (1) {
    // Chờ tín hiệu báo bắt đầu Invoke
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    g_min_free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    g_min_free_sram = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);

    while (g_is_inferencing && interpreter != nullptr) {
      // Soi Arena nội bộ TFLite
      size_t current_arena = interpreter->arena_used_bytes();
      if (current_arena > g_max_arena_used) {
        g_max_arena_used = current_arena;
      }

      // Soi thẳng PSRAM thực tế của ESP32
      size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
      if (free_psram < g_min_free_psram)
        g_min_free_psram = free_psram;

      // Soi thẳng SRAM thực tế
      size_t free_sram = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
      if (free_sram < g_min_free_sram)
        g_min_free_sram = free_sram;

      vTaskDelay(pdMS_TO_TICKS(1)); // Quét RAM mỗi 1ms
    }
  }
}
} // namespace

int tflite_init(void) {
  tflite::InitializeTarget();

  // 1. Cấp phát Tensor Arena (vị trí theo ARENA_USE_PSRAM).
#if ARENA_USE_PSRAM
  const uint32_t arena_caps = MALLOC_CAP_SPIRAM;
  const char *arena_loc = "PSRAM";
#else
  const uint32_t arena_caps = MALLOC_CAP_INTERNAL;
  const char *arena_loc = "SRAM";
#endif
  tensor_arena = (uint8_t *)heap_caps_malloc(kTensorArenaSize, arena_caps);
  if (tensor_arena == nullptr) {
    ESP_LOGE(TAG, "Lỗi: Không thể cấp phát %d bytes trên %s!", kTensorArenaSize,
             arena_loc);
    return -1;
  }
  ESP_LOGI(TAG, "Đã cấp phát thành công %d bytes trên %s.", kTensorArenaSize,
           arena_loc);

  // 2. Load mô hình
  model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG,
             "Model provided is schema version %d not equal to supported "
             "version %d.",
             (int)model->version(), (int)TFLITE_SCHEMA_VERSION);
    return -1;
  }

  // 3. Đăng ký Ops — CNN ESP-NN v30/v30_kd2/v32 dùng CHUNG 8 ops (lấy từ model_data.h auto-gen).
  //    CNN thuần ESP-NN; relu6 đã fuse vào Conv nên KHÔNG có op RELU6 riêng.
  //    (Model cũ LSTM/ResNet1D cần thêm AddAdd/AddMul/AddLogistic/AddReduceMax/
  //     AddStridedSlice/AddRelu/AddMinimum/AddMaximum/AddUnidirectionalSequenceLSTM — xem git history.)
  static tflite::MicroMutableOpResolver<8> resolver;
  resolver.AddConcatenation();
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddReshape();
  resolver.AddSoftmax();

  // 4. Build Interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // 5. Cấp phát Tensor vào vùng nhớ đã tạo
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    ESP_LOGE(TAG, "AllocateTensors() failed");
    return -1;
  }

  // In ra lượng RAM thực tế yêu cầu (Thêm vTaskDelay để chống trôi UART log)
  vTaskDelay(pdMS_TO_TICKS(20));
  ESP_LOGI(TAG, "================================================");
  ESP_LOGI(TAG, "TFLITE ARENA CALCULATION RESULTS:");
  ESP_LOGI(TAG, "Total Arena Size Configured: %d bytes (SRAM)", kTensorArenaSize);
  ESP_LOGI(TAG, "Actual Arena Used: %d bytes", (int)interpreter->arena_used_bytes());
  ESP_LOGI(TAG, "================================================");
  vTaskDelay(pdMS_TO_TICKS(20));

  input = interpreter->input(0);
  output = interpreter->output(0);

  ESP_LOGI(TAG, "Model initialized successfully!");
  ESP_LOGI(TAG, "Input shape: [%d, %d, %d]", input->dims->data[0],
           input->dims->data[1], input->dims->data[2]);
  return 0;
}

void tflite_run_inference(void) {
  if (interpreter == nullptr || input == nullptr || output == nullptr) {
    ESP_LOGE(TAG, "Interpreter not initialized!");
    return;
  }

  if (input->type == kTfLiteFloat32) {
    for (int i = 0; i < input->bytes / sizeof(float); ++i) {
      input->data.f[i] = 0.0f;
    }
  } else if (input->type == kTfLiteInt8) {
    for (int i = 0; i < input->bytes; ++i) {
      input->data.int8[i] = 0;
    }
  }

  int64_t start_time = esp_timer_get_time();
  TfLiteStatus invoke_status = interpreter->Invoke();
  int64_t end_time = esp_timer_get_time();
  int64_t inference_time_us = end_time - start_time;

  if (invoke_status != kTfLiteOk) {
    ESP_LOGE(TAG, "Invoke failed!");
    return;
  }

  ESP_LOGI(TAG, "================================================");
  ESP_LOGI(TAG, "INFERENCE PERFORMANCE:");
  ESP_LOGI(TAG, "Inference Time: %lld microseconds (%.2f ms)",
           inference_time_us, inference_time_us / 1000.0);
  ESP_LOGI(TAG, "================================================");
}

int get_input_bytes(void) {
  if (input != nullptr) {
    // Luôn trả về số byte của mảng FLOAT32 (để tương thích với data thật từ
    // IMU/Python)
    int elements = 1;
    for (int i = 0; i < input->dims->size; ++i) {
      elements *= input->dims->data[i];
    }
    // NẾU model là v31 (4 kênh), ta vẫn cần nhận 6 kênh từ Python/IMU
    if (input->dims->size == 3 && input->dims->data[2] == 4) {
      return (elements / 4) * 6 * sizeof(float);
    }
    return elements * sizeof(float);
  }
  return 0;
}

void tflite_run_inference_with_data(float *rx_data, size_t num_bytes) {
  if (interpreter == nullptr || input == nullptr || output == nullptr) {
    ESP_LOGE(TAG, "Interpreter not initialized!");
    return;
  }

  int elements = num_bytes / sizeof(float);
  int expected_elements = 1;
  for (int i = 0; i < input->dims->size; ++i) {
    expected_elements *= input->dims->data[i];
  }

  bool is_4_axis = (input->dims->size == 3 && input->dims->data[2] == 4);

  if (is_4_axis) {
    if (elements != (expected_elements / 4) * 6) {
      ESP_LOGE(TAG, "Size mismatch: Expected %d elements (6-axis), got %d elements",
               (expected_elements / 4) * 6, elements);
      return;
    }
  } else {
    if (elements != expected_elements) {
      ESP_LOGE(TAG, "Size mismatch: Expected %d elements, got %d elements",
               expected_elements, elements);
      return;
    }
  }

  // Ép kiểu (Quantize) từ dữ liệu Float thật sang INT8 (nếu model là INT8)
  if (input->type == kTfLiteInt8) {
    if (is_4_axis) {
      // Logic xử lý 4 trục (v31)
      int num_steps = expected_elements / 4;
      for (int step = 0; step < num_steps; step++) {
        float ax = rx_data[step * 6 + 0];
        float ay = rx_data[step * 6 + 1];
        float az = rx_data[step * 6 + 2];
        float gx = rx_data[step * 6 + 3];
        float gy = rx_data[step * 6 + 4];
        float gz = rx_data[step * 6 + 5];

        // Tính Gyro RMS (gyro_mag)
        float gyro_mag = sqrtf(gx * gx + gy * gy + gz * gz);

        float features[4] = {ax, ay, az, gyro_mag};

        for (int f = 0; f < 4; f++) {
          float val = features[f];
          if (f < 3) {
            // Xử lý Accel: Kẹp giới hạn -8.0g đến 8.0g rồi scale
            if (val > 8.0f) val = 8.0f;
            if (val < -8.0f) val = -8.0f;
            val = val / 8.0f;
          } else {
            // Xử lý Gyro Mag: Scale theo 500 dps, không có âm vì là độ lớn
            if (val > 500.0f) val = 500.0f;
            if (val < 0.0f) val = 0.0f;
            val = val / 500.0f;
          }

          int32_t quantized_val = round(val / input->params.scale) + input->params.zero_point;
          if (quantized_val > 127) quantized_val = 127;
          if (quantized_val < -128) quantized_val = -128;
          input->data.int8[step * 4 + f] = (int8_t)quantized_val;
        }
      }
    } else {
      // Logic xử lý 6 trục (cũ)
      for (int i = 0; i < expected_elements; i++) {
        float val = rx_data[i];
        int feature_idx = i % 6;

        if (feature_idx < 3) {
          if (val > 8.0f) val = 8.0f;
          if (val < -8.0f) val = -8.0f;
          val = val / 8.0f;
        } else {
          if (val > 500.0f) val = 500.0f;
          if (val < -500.0f) val = -500.0f;
          val = val / 500.0f;
        }

        int32_t quantized_val = round(val / input->params.scale) + input->params.zero_point;
        if (quantized_val > 127) quantized_val = 127;
        if (quantized_val < -128) quantized_val = -128;
        input->data.int8[i] = (int8_t)quantized_val;
      }
    }
  } else if (input->type == kTfLiteFloat32) {
    // Không hỗ trợ Float32 tự động cho 4-axis (do hiếm khi dùng float32 trên ESP32)
    memcpy(input->data.f, rx_data, expected_elements * sizeof(float));
  } else {
    ESP_LOGE(TAG, "Unsupported input type: %d", input->type);
    return;
  }

  g_max_arena_used = interpreter->arena_used_bytes();
  g_min_free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  g_min_free_sram = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  g_is_inferencing = true;

  // BẮT BUỘC ĐỐI VỚI LSTM: Xóa trạng thái ẩn (hidden state) của suy luận trước
  // đó
  interpreter->Reset();

  int64_t start_time = esp_timer_get_time();
  TfLiteStatus invoke_status = interpreter->Invoke();
  int64_t end_time = esp_timer_get_time();

  g_is_inferencing = false;

  if (invoke_status != kTfLiteOk) {
    ESP_LOGE(TAG, "Invoke failed!");
    return;
  }

  int64_t inference_time_us = end_time - start_time;

  // Lấy số lượng class
  int num_classes = 1;
  if (output->dims->size > 1) {
    num_classes = output->dims->data[1];
  }

  // --- Giải Lượng tử hóa (Dequantize) Đầu Ra ---
  float probs[10]; // Giả sử model có tối đa 10 class
  if (num_classes > 10)
    num_classes = 10;

  for (int i = 0; i < num_classes; i++) {
    if (output->type == kTfLiteInt8) {
      probs[i] = (output->data.int8[i] - output->params.zero_point) *
                 output->params.scale;
    } else {
      probs[i] = output->data.f[i];
    }
  }

  int predicted_class = 0;
  float max_prob = probs[0];
  for (int i = 1; i < num_classes; i++) {
    if (probs[i] > max_prob) {
      max_prob = probs[i];
      predicted_class = i;
    }
  }

  // --- Cập nhật: Logic 25% Threshold cho Fall ---
  const int FALL_CLASS_INDEX = 4; // 'Fall' là class thứ 5 theo CLASS_NAMES
  if (num_classes > FALL_CLASS_INDEX && probs[FALL_CLASS_INDEX] >= 0.25f) {
    predicted_class = FALL_CLASS_INDEX;
  }

  // --- Tính toán tư thế (Posture Logic) CHỈ KHI LÀ IDLE ---
  const int IDLE_CLASS_INDEX = 2; // 'Idle' là class thứ 3 theo CLASS_NAMES

  bool is_stand_sit = true;
  bool posture_computed = false;

  if (predicted_class == IDLE_CLASS_INDEX) {
    double sum_y_sq = 0.0f;
    double sum_xz_sq = 0.0f;
    int num_samples = num_bytes / (6 * sizeof(float)); // = WINDOW_SIZE (v32_w128->128 / w256->256)
    for (int i = 0; i < num_samples; i++) {
      float ax = rx_data[i * 6 + 0];
      float ay = rx_data[i * 6 + 1];
      float az = rx_data[i * 6 + 2];

      sum_y_sq += (ay * ay);
      sum_xz_sq += (ax * ax + az * az);
    }
    is_stand_sit = (sum_y_sq > sum_xz_sq);
    posture_computed = true;
  }

  // Gửi kết quả về Python. Dùng "null" nếu không tính toán posture
  printf("{\"time_us\": %lld, \"is_stand_sit\": %s, \"arena_used\": %d, "
         "\"free_psram\": %d, \"free_sram\": %d, \"probs\": [",
         inference_time_us,
         posture_computed ? (is_stand_sit ? "true" : "false") : "null",
         (int)g_max_arena_used, (int)g_min_free_psram, (int)g_min_free_sram);

  for (int i = 0; i < num_classes; i++) {
    printf("%.4f", probs[i]);
    if (i < num_classes - 1)
      printf(", ");
  }
  printf("]}\n");

  fflush(stdout); // Ép ESP32 đẩy toàn bộ JSON qua UART ngay lập tức
}