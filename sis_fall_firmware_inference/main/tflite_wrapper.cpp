#include "tflite_wrapper.h"

#include <stdio.h>
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h" // Thêm thư viện để cấp phát PSRAM
#include <math.h>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

static const char* TAG = "TFLiteWrapper";

namespace {
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;

    // Tăng kích thước lên 1MB
    constexpr int kTensorArenaSize = 100 * 1024;
    
    uint8_t tensor_arena[kTensorArenaSize];
    // Đổi từ mảng tĩnh sang con trỏ để cấp phát động
    //uint8_t* tensor_arena = nullptr;
}  // namespace

int tflite_init(void) {
    tflite::InitializeTarget();

    // 1. Cấp phát Tensor Arena vào PSRAM
    // tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
    // if (tensor_arena == nullptr) {
    //     ESP_LOGE(TAG, "Lỗi: Không thể cấp phát %d bytes trên PSRAM!", kTensorArenaSize);
    //     return -1;
    // }
    // ESP_LOGI(TAG, "Đã cấp phát thành công %d bytes trên PSRAM.", kTensorArenaSize);

    // 2. Load mô hình
    model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model provided is schema version %d not equal to supported version %d.",
                 (int)model->version(), (int)TFLITE_SCHEMA_VERSION);
        return -1;
    }

    // 3. Đăng ký các toán tử
    static tflite::MicroMutableOpResolver<65> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddUnidirectionalSequenceLSTM();
    resolver.AddConcatenation();
    resolver.AddSoftmax();
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddReduceMax();
    resolver.AddStridedSlice();
    resolver.AddPack();
    resolver.AddWhile();
    resolver.AddLess();
    resolver.AddLessEqual();
    resolver.AddGreater();
    resolver.AddGreaterEqual();
    resolver.AddEqual();
    resolver.AddNotEqual();
    resolver.AddSelectV2();
    resolver.AddGather();
    resolver.AddShape();
    resolver.AddZerosLike();
    resolver.AddFill();
    resolver.AddLogicalAnd();
    resolver.AddLogicalOr();
    resolver.AddLogicalNot();
    resolver.AddSplit();
    resolver.AddSplitV();
    resolver.AddSlice();
    resolver.AddTranspose();
    resolver.AddSqueeze();
    resolver.AddUnpack();
    resolver.AddLogistic(); 
    resolver.AddTanh();     
    resolver.AddSub();
    resolver.AddExp();
    resolver.AddSquare();
    resolver.AddSqrt();
    resolver.AddRsqrt();
    resolver.AddMaximum();
    resolver.AddMinimum();
    resolver.AddRelu();
    resolver.AddRelu6();
    resolver.AddPad();
    resolver.AddPadV2();
    resolver.AddMean();
    resolver.AddReduceMin();
    resolver.AddCast();
    resolver.AddExpandDims();
    resolver.AddSpaceToBatchNd();
    resolver.AddBatchToSpaceNd();

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

    // In ra lượng RAM thực tế yêu cầu
    ESP_LOGI(TAG, "================================================");
    ESP_LOGI(TAG, "TFLITE ARENA CALCULATION RESULTS:");
    ESP_LOGI(TAG, "Total Arena Size Configured: %d bytes (SRAM)", kTensorArenaSize);
    ESP_LOGI(TAG, "Actual Arena Used: %d bytes", (int)interpreter->arena_used_bytes());
    ESP_LOGI(TAG, "================================================");

    input = interpreter->input(0);
    output = interpreter->output(0);
    
    ESP_LOGI(TAG, "Model initialized successfully!");
    ESP_LOGI(TAG, "Input shape: [%d, %d, %d]", input->dims->data[0], input->dims->data[1], input->dims->data[2]);
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
    ESP_LOGI(TAG, "Inference Time: %lld microseconds (%.2f ms)", inference_time_us, inference_time_us / 1000.0);
    ESP_LOGI(TAG, "================================================");
}

int get_input_bytes(void) {
    if (input != nullptr) {
        // Luôn trả về số byte của mảng FLOAT32 (để tương thích với data thật từ IMU/Python)
        int elements = 1;
        for (int i = 0; i < input->dims->size; ++i) {
            elements *= input->dims->data[i];
        }
        return elements * sizeof(float);
    }
    return 0;
}

void tflite_run_inference_with_data(float* rx_data, size_t num_bytes) {
    if (interpreter == nullptr || input == nullptr || output == nullptr) {
        ESP_LOGE(TAG, "Interpreter not initialized!");
        return;
    }

    int elements = num_bytes / sizeof(float);
    int expected_elements = 1;
    for (int i = 0; i < input->dims->size; ++i) {
        expected_elements *= input->dims->data[i];
    }

    if (elements != expected_elements) {
        ESP_LOGE(TAG, "Size mismatch: Expected %d elements, got %d elements", expected_elements, elements);
        return;
    }

    // Ép kiểu (Quantize) từ dữ liệu Float thật sang INT8 (nếu model là INT8)
    if (input->type == kTfLiteInt8) {
        for (int i = 0; i < elements; i++) {
            float val = rx_data[i];
            int32_t quantized_val = round(val / input->params.scale) + input->params.zero_point;
            // Kẹp (Clamp) giá trị vào giới hạn int8_t
            if (quantized_val > 127) quantized_val = 127;
            if (quantized_val < -128) quantized_val = -128;
            input->data.int8[i] = (int8_t)quantized_val;
        }
    } else if (input->type == kTfLiteFloat32) {
        memcpy(input->data.f, rx_data, num_bytes);
    } else {
        ESP_LOGE(TAG, "Unsupported input type: %d", input->type);
        return;
    }

    int64_t start_time = esp_timer_get_time();
    TfLiteStatus invoke_status = interpreter->Invoke();
    int64_t end_time = esp_timer_get_time();
    
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
    if (num_classes > 10) num_classes = 10;
    
    for (int i = 0; i < num_classes; i++) {
        if (output->type == kTfLiteInt8) {
            probs[i] = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
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
        // Chỉ tính toán căn bậc 2 khi thực sự cần thiết để tiết kiệm CPU
        float sum_y = 0.0f;
        float sum_xz = 0.0f;
        int num_samples = num_bytes / (6 * sizeof(float)); // Thường là 200
        for (int i = 0; i < num_samples; i++) {
            float ax = rx_data[i * 6 + 0];
            float ay = rx_data[i * 6 + 1];
            float az = rx_data[i * 6 + 2];
            
            float abs_y = ay > 0 ? ay : -ay;
            sum_y += abs_y;
            sum_xz += sqrtf(ax * ax + az * az);
        }
        is_stand_sit = (sum_y > sum_xz);
        posture_computed = true;
    }

    // Gửi kết quả về Python. Dùng "null" nếu không tính toán posture
    printf("{\"time_us\": %lld, \"is_stand_sit\": %s, \"probs\": [", 
            inference_time_us, posture_computed ? (is_stand_sit ? "true" : "false") : "null");
            
    for (int i = 0; i < num_classes; i++) {
        printf("%.4f", probs[i]);
        if (i < num_classes - 1) printf(", ");
    }
    printf("]}\n");
    
    fflush(stdout); // Ép ESP32 đẩy toàn bộ JSON qua UART ngay lập tức
}