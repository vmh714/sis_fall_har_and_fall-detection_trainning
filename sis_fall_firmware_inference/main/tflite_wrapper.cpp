#include "tflite_wrapper.h"

#include <stdio.h>
#include "esp_log.h"
#include "esp_timer.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

static const char* TAG = "TFLiteWrapper";

// Globals, used for compatibility with Arduino-style sketches.
namespace {
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;

    // Kích thước Tensor Arena giả định (200KB) để an toàn cho mô hình Float32
    // Sau khi chạy xong, hàm interpreter->arena_used_bytes() sẽ trả về số thực tế
    constexpr int kTensorArenaSize = 200 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

int tflite_init(void) {
    tflite::InitializeTarget();

    // Map the model into a usable data structure.
    model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model provided is schema version %d not equal to supported version %d.",
                 (int)model->version(), (int)TFLITE_SCHEMA_VERSION);
        return -1;
    }

    // Đăng ký các toán tử cần thiết cho mô hình
    static tflite::MicroMutableOpResolver<60> resolver;
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
    resolver.AddLogistic(); // Sigmoid cho LSTM
    resolver.AddTanh();     // Tanh cho LSTM
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

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return -1;
    }

    // In ra lượng RAM thực tế yêu cầu để cấp phát cho mô hình!
    ESP_LOGI(TAG, "================================================");
    ESP_LOGI(TAG, "TFLITE ARENA CALCULATION RESULTS:");
    ESP_LOGI(TAG, "Total Arena Size Configured: %d bytes", kTensorArenaSize);
    ESP_LOGI(TAG, "Actual Arena Used: %d bytes", (int)interpreter->arena_used_bytes());
    ESP_LOGI(TAG, "================================================");

    // Obtain pointers to the model's input and output tensors.
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

    // Tạo dữ liệu giả định (ví dụ: gán bằng 0.0)
    // Đối với mô hình int8, input type sẽ là kTfLiteInt8
    // Đối với mô hình float32, input type sẽ là kTfLiteFloat32
    if (input->type == kTfLiteFloat32) {
        for (int i = 0; i < input->bytes / sizeof(float); ++i) {
            input->data.f[i] = 0.0f;
        }
    } else if (input->type == kTfLiteInt8) {
        for (int i = 0; i < input->bytes; ++i) {
            input->data.int8[i] = 0; // Giá trị tương đương 0.0 sau khi quantize (phụ thuộc vào zero_point)
        }
    }

    // Bấm giờ Inference
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
