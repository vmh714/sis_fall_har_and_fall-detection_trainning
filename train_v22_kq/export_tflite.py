import tensorflow as tf
import os
import sys

# Disable mixed precision just in case
tf.keras.backend.set_floatx('float32')

model_path = '/home/linh_linh/dataset/sis_fall_har_and_fall-detection_trainning/train_v22_kq/best_model_v22.keras'
tflite_path = '/home/linh_linh/dataset/sis_fall_har_and_fall-detection_trainning/sis_fall_firmware_inference/main/model_v22_float32.tflite'
cc_path = '/home/linh_linh/dataset/sis_fall_har_and_fall-detection_trainning/sis_fall_firmware_inference/main/model_data.cc'
h_path = '/home/linh_linh/dataset/sis_fall_har_and_fall-detection_trainning/sis_fall_firmware_inference/main/model_data.h'

print(f"[*] Loading Keras model from: {model_path}")
model = tf.keras.models.load_model(model_path)

print("[*] Converting to TFLite (Float32)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Optimize for size/latency if needed, but keeping default Float32 for ESP32-S3 FPU compatibility
tflite_model = converter.convert()

with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"[*] Saved TFLite binary to: {tflite_path}")

print("[*] Generating C array...")
hex_array = [f'0x{byte:02x}' for byte in tflite_model]
formatted_hex = ""
for i, h in enumerate(hex_array):
    formatted_hex += h + ', '
    if (i + 1) % 12 == 0:
        formatted_hex += '\n  '

cc_content = f"""#include "model_data.h"

// TCN v22 Model for Fall Detection (Float32)
// Architecture: Kernel=7, Dilation=[1,2,4,8]x2, SE Block, 5 Classes
// Input: (200, 6)
// Output: (5) -> Walk, Run, Idle, Trans, Fall
const unsigned char g_model_data[] = {{
  {formatted_hex.strip(', ')}
}};

const unsigned int g_model_data_len = {len(tflite_model)};
"""

with open(cc_path, 'w') as f:
    f.write(cc_content)

h_content = """#ifndef MODEL_DATA_H_
#define MODEL_DATA_H_

#ifdef __cplusplus
extern "C" {
#endif

extern const unsigned char g_model_data[];
extern const unsigned int g_model_data_len;

#ifdef __cplusplus
}
#endif

#endif // MODEL_DATA_H_
"""

with open(h_path, 'w') as f:
    f.write(h_content)

print(f"[*] Successfully generated {cc_path} and {h_path}")
print(f"[*] Array name: g_model_data (Length: {len(tflite_model)} bytes)")
