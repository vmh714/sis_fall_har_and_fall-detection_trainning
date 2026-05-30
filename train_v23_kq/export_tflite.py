import tensorflow as tf
import os
import sys
import numpy as np

# Disable mixed precision just in case
tf.keras.backend.set_floatx('float32')

ROOT_DIR = '/home/linh_linh/dataset/sis_fall_har_and_fall-detection_trainning'
model_path = os.path.join(ROOT_DIR, 'train_v23_kq', 'best_model_v23.keras')
tflite_path = os.path.join(ROOT_DIR, 'train_v23_kq', 'model_v23_int8.tflite')
cc_path = os.path.join(ROOT_DIR, 'train_v23_kq', 'model_data_v23.cc')
h_path = os.path.join(ROOT_DIR, 'train_v23_kq', 'model_data_v23.h')

print(f"[*] Loading Keras model from: {model_path}")
model = tf.keras.models.load_model(model_path)

# Load representative dataset for quantization
print("[*] Loading representative dataset for INT8 Quantization...")
cache_file = os.path.join(ROOT_DIR, 'train_cache_v18_v19_idle_trans', 'X_train.npy')
X_train = np.load(cache_file)
# Use 100 samples for calibration
def representative_data_gen():
    for i in range(100):
        yield [X_train[i:i+1].astype(np.float32)]

print("[*] Converting to TFLite (INT8 Quantization)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

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

// TCN v23 Model for Fall Detection (INT8 Quantized - Strides + Pooling, NO DILATION)
// Architecture: MaxPooling1D(2) -> 4x(Conv1D(k=7) + SE Block), GAP+GMP, 5 Classes
// Input: (200, 6) INT8
// Output: (5) INT8 -> Walk, Run, Idle, Trans, Fall
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
