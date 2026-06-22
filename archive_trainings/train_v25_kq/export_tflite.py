import tensorflow as tf
import os
import sys
import numpy as np

# Disable mixed precision just in case
tf.keras.backend.set_floatx('float32')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

model_path = os.path.join(CURRENT_DIR, 'best_model_v25.keras')
tflite_path = os.path.join(CURRENT_DIR, 'model_v25_int8.tflite')
cc_path = os.path.join(CURRENT_DIR, 'model_data_v25.cc')
h_path = os.path.join(CURRENT_DIR, 'model_data_v25.h')

print(f"[*] Loading Keras model from: {model_path}")
model = tf.keras.models.load_model(model_path)

# Load representative dataset for quantization
print("[*] Loading representative dataset for INT8 Quantization...")
cache_file = os.path.join(ROOT_DIR, 'train_cache_v18_v19_idle_trans', 'X_train.npy')
X_train = np.load(cache_file)

# Sử dụng 300 mẫu ngẫu nhiên để đại diện phân phối
np.random.seed(42)
random_indices = np.random.choice(len(X_train), size=300, replace=False)

def representative_data_gen():
    for i in random_indices:
        features = X_train[i:i+1].astype(np.float32)
        
        # --- Áp dụng Scaling cho V25 ---
        features[:, :, 0:3] = np.clip(features[:, :, 0:3], -8.0, 8.0)
        features[:, :, 0:3] = features[:, :, 0:3] / 8.0
        features[:, :, 3:6] = features[:, :, 3:6] / 2000.0
        
        yield [features]

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

// ResNet-1D v25 Model for Fall Detection (INT8 Quantized)
// Architecture: Conv1D(16) -> 4x(ResNet Block + SE Block), GAP+GMP, 5 Classes
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
