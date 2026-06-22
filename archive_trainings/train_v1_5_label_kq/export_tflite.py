"""Bước B: convert v1 CNN-LSTM -> TFLite INT8 (full integer) cho ESP32-S3 / TFLM.

QUAN TRỌNG: dùng Keras 2 (tf-keras) qua TF_USE_LEGACY_KERAS=1.
Lý do: Keras 3 (mặc định TF 2.20) KHÔNG fuse LSTM -> converter sinh op WHILE,
khiến int8 quantize bị segfault và TFLM không chạy được. Fuser LSTM của TFLite
chỉ nhận diện pattern do Keras 2 sinh ra -> ra UNIDIRECTIONAL_SEQUENCE_LSTM thật.

Chạy 2 bước:
  1) python dump_weights.py     (Keras 3) -> tạo v1_weights.npz
  2) python export_tflite.py    (Keras 2) -> tạo model_v1_int8.tflite + .cc/.h
"""
import os
# Phải set TRƯỚC khi import tensorflow để kích hoạt Keras 2 legacy.
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import sys
import numpy as np
import tensorflow as tf

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
)
from tensorflow.keras.models import Model

try:
    import keras as _k
    _kver = _k.__version__
except Exception:
    _kver = '?'
print(f"[*] tf {tf.__version__} | keras {_kver} (legacy={os.environ.get('TF_USE_LEGACY_KERAS')})")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

weights_npz = os.path.join(CURRENT_DIR, 'v1_weights.npz')
tflite_path = os.path.join(CURRENT_DIR, 'model_v1_int8.tflite')
cc_path = os.path.join(CURRENT_DIR, 'model_data_v1.cc')
h_path = os.path.join(CURRENT_DIR, 'model_data_v1.h')


def build_cnn_lstm():
    """Kiến trúc v1 (khớp train_v1_5_label.py), batch cố định = 1 để LSTM fuse."""
    inp = Input(batch_shape=(1, 200, 6), dtype='float32')
    x = Conv1D(64, 3, activation='relu', padding='same')(inp)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(64)(x)
    x = Dropout(0.4)(x)
    x = Dense(32, activation='relu')(x)
    out = Dense(5, activation='softmax')(x)
    return Model(inp, out)


print("[*] Dựng lại kiến trúc (Keras 2) và nạp trọng số từ npz...")
if not os.path.exists(weights_npz):
    raise FileNotFoundError("Thiếu v1_weights.npz -> chạy: python dump_weights.py trước")
model = build_cnn_lstm()
npz = np.load(weights_npz)
weights = [npz[f'arr_{i}'] for i in range(len(npz.files))]
model.set_weights(weights)
print(f"[*] Đã nạp {len(weights)} mảng trọng số.")

# --- Representative dataset cho INT8 ---
print("[*] Loading representative dataset for INT8 Quantization...")
CACHE_CANDIDATES = ['train_cache_v18_v19_idle_trans', 'train_cache']
cache_file = None
for c in CACHE_CANDIDATES:
    p = os.path.join(ROOT_DIR, c, 'X_train.npy')
    if os.path.exists(p):
        cache_file = p
        print(f"[*] Dùng representative cache: {p}")
        break
if cache_file is None:
    raise FileNotFoundError(f"Không tìm thấy X_train.npy trong: {CACHE_CANDIDATES}")
X_train = np.load(cache_file)
assert X_train.shape[1:] == (200, 6), f"Cache shape {X_train.shape} != (*,200,6)"

np.random.seed(42)
random_indices = np.random.choice(len(X_train), size=300, replace=False)

def representative_data_gen():
    for i in random_indices:
        features = X_train[i:i+1].astype(np.float32)
        # Scaling khớp ml_pipeline.apply_preprocessing (v1):
        # acc (0:3): clip ±8g rồi /8 | gyro (3:6): /2000
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
print(f"[*] Saved TFLite binary to: {tflite_path}  ({len(tflite_model)} bytes)")

# --- Kiểm tra IO + op list (xác nhận LSTM đã fuse) ---
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
print(f"[*] Input : {inp['shape']} {inp['dtype']} scale/zp={inp['quantization']}")
print(f"[*] Output: {out['shape']} {out['dtype']} scale/zp={out['quantization']}")
print("[*] Ops in model: " + ", ".join(
    sorted({d['op_name'] for d in interpreter._get_ops_details()})))
# arena_used_bytes() chỉ có ở C++ API; xem log firmware sau Invoke() đầu tiên.

print("[*] Generating C array...")
hex_array = [f'0x{byte:02x}' for byte in tflite_model]
formatted_hex = ""
for i, h in enumerate(hex_array):
    formatted_hex += h + ', '
    if (i + 1) % 12 == 0:
        formatted_hex += '\n  '

cc_content = f"""#include "model_data.h"

// CNN-LSTM v1 Model for Fall Detection (INT8 Quantized)
// Architecture: Conv1D(64) -> Conv1D(64) -> MaxPool -> LSTM(128) -> LSTM(64) -> Dense(32) -> Dense(5)
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
