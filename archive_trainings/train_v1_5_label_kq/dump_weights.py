"""Bước A: chạy bằng Keras 3 (mặc định của venv).

Load best_model_colab_v1.keras (vá bỏ quantization_config nếu cần) rồi
dump toàn bộ trọng số ra .npz để bước B (Keras 2 legacy) nạp lại.
Tách 2 bước vì Keras 2 không đọc được định dạng .keras của Keras 3, mà
fuser LSTM của TFLite chỉ nhận diện pattern do Keras 2 sinh ra.
"""
import os
import sys
import json
import zipfile
import numpy as np
import tensorflow as tf

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(CURRENT_DIR, 'best_model_colab_v1.keras')
out_npz = os.path.join(CURRENT_DIR, 'v1_weights.npz')


def load_model_compat(path):
    try:
        return tf.keras.models.load_model(path)
    except (TypeError, ValueError):
        def strip(o):
            if isinstance(o, dict):
                o.pop('quantization_config', None)
                for v in o.values():
                    strip(v)
            elif isinstance(o, list):
                for v in o:
                    strip(v)
        patched = path.replace('.keras', '_patched.keras')
        with zipfile.ZipFile(path) as zi, \
             zipfile.ZipFile(patched, 'w', zipfile.ZIP_DEFLATED) as zo:
            for n in zi.namelist():
                d = zi.read(n)
                if n == 'config.json':
                    c = json.loads(d)
                    strip(c)
                    d = json.dumps(c).encode('utf-8')
                zo.writestr(n, d)
        return tf.keras.models.load_model(patched)


print(f"[*] (Keras {tf.keras.__version__ if hasattr(tf.keras,'__version__') else '3'}) Load: {model_path}")
model = load_model_compat(model_path)
weights = model.get_weights()  # list np.ndarray theo thứ tự tạo layer
np.savez(out_npz, *weights)
print(f"[*] Đã dump {len(weights)} mảng trọng số -> {out_npz}")
for i, w in enumerate(weights):
    print(f"    [{i}] {w.shape}")
