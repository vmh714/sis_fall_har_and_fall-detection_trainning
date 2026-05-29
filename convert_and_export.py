import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

# Cấu hình UTF-8 cho stdout trên Windows để in được tiếng Việt
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# ====================================================
# CẤU HÌNH ĐƯỜNG DẪN TỚI MODEL V4
# ====================================================
MODEL_PATH = Path(r'c:\Users\hung.vumanh2\Documents\SisFall-PreProcessing\train_v6_kq\best_model_v6.keras')
OUTPUT_DIR = Path(r'c:\Users\hung.vumanh2\Documents\SisFall-PreProcessing\train_v6_kq')
DATASET_DIR = Path(r'c:\Users\hung.vumanh2\Documents\SisFall-PreProcessing\SisFall_dataset_Windowed')

# File đích C++
CC_PATH = OUTPUT_DIR / 'model_data.cc'
H_PATH = OUTPUT_DIR / 'model_data.h'

def get_file_size(file_path):
    """Lấy kích thước file tính bằng KB"""
    return os.path.getsize(file_path) / 1024

def representative_data_gen():
    """
    Trình phát dữ liệu đại diện để lượng tử hóa INT8 (Full Integer Quantization).
    Sẽ đọc ngẫu nhiên một vài file CSV mẫu từ tập dữ liệu đã cắt cửa sổ.
    """
    print("[*] Đang chuẩn bị dữ liệu đại diện cho lượng tử hóa INT8...")
    all_files = list(DATASET_DIR.rglob('*.csv'))
    if not all_files:
        raise ValueError(f"Không tìm thấy file CSV nào trong {DATASET_DIR} để tạo dữ liệu đại diện.")
    
    # Lấy ngẫu nhiên khoảng 100 mẫu để làm mẫu đại diện lượng tử hóa
    np.random.seed(42)
    sample_files = np.random.choice(all_files, min(100, len(all_files)), replace=False)
    
    for file_path in sample_files:
        try:
            df = pd.read_csv(file_path)
            if len(df) == 200:
                data = df.to_numpy().astype(np.float32)
                # Model nhận đầu vào là (Batch, TimeSteps=200, Channels=6)
                yield [np.expand_dims(data, axis=0)]
        except Exception as e:
            continue

def convert_tflite_to_cc(tflite_path):
    """Chuyển đổi file .tflite thành file C/C++ Header (.h) và Source (.cc)"""
    if not tflite_path.exists():
        print(f"[!] Lỗi: Không tìm thấy file TFLite tại: {tflite_path}")
        return

    print(f"\n[*] Đang đọc file TFLite: {tflite_path.name} để xuất ra C++...")
    with open(tflite_path, 'rb') as f:
        tflite_data = f.read()
    
    data_len = len(tflite_data)
    print(f"[*] Kích thước mô hình nhúng: {data_len} bytes (~{data_len / 1024:.2f} KB)")

    # 1. Tạo nội dung cho file Header (.h)
    print(f"[*] Đang tạo file Header: {H_PATH.name}...")
    h_content = f"""#ifndef MODEL_DATA_H_
#define MODEL_DATA_H_

// Khai báo mảng byte chứa mô hình TFLite và độ dài của nó
extern const unsigned char g_model_data[];
extern const unsigned int g_model_data_len;

#endif // MODEL_DATA_H_
"""
    with open(H_PATH, 'w', encoding='utf-8') as f:
        f.write(h_content)
    print(f"    -> Đã lưu thành công: {H_PATH.name}")

    # 2. Tạo nội dung cho file Source C++ (.cc)
    print(f"[*] Đang tạo file Source C++: {CC_PATH.name}...")
    
    # Định dạng mảng byte dạng hex (ví dụ: 0x1c, 0x00, ...)
    hex_bytes = []
    # In tối đa 12 byte trên một dòng để file code đẹp và gọn gàng
    bytes_per_line = 12
    
    for i, byte in enumerate(tflite_data):
        hex_bytes.append(f"0x{byte:02x}")
        
    # Tạo các dòng định dạng thụt lề
    lines = []
    for i in range(0, len(hex_bytes), bytes_per_line):
        chunk = hex_bytes[i:i + bytes_per_line]
        lines.append("  " + ", ".join(chunk))
        
    array_content = ",\n".join(lines)

    cc_content = f"""#include "model_data.h"

// Align 16-byte phù hợp cho tối ưu hóa phần cứng trên vi điều khiển (ARM Cortex-M, ESP32, v.v.)
alignas(16) const unsigned char g_model_data[] = {{
{array_content}
}};

const unsigned int g_model_data_len = {data_len};
"""
    
    with open(CC_PATH, 'w', encoding='utf-8') as f:
        f.write(cc_content)
    print(f"    -> Đã lưu thành công: {CC_PATH.name}")

    print("\n" + "="*50)
    print(" 🎉 XUẤT FILE C++ THÀNH CÔNG CHO TINYML!")
    print("="*50)
    print(f" - File Header: {H_PATH.name}")
    print(f" - File Source: {CC_PATH.name}")
    print(f" - Tên mảng:    g_model_data")
    print(f" - Độ dài mảng: {data_len} bytes")
    print("="*50)
    print("[*] Hướng dẫn sử dụng:")
    print("  1. Copy 2 file 'model_data.h' và 'model_data.cc' vào project C/C++ trên STM32, Arduino, hoặc ESP32.")
    print("  2. Gọi mô hình trong code C++ bằng cách sử dụng: ")
    print("     tflite::GetModel(g_model_data)")
    print("="*50)


def convert_pipeline():
    print("\n" + "="*50)
    print(" BẮT ĐẦU CHUỖI PIPELINE CHUYỂN ĐỔI: KERAS -> TFLITE -> C++")
    print("="*50)
    
    if not MODEL_PATH.exists():
        print(f"[!] Lỗi: Không tìm thấy file mô hình tại {MODEL_PATH}")
        return
    
    print(f"[*] Đang nạp mô hình Keras từ: {MODEL_PATH.name}")
    # Load mô hình Keras
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print("[*] Nạp mô hình thành công!")
    print(f"    Kích thước file gốc (.keras): {get_file_size(MODEL_PATH):.2f} KB\n")
    
    # --- BẮT BUỘC BATCH SIZE = 1 ĐỂ TRÁNH LỖI LSTM (TENSOR_LIST_RESERVE) TRONG TFLITE MICRO ---
    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec(shape=[1, 200, 6], dtype=tf.float32)
    )
    # ---------------------------------------------------------------------------------------

    # ----------------------------------------------------
    # 1. Chuyển đổi Standard Float32 TFLite (Mặc định)
    # ----------------------------------------------------
    print("[*] 1. Đang chuyển đổi sang TFLite Float32 (Mặc định)...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    tflite_float32 = converter.convert()
    float32_path = OUTPUT_DIR / "model_float32.tflite"
    with open(float32_path, "wb") as f:
        f.write(tflite_float32)
    print(f"    -> Đã lưu: {float32_path.name} | Kích thước: {get_file_size(float32_path):.2f} KB")

    # ----------------------------------------------------
    # 2. Chuyển đổi Float16 Quantization
    # ----------------------------------------------------
    print("\n[*] 2. Đang chuyển đổi sang TFLite Float16 Quantization...")
    converter_f16 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter_f16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_f16.target_spec.supported_types = [tf.float16]
    converter_f16.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    tflite_float16 = converter_f16.convert()
    float16_path = OUTPUT_DIR / "model_float16.tflite"
    with open(float16_path, "wb") as f:
        f.write(tflite_float16)
    print(f"    -> Đã lưu: {float16_path.name} | Kích thước: {get_file_size(float16_path):.2f} KB")

    # ----------------------------------------------------
    # 3. Lượng tử hóa Full INT8 (Bắt buộc cho TFLite Micro ESP-NN) - BỎ QUA VÌ LỖI COMPILER
    # ----------------------------------------------------
    print("\n[*] 3. Bỏ qua Full INT8 Quantization vì lỗi trình biên dịch TensorFlow với LSTM (segmentation fault).")
    
    # Bảng tổng kết Model Size
    print("\n" + "="*50)
    print(" BẢNG SO SÁNH KÍCH THƯỚC MODEL")
    print("="*50)
    print(f" - Keras Model (.keras):       {get_file_size(MODEL_PATH):.2f} KB")
    print(f" - TFLite Float32:             {get_file_size(float32_path):.2f} KB  <-- SẼ CHỌN BẢN NÀY ĐỂ XUẤT C++")
    print(f" - TFLite Float16:             {get_file_size(float16_path):.2f} KB")
    print("="*50)

    # ----------------------------------------------------
    # 4. Xuất TFLite nén tốt nhất ra C++
    # ----------------------------------------------------
    convert_tflite_to_cc(float32_path)

if __name__ == "__main__":
    convert_pipeline()
