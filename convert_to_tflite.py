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

# Cấu hình đường dẫn
MODEL_PATH = Path(r'c:\Users\hung.vumanh2\Documents\SisFall-PreProcessing\best_model.keras')
OUTPUT_DIR = Path(r'c:\Users\hung.vumanh2\Documents\SisFall-PreProcessing')
DATASET_DIR = Path(r'c:\Users\hung.vumanh2\Documents\SisFall-PreProcessing\SisFall_dataset_Windowed')

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
                # Model nhận đầu vào là (Batch, TimeSteps=200, Channels=9)
                # Cần thêm chiều batch size ở đầu: (1, 200, 9)
                yield [np.expand_dims(data, axis=0)]
        except Exception as e:
            continue

def convert_models():
    if not MODEL_PATH.exists():
        print(f"[!] Lỗi: Không tìm thấy file mô hình tại {MODEL_PATH}")
        return
    
    print(f"[*] Đang nạp mô hình Keras từ: {MODEL_PATH}")
    # Load mô hình Keras
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print("[*] Nạp mô hình thành công!")
    print(f"    Kích thước file gốc (.keras): {get_file_size(MODEL_PATH):.2f} KB\n")
    
    # ----------------------------------------------------
    # 1. Chuyển đổi Standard Float32 TFLite (Mặc định)
    # ----------------------------------------------------
    print("[*] 1. Đang chuyển đổi sang TFLite Float32 (Mặc định)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Hỗ trợ cấu trúc LSTM (cần các toán tử chọn lọc)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # Toán tử chuẩn của TFLite
        tf.lite.OpsSet.SELECT_TF_OPS    # Kích hoạt TensorFlow ops nếu có toán tử LSTM phức tạp
    ]
    tflite_float32 = converter.convert()
    float32_path = OUTPUT_DIR / "model_float32.tflite"
    with open(float32_path, "wb") as f:
        f.write(tflite_float32)
    print(f"    -> Đã lưu: {float32_path.name} | Kích thước: {get_file_size(float32_path):.2f} KB")

    # ----------------------------------------------------
    # 2. Chuyển đổi Float16 Quantization (Giảm 2 lần kích thước, chạy trên GPU/CPU nhanh)
    # ----------------------------------------------------
    print("\n[*] 2. Đang chuyển đổi sang TFLite Float16 Quantization...")
    converter_f16 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_f16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_f16.target_spec.supported_types = [tf.float16]
    converter_f16.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_float16 = converter_f16.convert()
    float16_path = OUTPUT_DIR / "model_float16.tflite"
    with open(float16_path, "wb") as f:
        f.write(tflite_float16)
    print(f"    -> Đã lưu: {float16_path.name} | Kích thước: {get_file_size(float16_path):.2f} KB")

    # ----------------------------------------------------
    # 3. Chuyển đổi Full INT8 Quantization (Tối ưu nhất cho TinyML / Vi điều khiển)
    # ----------------------------------------------------
    print("\n[*] 3. Đang chuyển đổi sang TFLite Full INT8 Quantization (Cần dữ liệu đại diện)...")
    try:
        converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_int8.representative_dataset = representative_data_gen
        
        # Ép buộc lượng tử hóa toàn bộ toán tử đầu vào/đầu ra sang int8
        converter_int8.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        # Nếu muốn đầu vào và đầu ra là float32 nhưng các trọng số bên trong chạy int8 siêu tốc:
        # converter_int8.inference_input_type = tf.int8
        # converter_int8.inference_output_type = tf.int8
        
        tflite_int8 = converter_int8.convert()
        int8_path = OUTPUT_DIR / "model_int8.tflite"
        with open(int8_path, "wb") as f:
            f.write(tflite_int8)
        print(f"    -> Đã lưu: {int8_path.name} | Kích thước: {get_file_size(int8_path):.2f} KB")
    except Exception as e:
        print(f"    [!] Không thể lượng tử hóa INT8 trực tiếp (Lỗi: {e}).")
        print("    [!] Lưu ý: Một số toán tử LSTM đặc thù có thể yêu cầu toán tử dự phòng (SELECT_TF_OPS) hoặc lượng tử hóa Dynamic Range.")
        
        # ----------------------------------------------------
        # 3b. Lượng tử hóa Dynamic Range (Lượng tử trọng số siêu nhỏ, chạy CPU cực nhanh)
        # ----------------------------------------------------
        print("\n[*] 3b. Thử nghiệm lượng tử hóa Dynamic Range Quantization thay thế...")
        converter_dr = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_dr.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_dr.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite_dr = converter_dr.convert()
        dr_path = OUTPUT_DIR / "model_dynamic_range.tflite"
        with open(dr_path, "wb") as f:
            f.write(tflite_dr)
        print(f"    -> Đã lưu: {dr_path.name} | Kích thước: {get_file_size(dr_path):.2f} KB")

    # ----------------------------------------------------
    # Bảng tổng kết
    # ----------------------------------------------------
    print("\n" + "="*50)
    print(" BẢNG SO SÁNH KÍCH THƯỚC MODEL")
    print("="*50)
    print(f" - Keras Model (.keras):       {get_file_size(MODEL_PATH):.2f} KB")
    if float32_path.exists():
        print(f" - TFLite Float32:             {get_file_size(float32_path):.2f} KB  (Đầy đủ, chuẩn xác nhất)")
    if float16_path.exists():
        print(f" - TFLite Float16:             {get_file_size(float16_path):.2f} KB  (Giảm 2x dung lượng, dùng GPU)")
    if (OUTPUT_DIR / "model_int8.tflite").exists():
        print(f" - TFLite INT8 Quantized:      {get_file_size(OUTPUT_DIR / 'model_int8.tflite'):.2f} KB  (Tối ưu nhất cho vi điều khiển/TinyML)")
    elif (OUTPUT_DIR / "model_dynamic_range.tflite").exists():
        print(f" - TFLite Dynamic Range:       {get_file_size(OUTPUT_DIR / 'model_dynamic_range.tflite'):.2f} KB  (Nén cao, tính toán CPU nhanh)")
    print("="*50)

if __name__ == "__main__":
    convert_models()
