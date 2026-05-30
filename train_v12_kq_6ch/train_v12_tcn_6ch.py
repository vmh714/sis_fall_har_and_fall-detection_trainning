import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, Add, Cropping1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# --- QUAN TRỌNG: KHÔNG SỬ DỤNG MIXED_FLOAT16 ---
# Ép kiểu float32 chuẩn để tương thích với TFLite Micro Firmware
tf.keras.backend.set_floatx('float32')

# Fix unicode hiển thị trên Windows console
if sys.stdout.encoding.lower() != 'utf-8':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

CURRENT_DIR = Path('/home/linh_linh/dataset/sis_fall_har_and_fall-detection_trainning')
DATA_DIR = CURRENT_DIR / 'SisFall_dataset_Windowed'
CACHE_DIR = CURRENT_DIR / 'train_cache'
CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR = CURRENT_DIR / 'train_v12_kq_6ch'
OUT_DIR.mkdir(exist_ok=True)

# Ánh xạ 34 nhãn gốc thành 4 lớp
LABEL_MAP = {
    'D01': 0, 'D02': 0, 'D05': 0, 'D06': 0, 'D19': 0,
    'D03': 1, 'D04': 1,
    'D07': 2, 'D08': 2, 'D09': 2, 'D10': 2, 'D11': 2, 'D12': 2, 'D13': 2, 'D14': 2, 
    'D15': 2, 'D16': 2, 'D17': 2, 'D18': 2,
    'F01': 3, 'F02': 3, 'F03': 3, 'F04': 3, 'F05': 3, 'F06': 3, 'F07': 3, 'F08': 3, 'F09': 3,
    'F10': 3, 'F11': 3, 'F12': 3, 'F13': 3, 'F14': 3, 'F15': 3
}

CLASS_NAMES = ['Walk', 'Run', 'Static/ADL', 'Fall']

TRAIN_SUBJECTS = {f"SA{i:02d}" for i in range(1, 19)} | {f"SE{i:02d}" for i in range(1, 9)}
VAL_SUBJECTS = {f"SA{i:02d}" for i in range(19, 22)} | {f"SE{i:02d}" for i in range(9, 12)}
TEST_SUBJECTS = {f"SA{i:02d}" for i in range(22, 24)} | {f"SE{i:02d}" for i in range(12, 16)}

def load_single_csv(file_path):
    try:
        parts = file_path.stem.split('_')
        label_code = parts[0]
        subject_id = parts[1]
        
        if label_code not in LABEL_MAP:
            return None
        label = LABEL_MAP[label_code]
        
        df = pd.read_csv(file_path)
        if len(df) != 200:
            return None
            
        data = df.to_numpy()
        return data, label, subject_id
    except Exception as e:
        return None

def prepare_dataset():
    train_cache_x = CACHE_DIR / 'X_train.npy'
    train_cache_y = CACHE_DIR / 'y_train.npy'
    val_cache_x = CACHE_DIR / 'X_val.npy'
    val_cache_y = CACHE_DIR / 'y_val.npy'
    test_cache_x = CACHE_DIR / 'X_test.npy'
    test_cache_y = CACHE_DIR / 'y_test.npy'
    
    if (train_cache_x.exists() and train_cache_y.exists() and 
        val_cache_x.exists() and val_cache_y.exists() and 
        test_cache_x.exists() and test_cache_y.exists()):
        print("[*] Phát hiện cache dữ liệu. Đang tải từ cache...")
        X_train = np.load(train_cache_x)
        y_train = np.load(train_cache_y)
        X_val = np.load(val_cache_x)
        y_val = np.load(val_cache_y)
        X_test = np.load(test_cache_x)
        y_test = np.load(test_cache_y)
        return X_train, y_train, X_val, y_val, X_test, y_test

    print("[*] Đang quét thư mục dữ liệu windowed (không dùng cache)...")
    all_files = list(DATA_DIR.rglob('*.csv'))
    
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(load_single_csv, all_files)
        for res in results:
            if res is None: continue
            data, label, subject_id = res
            if subject_id in TRAIN_SUBJECTS:
                X_train_list.append(data)
                y_train_list.append(label)
            elif subject_id in VAL_SUBJECTS:
                X_val_list.append(data)
                y_val_list.append(label)
            elif subject_id in TEST_SUBJECTS:
                X_test_list.append(data)
                y_test_list.append(label)
                
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    X_val = np.array(X_val_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.int32)
    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.int32)
    
    np.save(train_cache_x, X_train)
    np.save(train_cache_y, y_train)
    np.save(val_cache_x, X_val)
    np.save(val_cache_y, y_val)
    np.save(test_cache_x, X_test)
    np.save(test_cache_y, y_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def build_tcn_mcu(input_shape, n_classes=4):
    inputs = Input(shape=input_shape, name='input_layer')
    x = BatchNormalization()(inputs)
    for stack in range(2):
        for d in [1, 2, 4, 8]:
            residual = x
            x = Conv1D(32, kernel_size=3, dilation_rate=d, padding='valid', activation='relu', use_bias=True)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            crop_size = 2 * d
            residual = Cropping1D(cropping=(crop_size, 0))(residual)
            
            if residual.shape[-1] != 32:
                residual = Conv1D(32, 1, padding='valid')(residual)
            x = Add()([x, residual])

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(n_classes, activation='softmax', name='output_layer')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_dataset()
    
    # --- CẮT DỮ LIỆU CÒN 6 TRỤC ---
    print(f"[*] Dữ liệu gốc có shape: {X_train.shape}. Tiến hành cắt 6 trục (Accel+Gyro)...")
    X_train = X_train[:, :, :6]
    X_val = X_val[:, :, :6]
    X_test = X_test[:, :, :6]
    print(f"[*] Dữ liệu sau khi cắt: {X_train.shape}")
    
    class_weights_vals = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(zip(np.unique(y_train), class_weights_vals))
    
    # Ở v12, ta giảm mức phạt xuống x3 để nhường não cho Walk và Static
    fall_idx = CLASS_NAMES.index('Fall')
    class_weights[fall_idx] *= 3.0
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_tcn_mcu(input_shape, n_classes=len(CLASS_NAMES))
    
    checkpoint_path = OUT_DIR / 'best_model_v12_6ch.keras'
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=str(checkpoint_path), monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    print("\n[*] Bắt đầu huấn luyện mô hình TCN v12 (Phiên bản 6 trục MCU)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Độ chính xác (Accuracy)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Độ mất mát (Loss)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'training_history_v12_6ch.png')
    plt.close()
    
    print("\n[*] Đang đánh giá trên tập Test...")
    best_model = tf.keras.models.load_model(str(checkpoint_path))
    
    y_pred_probs = best_model.predict(X_test, batch_size=256)
    
    # Giải pháp B: Điều chỉnh ngưỡng ra quyết định (Threshold = 0.25)
    fall_probs = y_pred_probs[:, fall_idx]
    y_pred = np.argmax(y_pred_probs, axis=1) # Dự đoán chuẩn ban đầu
    y_pred[fall_probs >= 0.25] = fall_idx    # Ghi đè Fall nếu xác suất >= 25%
    
    report_str = "\n" + "="*50 + "\n"
    report_str += "BÁO CÁO PHÂN LOẠI TRÊN TẬP KIỂM THỬ - TCN v12 (6 Trục - Threshold 0.25)\n"
    report_str += "="*50 + "\n"
    
    cls_report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    report_str += cls_report + "\n"
    print(report_str)
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - TCN v12 (6 Channels)')
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
    plt.yticks(tick_marks, CLASS_NAMES)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
                     
    plt.ylabel('Nhãn Thực Tế')
    plt.xlabel('Nhãn Dự Đoán')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'confusion_matrix_v12_6ch.png')
    plt.close()
    
    true_falls = np.sum(y_test == fall_idx)
    detected_falls = cm[fall_idx, fall_idx]
    recall_fall = (detected_falls / true_falls) * 100 if true_falls > 0 else 0
    
    eval_str = "\n" + "="*50 + "\n"
    eval_str += "KẾT QUẢ ĐÁNH GIÁ CHUYÊN BIỆT LỚP TÉ NGÃ (FALL):\n"
    eval_str += f"  - Số ca ngã thực tế trong tập Test: {true_falls}\n"
    eval_str += f"  - Số ca ngã mô hình phát hiện đúng: {detected_falls}\n"
    eval_str += f"  - TỶ LỆ RECALL PHÁT HIỆN TÉ NGÃ:    {recall_fall:.2f}%\n"
    
    if recall_fall >= 95.0:
        eval_str += "  => ĐẠT TIÊU CHUẨN! (Recall >= 95%)\n"
    else:
        eval_str += "  => CẦN CẢI THIỆN! (Recall < 95%)\n"
    eval_str += "="*50 + "\n"
    
    print(eval_str)
    report_str += eval_str
    
    with open(OUT_DIR / 'report_v12_6ch.txt', 'w', encoding='utf-8') as rf:
        rf.write(report_str)

    # ======================================================
    # TỰ ĐỘNG XUẤT TFLITE VÀ C++ FIRMWARE (BATCH = 1, CHANNELS = 6)
    # ======================================================
    print("\n" + "="*50)
    print(" XUẤT FIRMWARE C++ (6-CHANNELS)")
    print("="*50)
    
    firmware_dir = CURRENT_DIR / 'sis_fall_firmware_inference' / 'main'
    firmware_dir.mkdir(parents=True, exist_ok=True)
    
    tflite_path = firmware_dir / 'model_v12_6ch_float32.tflite'
    cc_path = firmware_dir / 'model_data.cc'
    h_path = firmware_dir / 'model_data.h'
    
    print("[*] Tạo Concrete Function với Batch Size = 1...")
    run_model = tf.function(lambda x: best_model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec(shape=[1, 200, 6], dtype=tf.float32)
    )

    print("[*] Chuyển đổi sang TFLite (Float32)...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    data_len = len(tflite_model)
    
    print("[*] Đang ghi mảng byte C++...")
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
    with open(h_path, 'w', encoding='utf-8') as f:
        f.write(h_content)

    hex_bytes = [f"0x{b:02x}" for b in tflite_model]
    bytes_per_line = 12
    lines = ["  " + ", ".join(hex_bytes[i:i + bytes_per_line]) for i in range(0, len(hex_bytes), bytes_per_line)]
    array_content = ",\\n".join(lines)

    cc_content = f"""#include "model_data.h"

#ifdef __GNUC__
__attribute__((aligned(16))) 
#else
alignas(16) 
#endif
const unsigned char g_model_data[] = {{
{array_content}
}};

const unsigned int g_model_data_len = {data_len};
"""
    with open(cc_path, 'w', encoding='utf-8') as f:
        f.write(cc_content)
        
    print(f"[*] Thành công rực rỡ! Đã lưu mã nguồn Firmware C++ tại: {firmware_dir}")

if __name__ == '__main__':
    main()
