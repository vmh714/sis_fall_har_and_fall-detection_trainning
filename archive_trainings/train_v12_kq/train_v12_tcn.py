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
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Fix unicode hiển thị trên Windows console
if sys.stdout.encoding.lower() != 'utf-8':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

CURRENT_DIR = Path('/home/linh_linh/dataset/sis_fall_har_and_fall-detection_trainning')
DATA_DIR = CURRENT_DIR / 'SisFall_dataset_Windowed'
CACHE_DIR = CURRENT_DIR / 'train_cache_v11_pro'
CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR = CURRENT_DIR / 'train_v12_kq'
OUT_DIR.mkdir(exist_ok=True)

# 1. Ánh xạ 34 nhãn gốc thành 4 lớp (Đã gộp)
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

def parse_filename_info(filename):
    parts = filename.split('_')
    label_code = parts[0]
    subject_id = parts[1]
    return label_code, subject_id

def load_single_csv(file_path):
    try:
        label_code, subject_id = parse_filename_info(file_path.stem)
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
        print("[*] Phát hiện cache dữ liệu đã được gộp. Đang tải từ cache...")
        X_train = np.load(train_cache_x)
        y_train = np.load(train_cache_y)
        X_val = np.load(val_cache_x)
        y_val = np.load(val_cache_y)
        X_test = np.load(test_cache_x)
        y_test = np.load(test_cache_y)
        return X_train, y_train, X_val, y_val, X_test, y_test

    print("[*] Đang quét thư mục dữ liệu windowed...")
    all_files = list(DATA_DIR.rglob('*.csv'))
    total_files = len(all_files)
    
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []
    
    processed_count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(load_single_csv, all_files)
        for res in results:
            processed_count += 1
            if res is None:
                continue
                
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
            # Dùng padding='valid' thay vì 'causal' để tránh sinh ra Pad, SpaceToBatchND trong TFLite
            x = Conv1D(32, kernel_size=3, dilation_rate=d, padding='valid', activation='relu', use_bias=True)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            # Cắt bớt phần đầu của residual để khớp kích thước sequence với x (vì padding='valid' làm x bị ngắn đi)
            crop_size = 2 * d # (kernel_size - 1) * dilation_rate = (3-1) * d = 2d
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

def export_to_tflite_int8(keras_model, X_train, tflite_path):
    print("\n[*] Đang chuyển đổi sang TFLite Full INT8...")
    def representative_dataset():
        indices = np.random.choice(len(X_train), size=200, replace=False)
        for i in indices:
            sample = X_train[i:i+1].astype(np.float32)
            yield [sample]
            
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"[*] Đã lưu mô hình TFLite Full INT8 tại: {tflite_path}")

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_dataset()
    
    class_weights_vals = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(zip(np.unique(y_train), class_weights_vals))
    
    # Ở v12, ta giảm mức phạt xuống x3 để nhường não cho Walk và Static
    fall_idx = CLASS_NAMES.index('Fall')
    class_weights[fall_idx] *= 3.0
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_tcn_mcu(input_shape, n_classes=len(CLASS_NAMES))
    
    checkpoint_path = OUT_DIR / 'best_model_v12.keras'
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=str(checkpoint_path), monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    print("\n[*] Bắt đầu huấn luyện mô hình TCN v12...")
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
    plt.savefig(OUT_DIR / 'training_history_v12.png')
    plt.close()
    
    print("\n[*] Đang đánh giá trên tập Test...")
    # Reset policy về float32 để TFLite converter không bị dính ops float16 (tránh nổ MLIR FlatBuffer)
    mixed_precision.set_global_policy('float32')
    best_model = tf.keras.models.load_model(str(checkpoint_path))
    
    y_pred_probs = best_model.predict(X_test, batch_size=256)
    
    # Giải pháp B: Điều chỉnh ngưỡng ra quyết định (Threshold = 0.25)
    fall_probs = y_pred_probs[:, fall_idx]
    y_pred = np.argmax(y_pred_probs, axis=1) # Dự đoán chuẩn ban đầu
    y_pred[fall_probs >= 0.25] = fall_idx    # Ghi đè Fall nếu xác suất >= 25%
    
    report_str = "\n" + "="*50 + "\n"
    report_str += "BÁO CÁO PHÂN LOẠI TRÊN TẬP KIỂM THỬ (TEST SET) - TCN v12 (Threshold 0.25)\n"
    report_str += "="*50 + "\n"
    
    cls_report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    report_str += cls_report + "\n"
    print(report_str)
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Ma trận nhầm lẫn (Confusion Matrix) - TCN v12')
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
    plt.savefig(OUT_DIR / 'confusion_matrix_v12.png')
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
    
    with open(OUT_DIR / 'report_v12.txt', 'w', encoding='utf-8') as rf:
        rf.write(report_str)
        
    # tflite_path = OUT_DIR / 'model_v11_int8.tflite'
    # export_to_tflite_int8(best_model, X_train, tflite_path)

if __name__ == '__main__':
    main()
