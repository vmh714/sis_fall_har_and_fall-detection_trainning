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
from tensorflow.keras.layers import Input, Conv1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, Add, Cropping1D, GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Thiết lập float32
tf.keras.backend.set_floatx('float32')

# Fix unicode hiển thị trên Windows console
if sys.stdout.encoding.lower() != 'utf-8':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

CURRENT_DIR = Path('/home/linh_linh/dataset/sis_fall_har_and_fall-detection_trainning')
DATA_DIR = CURRENT_DIR / 'SisFall_dataset_Windowed'
CACHE_DIR = CURRENT_DIR / 'train_cache_v11_pro'
CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR = CURRENT_DIR / 'train_v12_kq_6ch_opt'
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

    print("[*] Cache không tồn tại, bắt buộc phải sinh lại dữ liệu...")
    return None, None, None, None, None, None

def build_tcn_mcu(input_shape, n_classes=4):
    inputs = Input(shape=input_shape, name='input_layer')
    
    # [Cải tiến C]: Thêm Data Augmentation với Gaussian Noise
    x = GaussianNoise(0.05)(inputs)
    x = BatchNormalization()(x)
    
    for stack in range(2):
        for d in [1, 2, 4, 8]:
            residual = x
            
            # [Cải tiến B]: Lớp Conv1D đầu tiên (stack=0, d=1) dùng 64 filters, các lớp sau giữ nguyên 32
            current_filters = 64 if (stack == 0 and d == 1) else 32
            
            x = Conv1D(current_filters, kernel_size=3, dilation_rate=d, padding='valid', activation='relu', use_bias=True)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            crop_size = 2 * d
            residual = Cropping1D(cropping=(crop_size, 0))(residual)
            
            if residual.shape[-1] != current_filters:
                residual = Conv1D(current_filters, 1, padding='valid')(residual)
            x = Add()([x, residual])

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(n_classes, activation='softmax', name='output_layer')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_dataset()
    if X_train is None:
        print("[!] Không có cache 9ch, script yêu cầu có sẵn cache.")
        return
        
    print(f"[*] Dữ liệu gốc có shape: {X_train.shape}. Tiến hành cắt 6 trục (Accel+Gyro)...")
    X_train = X_train[:, :, :6]
    X_val = X_val[:, :, :6]
    X_test = X_test[:, :, :6]
    print(f"[*] Dữ liệu sau khi cắt: {X_train.shape}")
    
    class_weights_vals = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(zip(np.unique(y_train), class_weights_vals))
    
    # [Cải tiến A]: Giảm mức phạt xuống x2.5 thay vì x3.0
    fall_idx = CLASS_NAMES.index('Fall')
    class_weights[fall_idx] *= 2.5
    print(f"[*] Trọng số lớp Fall được thiết lập: {class_weights[fall_idx]:.2f}")
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_tcn_mcu(input_shape, n_classes=len(CLASS_NAMES))
    
    checkpoint_path = OUT_DIR / 'best_model_v12_6ch_opt.keras'
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=str(checkpoint_path), monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    print("\n[*] Bắt đầu huấn luyện mô hình TCN v12_opt (6 trục)...")
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
    plt.savefig(OUT_DIR / 'training_history_v12_6ch_opt.png')
    plt.close()
    
    print("\n[*] Đang đánh giá trên tập Test...")
    best_model = tf.keras.models.load_model(str(checkpoint_path))
    
    y_pred_probs = best_model.predict(X_test, batch_size=256)
    
    # [Cải tiến A]: Tăng ngưỡng ra quyết định lên 0.30 (thay vì 0.25)
    fall_probs = y_pred_probs[:, fall_idx]
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_pred[fall_probs >= 0.30] = fall_idx
    
    report_str = "\n" + "="*50 + "\n"
    report_str += "BÁO CÁO PHÂN LOẠI TRÊN TẬP KIỂM THỬ - TCN v12 OPT (6 Trục - Threshold 0.30)\n"
    report_str += "="*50 + "\n"
    
    cls_report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    report_str += cls_report + "\n"
    print(report_str)
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - TCN v12 OPT (6 Channels)')
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
    plt.savefig(OUT_DIR / 'confusion_matrix_v12_6ch_opt.png')
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
    
    with open(OUT_DIR / 'report_v12_6ch_opt.txt', 'w', encoding='utf-8') as rf:
        rf.write(report_str)
        
    print("[*] Đã hoàn tất huấn luyện. Lược bỏ bước nén Firmware theo yêu cầu.")

if __name__ == '__main__':
    main()
