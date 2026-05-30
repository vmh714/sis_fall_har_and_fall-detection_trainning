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

if sys.stdout.encoding.lower() != 'utf-8':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

CURRENT_DIR = Path('/home/linh_linh/dataset/sis_fall_har_and_fall-detection_trainning')
DATA_DIR = CURRENT_DIR / 'tool_for_new_dataset' / 'SisFall_dataset_Windowed'
CACHE_DIR = CURRENT_DIR / 'train_cache_v16_v17_6ch'
CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR = CURRENT_DIR / 'train_v16_kq'
OUT_DIR.mkdir(exist_ok=True)

# 6 lớp chuyên biệt
CLASS_NAMES = ['Walk', 'Run', 'StandSit', 'Lie', 'Trans', 'Fall']

TRAIN_SUBJECTS = {f"SA{i:02d}" for i in range(1, 19)} | {f"SE{i:02d}" for i in range(1, 9)}
VAL_SUBJECTS = {f"SA{i:02d}" for i in range(19, 22)} | {f"SE{i:02d}" for i in range(9, 12)}
TEST_SUBJECTS = {f"SA{i:02d}" for i in range(22, 24)} | {f"SE{i:02d}" for i in range(12, 16)}

def parse_filename_info(filename):
    if '_Trans_' in filename:
        return 'Trans'
    if '_StandSit_' in filename:
        return 'StandSit'
    if '_Lie_' in filename:
        return 'Lie'
    
    prefix = filename[:3]
    if prefix in ['D01', 'D02', 'D05', 'D06']:
        return 'Walk'
    if prefix in ['D03', 'D04']:
        return 'Run'
    if prefix.startswith('F'):
        return 'Fall'
        
    return None

def load_single_csv(file_path):
    try:
        filename = file_path.stem
        parts = filename.split('_')
        subject_id = parts[1]
        
        label_str = parse_filename_info(filename)
        if label_str is None:
            return None
            
        label = CLASS_NAMES.index(label_str)
        
        df = pd.read_csv(file_path)
        if len(df) != 200:
            return None
            
        # Dữ liệu mới đã có 6 features: ax, ay, az, gx, gy, gz
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
        print("[*] Phát hiện cache dữ liệu (6 classes). Đang tải...")
        X_train = np.load(train_cache_x)
        y_train = np.load(train_cache_y)
        X_val = np.load(val_cache_x)
        y_val = np.load(val_cache_y)
        X_test = np.load(test_cache_x)
        y_test = np.load(test_cache_y)
        return X_train, y_train, X_val, y_val, X_test, y_test

    print("[*] Đang quét thư mục dữ liệu windowed mới...")
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

def build_tcn_mcu(input_shape, n_classes=6):
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
    print(f"[*] Dữ liệu Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    class_weights_vals = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(zip(np.unique(y_train), class_weights_vals))
    
    fall_idx = CLASS_NAMES.index('Fall')
    class_weights[fall_idx] *= 3.0
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_tcn_mcu(input_shape, n_classes=len(CLASS_NAMES))
    
    checkpoint_path = OUT_DIR / 'best_model_v16.keras'
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=str(checkpoint_path), monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    print("\n[*] Bắt đầu huấn luyện mô hình TCN v16 (6 lớp - float32 - C++ firmware)...")
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
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'training_history_v16.png')
    plt.close()
    
    print("\n[*] Đang đánh giá trên tập Test...")
    best_model = tf.keras.models.load_model(str(checkpoint_path))
    
    y_pred_probs = best_model.predict(X_test, batch_size=256)
    
    fall_probs = y_pred_probs[:, fall_idx]
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_pred[fall_probs >= 0.25] = fall_idx
    
    report_str = "\n" + "="*50 + "\n"
    report_str += "BÁO CÁO PHÂN LOẠI TẬP KIỂM THỬ - TCN v16 (6 Lớp - Threshold 0.25)\n"
    report_str += "="*50 + "\n"
    
    cls_report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    report_str += cls_report + "\n"
    print(report_str)
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - TCN v16 (6 Classes)')
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
    plt.savefig(OUT_DIR / 'confusion_matrix_v16.png')
    plt.close()
    
    true_falls = np.sum(y_test == fall_idx)
    detected_falls = cm[fall_idx, fall_idx]
    recall_fall = (detected_falls / true_falls) * 100 if true_falls > 0 else 0
    
    eval_str = "\n" + "="*50 + "\n"
    eval_str += "KẾT QUẢ ĐÁNH GIÁ CHUYÊN BIỆT LỚP TÉ NGÃ (FALL):\n"
    eval_str += f"  - Số ca ngã thực tế: {true_falls}\n"
    eval_str += f"  - Số ca ngã phát hiện đúng: {detected_falls}\n"
    eval_str += f"  - TỶ LỆ RECALL TÉ NGÃ: {recall_fall:.2f}%\n"
    eval_str += "="*50 + "\n"
    
    print(eval_str)
    report_str += eval_str
    
    with open(OUT_DIR / 'report_v16.txt', 'w', encoding='utf-8') as rf:
        rf.write(report_str)

if __name__ == '__main__':
    main()
