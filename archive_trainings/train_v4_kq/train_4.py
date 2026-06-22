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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Fix unicode hiển thị trên Windows console
if sys.stdout.encoding.lower() != 'utf-8':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Định nghĩa các thư mục dữ liệu
CURRENT_DIR = Path(r'c:\Users\hung.vumanh2\Documents\SisFall-PreProcessing')
DATA_DIR = CURRENT_DIR / 'SisFall_dataset_Windowed'
CACHE_DIR = CURRENT_DIR / 'train_cache'
CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR = Path(r'c:\Users\hung.vumanh2\Documents\SisFall-PreProcessing\train_v4_kq')
OUT_DIR.mkdir(exist_ok=True)

# 1. Ánh xạ 34 nhãn gốc thành 4 lớp (Đã gộp)
LABEL_MAP = {
    # Walk (0) - Bao gồm cả Vấp ngã hụt (D19)
    'D01': 0, 'D02': 0, 'D05': 0, 'D06': 0, 'D19': 0,
    # Run (1)
    'D03': 1, 'D04': 1,
    # Static / ADL (2) - Bao gồm Đứng, Ngồi, Nằm, Cúi người, Chuyển tư thế
    'D07': 2, 'D08': 2, 'D09': 2, 'D10': 2, 'D11': 2, 'D12': 2, 'D13': 2, 'D14': 2, 
    'D15': 2, 'D16': 2, 'D17': 2, 'D18': 2,
    # Fall (3)
    'F01': 3, 'F02': 3, 'F03': 3, 'F04': 3, 'F05': 3, 'F06': 3, 'F07': 3, 'F08': 3, 'F09': 3,
    'F10': 3, 'F11': 3, 'F12': 3, 'F13': 3, 'F14': 3, 'F15': 3
}

CLASS_NAMES = ['Walk', 'Run', 'Static/ADL', 'Fall']

# 2. Phân chia Subjects theo Leave-Subjects-Out (LSO)
TRAIN_SUBJECTS = {f"SA{i:02d}" for i in range(1, 19)} | {f"SE{i:02d}" for i in range(1, 9)}
VAL_SUBJECTS = {f"SA{i:02d}" for i in range(19, 22)} | {f"SE{i:02d}" for i in range(9, 12)}
TEST_SUBJECTS = {f"SA{i:02d}" for i in range(22, 24)} | {f"SE{i:02d}" for i in range(12, 16)}

def parse_filename_info(filename):
    """
    Phân tích tên file để lấy loại nhãn và ID đối tượng
    Ví dụ: D01_SA01_R01_W000 -> label_code='D01', subject_id='SA01'
    """
    parts = filename.split('_')
    label_code = parts[0]
    subject_id = parts[1]
    return label_code, subject_id

def load_single_csv(file_path):
    """Đọc dữ liệu từ một file CSV window"""
    try:
        label_code, subject_id = parse_filename_info(file_path.stem)
        if label_code not in LABEL_MAP:
            return None
            
        label = LABEL_MAP[label_code]
        
        # Đọc dữ liệu gia tốc và góc quay (6 cột: ax,ay,az,gx,gy,gz)
        df = pd.read_csv(file_path)
        if len(df) != 200:
            return None # Bỏ qua các file lỗi kích thước
            
        data = df.to_numpy()
        return data, label, subject_id
    except Exception as e:
        return None

def prepare_dataset():
    """Gộp dữ liệu từ các file CSV và chia tập Train/Val/Test (LSO)"""
    train_cache_x = CACHE_DIR / 'X_train.npy'
    train_cache_y = CACHE_DIR / 'y_train.npy'
    val_cache_x = CACHE_DIR / 'X_val.npy'
    val_cache_y = CACHE_DIR / 'y_val.npy'
    test_cache_x = CACHE_DIR / 'X_test.npy'
    test_cache_y = CACHE_DIR / 'y_test.npy'
    
    # Nếu đã có cache, load lên cho cực nhanh
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
    print(f"[*] Tìm thấy {total_files} file windowed. Bắt đầu tải dữ liệu song song...")
    
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []
    
    # Sử dụng ThreadPool để tăng tốc độ đọc I/O đĩa
    processed_count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(load_single_csv, all_files)
        for res in results:
            processed_count += 1
            if processed_count % 10000 == 0:
                print(f"  > Đã đọc {processed_count}/{total_files} file...")
                
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
    
    print("\n[*] Thống kê kích thước tập dữ liệu:")
    print(f"  > Train shape: {X_train.shape}, Nhãn: {np.bincount(y_train)}")
    print(f"  > Val shape:   {X_val.shape}, Nhãn: {np.bincount(y_val)}")
    print(f"  > Test shape:  {X_test.shape}, Nhãn: {np.bincount(y_test)}")
    
    # Lưu cache npy
    print("[*] Đang lưu cache dữ liệu để tái sử dụng ở các lần chạy sau...")
    np.save(train_cache_x, X_train)
    np.save(train_cache_y, y_train)
    np.save(val_cache_x, X_val)
    np.save(val_cache_y, y_val)
    np.save(test_cache_x, X_test)
    np.save(test_cache_y, y_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def build_model(input_shape):
    """Xây dựng kiến trúc lai CNN-LSTM cho HAR (Đã giảm dung lượng và thêm Regularization)"""
    model = Sequential([
        Input(shape=input_shape),
        
        # 1. Trích xuất đặc trưng với L2 Regularization và Batch Norm
        Conv1D(32, kernel_size=3, activation='relu', padding='same', 
               kernel_regularizer=regularizers.l2(0.001)),
        BatchNormalization(),
        Conv1D(32, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # 2. LSTM được thu gọn và thêm L2
        LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)),
        LSTM(32, kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.4),
        
        # 3. Lớp đầu ra thu gọn
        Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def main():
    # 1. Load và chuẩn bị dữ liệu
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_dataset()
    
    # 2. Xử lý Class Imbalance
    # Tính trọng số class_weight tự động để bù đắp sự mất cân bằng dữ liệu
    class_weights_vals = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(zip(np.unique(y_train), class_weights_vals))
    print(f"[*] Trọng số cân bằng các lớp (Class Weights): {class_weights}")
    
    # 3. Build mô hình
    input_shape = (X_train.shape[1], X_train.shape[2]) # (200, 6)
    model = build_model(input_shape)
    
    # 4. Thiết lập các Callbacks tối ưu huấn luyện
    checkpoint_path = OUT_DIR / 'best_model_v4.keras'
    callbacks = [
        # Dừng sớm nếu Validation Loss không giảm sau 15 epochs để tránh overfit
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        # Giảm learning rate khi Val Loss đi ngang để tối ưu sâu
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
        # Lưu checkpoint model tốt nhất dựa trên val_loss
        ModelCheckpoint(filepath=str(checkpoint_path), monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    # 5. Huấn luyện mô hình
    print("\n[*] Bắt đầu huấn luyện mô hình...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=256,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. Vẽ và lưu đồ thị chất lượng huấn luyện
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Độ chính xác (Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Độ mất mát (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'training_history_v4.png')
    plt.close()
    print("[*] Đã vẽ và lưu đồ thị lịch sử huấn luyện thành 'training_history.png'")
    
    # 7. Đánh giá chất lượng trên tập kiểm thử (Test Set) độc lập
    print("\n[*] Đang tải mô hình tốt nhất để đánh giá trên tập Test...")
    best_model = tf.keras.models.load_model(str(checkpoint_path))
    
    y_pred_probs = best_model.predict(X_test, batch_size=256)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Báo cáo kết quả
    report_str = "\n" + "="*50 + "\n"
    report_str += "BÁO CÁO PHÂN LOẠI TRÊN TẬP KIỂM THỬ (TEST SET)\n"
    report_str += "="*50 + "\n"
    cls_report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    report_str += cls_report + "\n"
    
    print(report_str)
    
    # Tính Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Vẽ Confusion Matrix đẹp mắt bằng Matplotlib thuần túy
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Ma trận nhầm lẫn (Confusion Matrix) trên Test Set')
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
    plt.yticks(tick_marks, CLASS_NAMES)
    
    # Thêm chỉ số text vào từng ô
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
                     
    plt.ylabel('Nhãn Thực Tế (True Label)')
    plt.xlabel('Nhãn Dự Đoán (Predicted Label)')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'confusion_matrix_v4.png')
    plt.close()
    print("[*] Đã vẽ và lưu ma trận nhầm lẫn thành 'confusion_matrix_v4.png'")
    
    report_str += "\n[*] Đã vẽ và lưu ma trận nhầm lẫn thành 'confusion_matrix_v4.png'\n"
    
    # Kiểm tra chỉ số Recall của lớp Fall (nhãn 3)
    fall_idx = 3
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
    
    # Ghi report ra file
    with open(OUT_DIR / 'report.txt', 'w', encoding='utf-8') as rf:
        rf.write(report_str)
    print(f"[*] Đã lưu toàn bộ báo cáo vào {{OUT_DIR / 'report.txt'}}")

if __name__ == '__main__':

    main()
