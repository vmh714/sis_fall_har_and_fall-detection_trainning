import json

cells = []

def add_markdown(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")]
    })

add_markdown("# 1. Kết nối Google Drive và Giải nén Dataset\nKhởi tạo môi trường, giải nén dữ liệu thô.")
add_code("""from google.colab import drive
import os
import shutil

drive.mount('/content/drive')
ZIP_PATH = '/content/drive/MyDrive/SisFall_dataset.zip'
WORKSPACE = '/content/workspace'

if not os.path.exists(WORKSPACE):
    os.makedirs(WORKSPACE)
    !unzip -q "{ZIP_PATH}" -d {WORKSPACE}
    print("Giải nén hoàn tất!")
else:
    print("Workspace đã tồn tại!")""")

add_markdown("# 2. Tiền xử lý (Extract & Preprocess - Step 1)\nChuyển đổi gia tốc (g), góc quay (rad/s) và downsample 100Hz.")
add_code("""import glob
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm

INPUT_DIR = Path('/content/workspace/SisFall_dataset')
OUTPUT_DIR = Path('/content/workspace/SisFall_dataset_Processed')

target_har = {'D01', 'D02', 'D03', 'D04', 'D05', 'D07', 'D08', 'D11', 'D12', 'D13', 'D15', 'D17', 'D18', 'D19'}
ACCEL_SCALE = 32.0 / 8192.0
GYRO_SCALE = (4000.0 / 65536.0) * (np.pi / 180.0)

all_files = list(INPUT_DIR.rglob('*.txt'))
files_to_process = [f for f in all_files if f.name.startswith('F') or f.name[:3] in target_har]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for file_path in tqdm(files_to_process, desc="Preprocessing"):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        data = []
        for line in lines:
            line = line.strip().rstrip(';')
            if not line: continue
            parts = line.split(',')
            if len(parts) >= 6:
                data.append([float(x) for x in parts[:6]])
                
        if not data: continue
        df = pd.DataFrame(data, columns=['ax', 'ay', 'az', 'gx', 'gy', 'gz'])
        
        df[['ax', 'ay', 'az']] *= ACCEL_SCALE
        df[['gx', 'gy', 'gz']] *= GYRO_SCALE
        df = df.iloc[::2, :].reset_index(drop=True)
        
        rel_path = file_path.relative_to(INPUT_DIR)
        out_file = OUTPUT_DIR / rel_path.parent / (file_path.stem + '.csv')
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)
    except Exception as e:
        pass
print("Hoàn tất Step 1!")""")

add_markdown("# 3. Cắt Window (Windowing - Step 2)\nÁp dụng kỹ thuật cắt Peak và Sliding window.")
add_code("""from scipy.signal import find_peaks

WINDOW_OUTPUT_DIR = Path('/content/workspace/windowed_new')

def extract_window(df, center, window_size=200):
    start, end = center - window_size // 2, center + window_size // 2
    if start < 0: start, end = 0, window_size
    if end > len(df): end, start = len(df), max(0, len(df) - window_size)
    return df.iloc[start:end]

def classify_idle(win_df):
    ax_m, ay_m, az_m = win_df['ax'].abs().mean(), win_df['ay'].abs().mean(), win_df['az'].abs().mean()
    return 'Idle_StandSit' if (ay_m > ax_m and ay_m > az_m) else 'Idle_Lie'

all_processed = list(OUTPUT_DIR.rglob('*.csv'))
trans_har = {'D07', 'D08', 'D11', 'D12', 'D13', 'D15', 'D17'}
cont_har = {'D01', 'D02', 'D03', 'D04', 'D05', 'D18', 'D19'}

windows_generated = 0
for f in tqdm(all_processed, desc="Windowing"):
    df = pd.read_csv(f)
    if len(df) < 200: continue
    
    filename, prefix = f.stem, f.stem[:3]
    out_dir = WINDOW_OUTPUT_DIR / f.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    svm = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    
    if filename.startswith('F'):
        peak_idx = svm.argmax()
        for w_idx, shift in enumerate([-60, -30, 0, 30, 60]):
            win = extract_window(df, peak_idx + shift)
            if len(win) == 200:
                win.to_csv(out_dir / f"{filename}_Fall_W{w_idx:03d}.csv", index=False)
                windows_generated += 1
                
    elif prefix in trans_har:
        peaks, props = find_peaks(svm, height=1.0, distance=200)
        selected_peaks = np.sort(peaks[np.argsort(props['peak_heights'])[-2:]]) if len(peaks) >= 2 else peaks
        
        for i, peak in enumerate(selected_peaks):
            win = extract_window(df, peak)
            if len(win) == 200:
                win.to_csv(out_dir / f"{filename}_Trans_W{i:03d}.csv", index=False)
                windows_generated += 1
                
        for start in range(0, len(df) - 200 + 1, 100):
            center = start + 100
            if all(abs(center - p) >= 80 for p in selected_peaks):
                win = df.iloc[start:start+200]
                win.to_csv(out_dir / f"{filename}_{classify_idle(win)}_W{start//100:03d}.csv", index=False)
                windows_generated += 1
                
    elif prefix in cont_har:
        for start in range(0, len(df) - 200 + 1, 100):
            win = df.iloc[start:start+200]
            label = "Walk" if prefix in {'D01','D02','D05'} else "Run" if prefix in {'D03','D04'} else "Other"
            win.to_csv(out_dir / f"{filename}_{label}_W{start//100:03d}.csv", index=False)
            windows_generated += 1

print(f"Hoàn tất Windowing! Tổng số cửa sổ: {windows_generated}")""")

add_markdown("# 4. ML Pipeline I/O (DataPreprocessor & OutputReporter)\nĐây là 2 class cốt lõi chịu trách nhiệm tổ chức Dataloader, Caching, Split Subject và Report.")
add_code("""import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

class DataPreprocessor:
    def __init__(self, data_dir, cache_dir, class_names):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.class_names = class_names
        
        self.TRAIN_SUBJECTS = {f"SA{i:02d}" for i in range(1, 19)} | {f"SE{i:02d}" for i in range(1, 9)}
        self.VAL_SUBJECTS = {f"SA{i:02d}" for i in range(19, 22)} | {f"SE{i:02d}" for i in range(9, 12)}
        self.TEST_SUBJECTS = {f"SA{i:02d}" for i in range(22, 24)} | {f"SE{i:02d}" for i in range(12, 16)}

    def parse_filename_info(self, filename):
        if '_Trans_' in filename: return 'Trans'
        if '_StandSit_' in filename or '_Lie_' in filename: return 'Idle'
        prefix = filename[:3]
        if prefix in ['D01', 'D02', 'D05', 'D06']: return 'Walk'
        if prefix in ['D03', 'D04']: return 'Run'
        if prefix.startswith('F'): return 'Fall'
        return None

    def load_single_csv(self, file_path):
        try:
            filename = file_path.stem
            subject_id = filename.split('_')[1]
            label_str = self.parse_filename_info(filename)
            if label_str is None: return None
            label = self.class_names.index(label_str)
            df = pd.read_csv(file_path)
            if len(df) != 200: return None
            return df.to_numpy(), label, subject_id
        except: return None

    def load_or_create_dataset(self):
        train_x, train_y = self.cache_dir/'X_train.npy', self.cache_dir/'y_train.npy'
        val_x, val_y = self.cache_dir/'X_val.npy', self.cache_dir/'y_val.npy'
        test_x, test_y = self.cache_dir/'X_test.npy', self.cache_dir/'y_test.npy'
        
        if all(p.exists() for p in [train_x, train_y, val_x, val_y, test_x, test_y]):
            print("[*] Load cache...")
            return (np.load(train_x), np.load(train_y), np.load(val_x), np.load(val_y), np.load(test_x), np.load(test_y))

        print("[*] Đang đọc file CSV...")
        all_files = list(self.data_dir.rglob('*.csv'))
        X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            for res in tqdm(executor.map(self.load_single_csv, all_files), total=len(all_files)):
                if res is None: continue
                data, label, sub = res
                if sub in self.TRAIN_SUBJECTS: X_train.append(data); y_train.append(label)
                elif sub in self.VAL_SUBJECTS: X_val.append(data); y_val.append(label)
                elif sub in self.TEST_SUBJECTS: X_test.append(data); y_test.append(label)
                    
        X_train, y_train = np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.int32)
        X_val, y_val = np.array(X_val, dtype=np.float32), np.array(y_val, dtype=np.int32)
        X_test, y_test = np.array(X_test, dtype=np.float32), np.array(y_test, dtype=np.int32)
        
        # Thêm logic Data Augmentation (Scale 0.95-1.05 cho tập Train nếu cần thiết ở đây)
        # TODO: Add scaling logic here if needed
        
        np.save(train_x, X_train); np.save(train_y, y_train)
        np.save(val_x, X_val); np.save(val_y, y_val)
        np.save(test_x, X_test); np.save(test_y, y_test)
        
        return X_train, y_train, X_val, y_val, X_test, y_test

    def apply_preprocessing(self, X_train, X_val, X_test):
        for X in [X_train, X_val, X_test]:
            X[:, :, 0:3] = np.clip(X[:, :, 0:3], -8.0, 8.0) / 8.0
            X[:, :, 3:6] = X[:, :, 3:6] / 2000.0
        return X_train, X_val, X_test

    def get_balanced_class_weights(self, y_train, class_weight_modifiers=None):
        vals = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        weights = dict(zip(np.unique(y_train), vals))
        if class_weight_modifiers:
            for cls_name, mod in class_weight_modifiers.items():
                if cls_name in self.class_names:
                    weights[self.class_names.index(cls_name)] *= mod
        return weights

class OutputReporter:
    def __init__(self, out_dir, class_names):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self.class_names = class_names

    def plot_training_history(self, history, version="v_colab"):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1); plt.plot(history.history['accuracy'], label='Train'); plt.plot(history.history['val_accuracy'], label='Val'); plt.legend(); plt.title('Accuracy')
        plt.subplot(1, 2, 2); plt.plot(history.history['loss'], label='Train'); plt.plot(history.history['val_loss'], label='Val'); plt.legend(); plt.title('Loss')
        plt.tight_layout(); plt.savefig(self.out_dir / f'history_{version}.png'); plt.show()

    def evaluate_and_report(self, model, X_test, y_test, version="v_colab", fall_threshold=0.25):
        fall_idx = self.class_names.index('Fall')
        y_pred_probs = model.predict(X_test, batch_size=256)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_pred[y_pred_probs[:, fall_idx] >= fall_threshold] = fall_idx
        
        print(classification_report(y_test, y_pred, target_names=self.class_names, digits=4))
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8,6))
        plt.imshow(cm, cmap=plt.cm.Blues); plt.colorbar()
        plt.xticks(np.arange(len(self.class_names)), self.class_names, rotation=45)
        plt.yticks(np.arange(len(self.class_names)), self.class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i,j],'d'), ha="center", color="white" if cm[i,j] > cm.max()/2. else "black")
        plt.show()""")

add_markdown("# 5. Kiến trúc Model (Model Architecture)\nĐịnh nghĩa mô hình mạng Neural của bạn tại đây.")
add_code("""from tensorflow.keras.layers import Input, Conv1D, SeparableConv1D, BatchNormalization, Activation, Multiply, Add, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate, Dropout, Dense, Reshape
from tensorflow.keras.models import Model

# Bạn có thể copy paste các hàm ResNet-1D / CNN-LSTM vào đây
def build_model(input_shape=(200, 6), n_classes=5):
    inputs = Input(shape=input_shape)
    
    x = Conv1D(16, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), 
        metrics=['accuracy']
    )
    return model""")

add_markdown("# 6. Main Training Loop\nLớp quản lý luồng chạy.")
add_code("""from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

class FallDetectionTrainerColab(DataPreprocessor, OutputReporter):
    def __init__(self):
        data_dir = '/content/workspace/windowed_new'
        cache_dir = '/content/workspace/cache'
        out_dir = '/content/workspace/output'
        class_names = ['Walk', 'Run', 'Idle', 'Trans', 'Fall']
        
        DataPreprocessor.__init__(self, data_dir, cache_dir, class_names)
        OutputReporter.__init__(self, out_dir, class_names)
        self.version = "colab_v1"

trainer = FallDetectionTrainerColab()

# 1. Nạp và tiền xử lý
X_train, y_train, X_val, y_val, X_test, y_test = trainer.load_or_create_dataset()
X_train, X_val, X_test = trainer.apply_preprocessing(X_train, X_val, X_test)

# 2. Tính toán class weights
weight_modifiers = {'Fall': 3.0, 'Trans': 1.0, 'Idle': 1.0}
class_weights = trainer.get_balanced_class_weights(y_train, weight_modifiers)

# 3. One-hot
y_train_oh = to_categorical(y_train, num_classes=5)
y_val_oh = to_categorical(y_val, num_classes=5)

# 4. Build Model
model = build_model(input_shape=(200, 6), n_classes=5)
model.summary()

# 5. Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    # Có thể bật ModelCheckpoint nếu cần lưu file keras
]

# 6. Fit
history = model.fit(
    X_train, y_train_oh,
    validation_data=(X_val, y_val_oh),
    epochs=50,
    batch_size=64,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# 7. Evaluate
trainer.plot_training_history(history, version=trainer.version)
trainer.evaluate_and_report(model, X_test, y_test, version=trainer.version, fall_threshold=0.25)""")

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open("SisFall_Colab_Pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print("Đã tạo lại file SisFall_Colab_Pipeline.ipynb thành công!")
