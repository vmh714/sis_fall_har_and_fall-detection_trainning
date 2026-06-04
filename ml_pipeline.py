import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix

class DataPreprocessor:
    """Class xử lý logic nạp và tiền xử lý dữ liệu (Data Preprocessing)"""
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
            parts = filename.split('_')
            subject_id = parts[1]
            
            label_str = self.parse_filename_info(filename)
            if label_str is None:
                return None
                
            label = self.class_names.index(label_str)
            
            df = pd.read_csv(file_path)
            if len(df) != 200:
                return None
                
            data = df.to_numpy()
            return data, label, subject_id
        except Exception as e:
            return None

    def load_or_create_dataset(self):
        train_cache_x = self.cache_dir / 'X_train.npy'
        train_cache_y = self.cache_dir / 'y_train.npy'
        val_cache_x = self.cache_dir / 'X_val.npy'
        val_cache_y = self.cache_dir / 'y_val.npy'
        test_cache_x = self.cache_dir / 'X_test.npy'
        test_cache_y = self.cache_dir / 'y_test.npy'
        
        if all(p.exists() for p in [train_cache_x, train_cache_y, val_cache_x, val_cache_y, test_cache_x, test_cache_y]):
            print(f"[*] Phát hiện cache dữ liệu tại {self.cache_dir}. Đang tải...")
            return (
                np.load(train_cache_x), np.load(train_cache_y),
                np.load(val_cache_x), np.load(val_cache_y),
                np.load(test_cache_x), np.load(test_cache_y)
            )

        print(f"[*] Đang quét thư mục dữ liệu windowed tại {self.data_dir}...")
        all_files = list(self.data_dir.rglob('*.csv'))
        
        X_train_list, y_train_list = [], []
        X_val_list, y_val_list = [], []
        X_test_list, y_test_list = [], []
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(self.load_single_csv, all_files)
            for res in results:
                if res is None: continue
                data, label, subject_id = res
                if subject_id in self.TRAIN_SUBJECTS:
                    X_train_list.append(data)
                    y_train_list.append(label)
                elif subject_id in self.VAL_SUBJECTS:
                    X_val_list.append(data)
                    y_val_list.append(label)
                elif subject_id in self.TEST_SUBJECTS:
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

    def apply_preprocessing(self, X_train, X_val, X_test):
        print("[*] Đang tiền xử lý dữ liệu (Cắt ngọn 8g và Normalize theo Max Range)...")
        for X in [X_train, X_val, X_test]:
            X[:, :, 0:3] = np.clip(X[:, :, 0:3], -8.0, 8.0)
            X[:, :, 0:3] = X[:, :, 0:3] / 8.0
            X[:, :, 3:6] = X[:, :, 3:6] / 2000.0
        return X_train, X_val, X_test

    def get_balanced_class_weights(self, y_train, class_weight_modifiers=None):
        class_weights_vals = class_weight.compute_class_weight(
            class_weight='balanced', classes=np.unique(y_train), y=y_train
        )
        weights = dict(zip(np.unique(y_train), class_weights_vals))
        
        if class_weight_modifiers:
            for cls_name, modifier in class_weight_modifiers.items():
                if cls_name in self.class_names:
                    idx = self.class_names.index(cls_name)
                    weights[idx] *= modifier
        return weights


class OutputReporter:
    """Class xử lý logic output: Báo cáo, Accuracy, Loss, Confusion Matrix"""
    def __init__(self, out_dir, class_names):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self.class_names = class_names

    def plot_training_history(self, history, version="v_x"):
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
        plt.savefig(self.out_dir / f'training_history_{version}.png')
        plt.close()
        print(f"[*] Đã lưu biểu đồ history tại {self.out_dir / f'training_history_{version}.png'}")

    def evaluate_and_report(self, model, X_test, y_test, version="v_x", fall_threshold=0.25):
        print("\n[*] Đang đánh giá trên tập Test...")
        fall_idx = self.class_names.index('Fall')
        
        y_pred_probs = model.predict(X_test, batch_size=256)
        
        fall_probs = y_pred_probs[:, fall_idx]
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_pred[fall_probs >= fall_threshold] = fall_idx
        
        report_str = "\n" + "="*50 + "\n"
        report_str += f"BÁO CÁO PHÂN LOẠI TẬP KIỂM THỬ - {version} (Threshold {fall_threshold})\n"
        report_str += "="*50 + "\n"
        
        cls_report = classification_report(y_test, y_pred, target_names=self.class_names, digits=4)
        report_str += cls_report + "\n"
        print(report_str)
        
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {version}')
        plt.colorbar()
        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, rotation=45)
        plt.yticks(tick_marks, self.class_names)
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
                         
        plt.ylabel('Nhãn Thực Tế')
        plt.xlabel('Nhãn Dự Đoán')
        plt.tight_layout()
        plt.savefig(self.out_dir / f'confusion_matrix_{version}.png')
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
        
        with open(self.out_dir / f'report_{version}.txt', 'w', encoding='utf-8') as rf:
            rf.write(report_str)
        print(f"[*] Đã lưu báo cáo tại {self.out_dir / f'report_{version}.txt'}")
