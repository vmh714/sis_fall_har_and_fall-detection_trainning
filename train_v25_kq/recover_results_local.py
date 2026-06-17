import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
import tensorflow as tf
from scipy import stats

# --- CẤU HÌNH ĐƯỜNG DẪN LOCAL ---
# Bạn hãy chắc chắn rằng dữ liệu cửa sổ (Windowed_new) và 17 file .keras đã được tải về đúng thư mục này
WINDOWED_DIR = Path('./workspace/SisFall_dataset_Windowed_new')
CACHE_DIR = Path('./workspace/cache')
DRIVE_OUT = Path('./KFold_Results') # Thư mục chứa 17 file .keras tải từ Drive về

class_names = ['Walk', 'Run', 'Idle', 'Trans', 'Fall']
all_labels = [0, 1, 2, 3, 4]

class KFoldDataManager:
    def __init__(self, windowed_dir, cache_dir, class_names):
        self.windowed_dir = Path(windowed_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names
        
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
        
    def load_and_group_by_subject(self):
        cache_file = self.cache_dir / 'subject_data.pkl'
        if cache_file.exists():
            print("[*] Loading subject data from cache...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
                
        all_files = list(self.windowed_dir.rglob('*.csv'))
        if len(all_files) == 0:
            raise FileNotFoundError(f"LỖI: Không tìm thấy file CSV nào trong {self.windowed_dir}. Bạn đã tải thư mục SisFall_dataset_Windowed_new về chưa?")
            
        subject_data = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            for res in tqdm(executor.map(self.load_single_csv, all_files), total=len(all_files), desc="Loading CSV"):
                if res is None: continue
                data, label, sub = res
                if sub not in subject_data:
                    subject_data[sub] = {'X': [], 'y': []}
                subject_data[sub]['X'].append(data)
                subject_data[sub]['y'].append(label)
                
        for sub in subject_data:
            subject_data[sub]['X'] = np.array(subject_data[sub]['X'], dtype=np.float32)
            subject_data[sub]['y'] = np.array(subject_data[sub]['y'], dtype=np.int32)
            
        with open(cache_file, 'wb') as f:
            pickle.dump(subject_data, f)
            
        return subject_data

    def apply_preprocessing(self, X):
        if X.ndim == 3 and X.shape[0] > 0:
            X[:, :, 0:3] = np.clip(X[:, :, 0:3], -8.0, 8.0) / 8.0
            X[:, :, 3:6] = X[:, :, 3:6] / 2000.0
        return X

    def get_data_for_subjects(self, subject_data, subjects):
        X_list, y_list = [], []
        for sub in subjects:
            if sub in subject_data:
                X_list.append(subject_data[sub]['X'])
                y_list.append(subject_data[sub]['y'])
        if len(X_list) == 0:
            return np.array([]), np.array([])
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        X = self.apply_preprocessing(X)
        return X, y

data_manager = KFoldDataManager(WINDOWED_DIR, CACHE_DIR, class_names)
subject_data = data_manager.load_and_group_by_subject()

sa_subjects = [f"SA{i:02d}" for i in range(1, 24)]
se_subjects = [f"SE{i:02d}" for i in range(1, 16)]
sa_subjects = [s for s in sa_subjects if s in subject_data]
se_subjects = [s for s in se_subjects if s in subject_data]

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

results_recovered = {'S1_Elderly': [], 'S2_Young': [], 'S3_TrainSE_TestSA': None, 'S4_TrainSA_TestSE': None, 'S5_Both': []}

def recover_metrics(X_test, y_test, run_name):
    model_path = DRIVE_OUT / f"model_{run_name}.keras"
    if not model_path.exists():
        print(f"LỖI: Không tìm thấy file mô hình: {model_path}")
        return None
        
    print(f"Đang lấy kết quả cho: {run_name} ...")
    model = tf.keras.models.load_model(str(model_path))
    y_pred_probs = model.predict(X_test, batch_size=256, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_pred[y_pred_probs[:, 4] >= 0.25] = 4 # Fall threshold
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    report_dict = classification_report(y_test, y_pred, labels=all_labels, target_names=class_names, output_dict=True, zero_division=0)
    
    tf.keras.backend.clear_session()
    return {'acc': float(acc), 'f1': float(f1), 'report': report_dict}

# --- BẮT ĐẦU CHẠY PHỤC HỒI ---
print("\n=== Phục hồi S1 ===")
for i, (train_idx, test_idx) in enumerate(kf.split(se_subjects)):
    X_te, y_te = data_manager.get_data_for_subjects(subject_data, [se_subjects[idx] for idx in test_idx])
    results_recovered['S1_Elderly'].append(recover_metrics(X_te, y_te, f"S1_Elderly_Fold{i+1}"))

print("\n=== Phục hồi S2 ===")
for i, (train_idx, test_idx) in enumerate(kf.split(sa_subjects)):
    X_te, y_te = data_manager.get_data_for_subjects(subject_data, [sa_subjects[idx] for idx in test_idx])
    results_recovered['S2_Young'].append(recover_metrics(X_te, y_te, f"S2_Young_Fold{i+1}"))

print("\n=== Phục hồi S3 ===")
X_te, y_te = data_manager.get_data_for_subjects(subject_data, sa_subjects)
results_recovered['S3_TrainSE_TestSA'] = recover_metrics(X_te, y_te, "S3_TrainSE_TestSA")

print("\n=== Phục hồi S4 ===")
X_te, y_te = data_manager.get_data_for_subjects(subject_data, se_subjects)
results_recovered['S4_TrainSA_TestSE'] = recover_metrics(X_te, y_te, "S4_TrainSA_TestSE")

print("\n=== Phục hồi S5 ===")
kf_sa = list(kf.split(sa_subjects))
kf_se = list(kf.split(se_subjects))
for i in range(n_splits):
    test_subs = [sa_subjects[idx] for idx in kf_sa[i][1]] + [se_subjects[idx] for idx in kf_se[i][1]]
    X_te, y_te = data_manager.get_data_for_subjects(subject_data, test_subs)
    results_recovered['S5_Both'].append(recover_metrics(X_te, y_te, f"S5_Both_Fold{i+1}"))

print("\n" + "="*50)
print("=== BÁO CÁO KẾT QUẢ T-TEST ===")
print("="*50)

f1_s1 = [r['f1'] for r in results_recovered['S1_Elderly'] if r]
f1_s2 = [r['f1'] for r in results_recovered['S2_Young'] if r]
f1_s5 = [r['f1'] for r in results_recovered['S5_Both'] if r]

if f1_s1 and f1_s2:
    t_stat1, p_val1 = stats.ttest_ind(f1_s1, f1_s2)
    print(f"H1 (Elderly vs Young F1): T-stat={t_stat1:.4f}, p-value={p_val1:.4f}")

if f1_s5 and f1_s1:
    t_stat2, p_val2 = stats.ttest_ind(f1_s5, f1_s1)
    print(f"H2 (Both vs Elderly F1): T-stat={t_stat2:.4f}, p-value={p_val2:.4f}")

out_json = DRIVE_OUT / 'final_metrics.json'
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(results_recovered, f, ensure_ascii=False, indent=4)
print(f"\n[*] Đã khôi phục hoàn tất! File JSON đã được tạo thành công tại: {out_json}")
