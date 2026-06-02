#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script đánh giá hiệu năng (Performance Evaluation) của model TCN v23 trên Firmware ESP32
dựa trên file kết quả inference_results.csv.
Tự động cài đặt thư viện thiếu (pandas, numpy, matplotlib, seaborn, scikit-learn).
"""

import sys
import os
import subprocess

# Đảm bảo in được tiếng Việt trên console Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Hỗ trợ các phiên bản Python cũ hơn không có reconfigure
        pass

# 1. Tự động kiểm tra và cài đặt thư viện thiếu
REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn"
}

missing_packages = []
for module_name, pip_name in REQUIRED_PACKAGES.items():
    try:
        __import__(module_name)
    except ImportError:
        missing_packages.append(pip_name)

if missing_packages:
    print("=" * 60)
    print(f"[!] Thiếu các thư viện cần thiết: {', '.join(missing_packages)}")
    print("[*] Đang tiến hành cài đặt tự động qua pip...")
    print("=" * 60)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("\n[+] Đã cài đặt thành công tất cả thư viện!")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n[LỖI] Không thể tự động cài đặt thư viện: {e}")
        print("[*] Vui lòng chạy lệnh sau bằng tay trong terminal:")
        print(f"    pip install {' '.join(missing_packages)}")
        sys.exit(1)

# Import các thư viện sau khi đã chắc chắn được cài đặt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Thiết lập hiển thị
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Đường dẫn file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(CURRENT_DIR, "inference_results.csv")
OUT_IMG_PATH = os.path.join(CURRENT_DIR, "confusion_matrix_v23_firmware.png")
OUT_TXT_PATH = os.path.join(CURRENT_DIR, "report_v23_firmware.txt")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"[LỖI] Không tìm thấy file '{CSV_PATH}'!")
        print("[*] Hãy chắc chắn rằng file csv này đang nằm cùng thư mục với script.")
        return

    print(f"[*] Đang đọc dữ liệu từ '{CSV_PATH}'...")
    df = pd.read_csv(CSV_PATH)
    
    # 5 lớp chuyên biệt theo TCN v23
    CLASS_NAMES = ['Walk', 'Run', 'Idle', 'Trans', 'Fall']
    
    # Ánh xạ nhãn thực tế (Expected_Class) từ CSV về 5 lớp
    def map_expected(name):
        name = str(name).strip()
        if name in ['Idle_StandSit', 'Idle_Lie', 'Idle']:
            return 'Idle'
        if name in ['Transition', 'Trans']:
            return 'Trans'
        return name

    df['Expected_Mapped'] = df['Expected_Class'].apply(map_expected)
    
    # Ánh xạ nhãn dự đoán (Predicted_Class_Index) từ CSV về 5 lớp
    def map_predicted(idx):
        try:
            val = int(idx)
            if val == 0: return 'Walk'
            elif val == 1: return 'Run'
            elif val == 2: return 'Idle'
            elif val == 3: return 'Trans'
            elif val == 4: return 'Fall'
        except:
            pass
        return 'Unknown'
        
    df['Predicted_Mapped'] = df['Predicted_Class_Index'].apply(map_predicted)
    
    # Lọc bỏ các dòng lỗi hoặc không khớp (đảm bảo nhãn nằm trong CLASS_NAMES)
    valid_mask = df['Expected_Mapped'].isin(CLASS_NAMES) & df['Predicted_Mapped'].isin(CLASS_NAMES)
    df_valid = df[valid_mask]
    
    if len(df_valid) == 0:
        print("[LỖI] Không có dữ liệu hợp lệ để đánh giá!")
        return
        
    y_true = df_valid['Expected_Mapped'].values
    y_pred = df_valid['Predicted_Mapped'].values
    
    total_samples = len(df_valid)
    print(f"[+] Đã xử lý {total_samples}/{len(df)} mẫu hợp lệ.")
    
    # Tính toán Confusion Matrix và Classification Report
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    
    # Tạo chuỗi báo cáo
    report_str = "\n" + "="*50 + "\n"
    report_str += "BÁO CÁO PHÂN LOẠI TẬP KIỂM THỬ TRÊN FIRMWARE - TCN v23 (5 Lớp)\n"
    report_str += f"Dựa trên file: {CSV_PATH}\n"
    report_str += "="*50 + "\n"
    
    cls_report = classification_report(y_true, y_pred, labels=CLASS_NAMES, target_names=CLASS_NAMES, digits=4)
    report_str += cls_report + "\n"
    
    # Đánh giá chuyên biệt lớp té ngã
    true_falls = np.sum(y_true == 'Fall')
    fall_class_idx = CLASS_NAMES.index('Fall')
    detected_falls = cm[fall_class_idx, fall_class_idx]
    recall_fall = (detected_falls / true_falls) * 100 if true_falls > 0 else 0
    
    eval_str = "="*50 + "\n"
    eval_str += "KẾT QUẢ ĐÁNH GIÁ CHUYÊN BIỆT LỚP TÉ NGÃ (FALL) TRÊN FIRMWARE:\n"
    eval_str += f"  - Số ca ngã thực tế: {true_falls}\n"
    eval_str += f"  - Số ca ngã phát hiện đúng: {detected_falls}\n"
    eval_str += f"  - TỶ LỆ RECALL TÉ NGÃ (FIRMWARE): {recall_fall:.2f}%\n"
    eval_str += "="*50 + "\n"
    
    report_str += eval_str
    print(report_str)
    
    # Lưu báo cáo text
    with open(OUT_TXT_PATH, 'w', encoding='utf-8') as rf:
        rf.write(report_str)
    print(f"[+] Đã lưu báo cáo chi tiết vào file: '{OUT_TXT_PATH}'")
    
    # Vẽ Confusion Matrix bằng Seaborn (giao diện premium)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={"size": 14, "weight": "bold"})
    
    plt.title('Confusion Matrix - TCN v23 on Firmware (5 Classes)', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Nhãn Thực Tế (Ground Truth)', fontsize=12, fontweight='bold')
    plt.xlabel('Nhãn Dự Đoán (Firmware Predict)', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    
    # Lưu ảnh ma trận nhầm lẫn
    plt.savefig(OUT_IMG_PATH, dpi=300)
    plt.close()
    
    print(f"[+] Đã vẽ và lưu ma trận nhầm lẫn thành file: '{OUT_IMG_PATH}'")
    print("\n>>> Đánh giá hoàn tất thành công! Bạn có thể xem kết quả ngay bây giờ.")

if __name__ == '__main__':
    main()
