import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')

dataset_dir = Path('SisFall_dataset_Windowed_new')
all_files = list(dataset_dir.rglob('*.csv'))

# Gom nhóm file theo nhãn
labels = ['Fall', 'Trans', 'Idle_StandSit', 'Idle_Lie', 'Walk', 'Run']
files_by_label = {lbl: [] for lbl in labels}

for f in all_files:
    name = f.stem
    for lbl in labels:
        if f'_{lbl}_' in name:
            files_by_label[lbl].append(f)
            break

# Chọn ngẫu nhiên 1 file cho mỗi nhãn
selected_files = {}
for lbl in labels:
    if files_by_label[lbl]:
        selected_files[lbl] = random.choice(files_by_label[lbl])

# Vẽ đồ thị (6 subplots)
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

for i, lbl in enumerate(labels):
    ax = axes[i]
    if lbl in selected_files:
        f_path = selected_files[lbl]
        df = pd.read_csv(f_path)
        svm = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
        
        ax.plot(svm, label='SVM', color='black', linewidth=2, alpha=0.8)
        ax.plot(df['ax'], label='ax', alpha=0.7)
        ax.plot(df['ay'], label='ay', alpha=0.7)
        ax.plot(df['az'], label='az', alpha=0.7)
        
        ax.set_title(f"{lbl}: {f_path.name}")
        ax.set_ylim([-2, 5]) # Scale cố định để dễ so sánh
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title(f"{lbl}: KHÔNG TÌM THẤY DATA")

plt.tight_layout()
out_img = r'C:\Users\vuman\.gemini\antigravity-ide\brain\a0ec8992-5c83-4a74-b313-17c1457525ef\random_windows_preview.png'
plt.savefig(out_img)
print(f"Đã lưu ảnh tại {out_img}")
