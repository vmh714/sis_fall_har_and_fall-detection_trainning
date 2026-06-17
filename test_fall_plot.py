import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

sys.stdout.reconfigure(encoding='utf-8')

file_path = 'SisFall_dataset_Processed/SA01/F01_SA01_R01.csv'
df = pd.read_csv(file_path)

svm = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)

# Tìm đỉnh ngã (impact) bằng Peak cao nhất (giống windowing_pipeline.py)
peak_idx = svm.argmax()
selected_peaks = [peak_idx]

print(f"File {file_path}")
print(f"Tổng số mẫu: {len(df)}")
print(f"Đỉnh va chạm (Impact Peak): {peak_idx}")

# Lấy luôn peak làm tâm
midpoint = peak_idx

print(f"Trung điểm (Midpoint): {midpoint}")

# Tạo plot
fig, axes = plt.subplots(4, 1, figsize=(12, 16))

# 1. Plot toàn bộ tín hiệu SVM và gia tốc
ax = axes[0]
ax.plot(svm, label='SVM', color='black', alpha=0.5)
ax.plot(df['ax'], label='ax', alpha=0.7)
ax.plot(df['ay'], label='ay', alpha=0.7)
ax.plot(df['az'], label='az', alpha=0.7)
for p in selected_peaks:
    ax.axvline(x=p, color='red', linestyle='--', label='Peak')
ax.axvline(x=midpoint, color='purple', linestyle='-', linewidth=2, label='Midpoint')
ax.set_title('Toàn bộ tín hiệu F01_SA01_R01 (Té ngã) và 2 đỉnh, trung điểm')
ax.legend(loc='upper right')

# Hàm lấy window
def get_window(center, shift):
    start = center + shift - 100
    end = center + shift + 100
    if start < 0:
        start = 0
        end = 200
    if end > len(df):
        end = len(df)
        start = max(0, end - 200)
    return start, end

shifts = [0, -50, 50]
titles = ['Window (Shift=0 tại Midpoint)', 'Window (Shift=-50)', 'Window (Shift=+50)']

for i, shift in enumerate(shifts):
    start, end = get_window(midpoint, shift)
    win_df = df.iloc[start:end]
    win_svm = svm.iloc[start:end]
    
    ax = axes[i+1]
    ax.plot(range(start, end), win_svm, label='SVM', color='black', alpha=0.5)
    ax.plot(range(start, end), win_df['ax'], label='ax', alpha=0.7)
    ax.plot(range(start, end), win_df['ay'], label='ay', alpha=0.7)
    ax.plot(range(start, end), win_df['az'], label='az', alpha=0.7)
    
    for p in selected_peaks:
        if start <= p <= end:
            ax.axvline(x=p, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=midpoint, color='purple', linestyle='-', linewidth=2, label='Midpoint')
    ax.axvspan(start, end, color='green', alpha=0.1, label='Window bounds')
    
    ax.set_title(f"{titles[i]} | Range: [{start}, {end}]")
    ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(r'C:\Users\vuman\.gemini\antigravity-ide\brain\a0ec8992-5c83-4a74-b313-17c1457525ef\peak_windows_fall_preview.png')
print("Đã lưu hình ảnh xem trước vào artifacts")
