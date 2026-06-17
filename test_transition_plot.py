import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

sys.stdout.reconfigure(encoding='utf-8')

# Chọn một file ADL có transition (ví dụ D12: Ngồi -> Nằm -> Ngồi dậy)
file_path = 'SisFall_dataset_Processed/SA01/D12_SA01_R01.csv'
df = pd.read_csv(file_path)

# Tính Vector Magnitude (SVM)
svm = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)

# Tìm các đỉnh (Peak). Giữ khoảng cách ít nhất 200 mẫu (2 giây) giữa các đỉnh
peaks, properties = find_peaks(svm, height=1.0, distance=200)

# Lấy 2 đỉnh cao nhất
if len(peaks) >= 2:
    top_2_idx = np.argsort(properties['peak_heights'])[-2:]
    selected_peaks = peaks[top_2_idx]
    selected_peaks.sort()
else:
    selected_peaks = peaks

print(f"File {file_path}")
print(f"Tổng số mẫu: {len(df)}")
print(f"Các đỉnh được chọn: {selected_peaks}")

# Tạo plot
fig, axes = plt.subplots(4, 1, figsize=(12, 16))

# 1. Plot toàn bộ tín hiệu SVM và ax, ay, az
ax = axes[0]
ax.plot(svm, label='SVM', color='black', alpha=0.5)
ax.plot(df['ax'], label='ax', alpha=0.7)
ax.plot(df['ay'], label='ay', alpha=0.7)
ax.plot(df['az'], label='az', alpha=0.7)
for p in selected_peaks:
    ax.axvline(x=p, color='red', linestyle='--', label='Peak')
ax.set_title('Toàn bộ tín hiệu D12_SA01_R01 và các đỉnh phát hiện')
ax.legend(loc='upper right')

if len(selected_peaks) > 0:
    peak = selected_peaks[0]
    
    # Hàm tiện ích để lấy window (xử lý out-of-bounds)
    def get_window(center, shift):
        start = center + shift - 100
        end = center + shift + 100
        # Nếu start < 0 (peak ở đầu file)
        if start < 0:
            start = 0
            end = 200
        # Nếu end > len
        if end > len(df):
            end = len(df)
            start = end - 200
            
        if start < 0: # File quá ngắn (dưới 200 mẫu)
            start = 0
            
        return start, end

    shifts = [0, -20, 20]
    titles = ['Window Center (Shift=0)', 'Window Left (Shift=-20)', 'Window Right (Shift=+20)']
    
    for i, shift in enumerate(shifts):
        start, end = get_window(peak, shift)
        win_df = df.iloc[start:end]
        win_svm = svm.iloc[start:end]
        
        ax = axes[i+1]
        ax.plot(range(start, end), win_svm, label='SVM', color='black', alpha=0.5)
        ax.plot(range(start, end), win_df['ax'], label='ax', alpha=0.7)
        ax.plot(range(start, end), win_df['ay'], label='ay', alpha=0.7)
        ax.plot(range(start, end), win_df['az'], label='az', alpha=0.7)
        
        ax.axvline(x=peak, color='red', linestyle='--', linewidth=2, label='Actual Peak')
        
        # Vẽ ranh giới window
        ax.axvspan(start, end, color='green', alpha=0.1, label='Window bounds')
        
        ax.set_title(f"{titles[i]} | Range: [{start}, {end}]")
        ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(r'C:\Users\vuman\.gemini\antigravity-ide\brain\a0ec8992-5c83-4a74-b313-17c1457525ef\peak_windows_preview.png')
print("Đã lưu hình ảnh xem trước vào artifacts")
