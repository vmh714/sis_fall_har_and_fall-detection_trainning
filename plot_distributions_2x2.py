import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập phong cách vẽ hình Times New Roman chuẩn học thuật
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 16
})

# Đường dẫn dataset
dataset_dir = r"d:\New folder\sis_fall_har_and_fall-detection_trainning\SisFall_dataset_Windowed_v30_TEST"
csv_files = glob.glob(os.path.join(dataset_dir, "*.csv"))

# Chọn ngẫu nhiên 3000 file đại diện để tăng tốc độ xử lý
np.random.seed(42)
if len(csv_files) > 3000:
    selected_files = np.random.choice(csv_files, 3000, replace=False)
else:
    selected_files = csv_files

print(f"Loading data from {len(selected_files)} windows...")
ax_list, ay_list, az_list = [], [], []
gx_list, gy_list, gz_list = [], [], []

for f in selected_files:
    try:
        df = pd.read_csv(f)
        ax_list.append(np.abs(df['ax'].values))
        ay_list.append(np.abs(df['ay'].values))
        az_list.append(np.abs(df['az'].values))
        gx_list.append(np.abs(df['gx'].values))
        gy_list.append(np.abs(df['gy'].values))
        gz_list.append(np.abs(df['gz'].values))
    except Exception as e:
        pass

# Gộp dữ liệu
ax_all = np.concatenate(ax_list)
ay_all = np.concatenate(ay_list)
az_all = np.concatenate(az_list)
svm_all = np.sqrt(ax_all**2 + ay_all**2 + az_all**2)

gx_all = np.concatenate(gx_list)
gy_all = np.concatenate(gy_list)
gz_all = np.concatenate(gz_list)
rms_all = np.sqrt(gx_all**2 + gy_all**2 + gz_all**2)

print("Data loaded successfully. Starting to plot...")

def plot_log_hist(data, ax, color, title, xlabel, ylabel, vline_val, vline_label, show_gravity=False):
    # Tự tính toán histogram tần suất
    counts, bins = np.histogram(data, bins=100)
    log_counts = np.zeros_like(counts, dtype=float)
    nonzero = counts > 0
    # Lấy Log10 của tần suất (hệ số 1, 2, 3, 4, 5...)
    log_counts[nonzero] = np.log10(counts[nonzero])
    
    width = np.diff(bins)
    ax.bar(bins[:-1], log_counts, width=width, align='edge', color=color, edgecolor='none', alpha=0.8)
    ax.axvline(vline_val, color='red', linestyle='--', linewidth=2, label=vline_label)
    if show_gravity:
        ax.axvline(1.0, color='black', linestyle=':', linewidth=1.5, label='Trong luc Trai Dat (1.0g)')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

# 1. Vẽ đồ thị Gia tốc 2x2
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Phân Bố Giá Trị Tuyệt Đối Của Gia Tốc (Accelerometer Distribution)", y=0.96, fontweight='bold')

# Accel X
plot_log_hist(ax_all, axes[0, 0], '#1f77b4', "Trục X (Gia tốc tuyệt đối)", "Gia tốc (g)", "Log10(Số lượng mẫu)", 8.0, 'Nguong hardware (8.0g)')

# Accel Y
plot_log_hist(ay_all, axes[0, 1], '#ff7f0e', "Trục Y (Gia tốc tuyệt đối)", "Gia tốc (g)", "Log10(Số lượng mẫu)", 8.0, 'Nguong hardware (8.0g)', show_gravity=True)

# Accel Z
plot_log_hist(az_all, axes[1, 0], '#2ca02c', "Trục Z (Gia tốc tuyệt đối)", "Gia tốc (g)", "Log10(Số lượng mẫu)", 8.0, 'Nguong hardware (8.0g)', show_gravity=True)

# Accel SVM
plot_log_hist(svm_all, axes[1, 1], '#d62728', "Độ lớn gia tốc tổng hợp (SVM)", "Gia tốc (g)", "Log10(Số lượng mẫu)", 8.0, 'Nguong hardware (8.0g)', show_gravity=True)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("accel_distribution_2x2.png", dpi=300)
plt.close()
print("Saved accel_distribution_2x2.png")

# 2. Vẽ đồ thị Vận tốc góc 2x2
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Phân Bố Giá Trị Tuyệt Đối Của Vận Tốc Góc (Gyroscope Distribution)", y=0.96, fontweight='bold')

# Gyro X
plot_log_hist(gx_all, axes[0, 0], '#9467bd', "Trục X (Vận tốc góc tuyệt đối)", "Vận tốc góc (dps)", "Log10(Số lượng mẫu)", 500.0, 'Nguong hardware (500 dps)')

# Gyro Y
plot_log_hist(gy_all, axes[0, 1], '#8c564b', "Trục Y (Vận tốc góc tuyệt đối)", "Vận tốc góc (dps)", "Log10(Số lượng mẫu)", 500.0, 'Nguong hardware (500 dps)')

# Gyro Z
plot_log_hist(gz_all, axes[1, 0], '#e377c2', "Trục Z (Vận tốc góc tuyệt đối)", "Vận tốc góc (dps)", "Log10(Số lượng mẫu)", 500.0, 'Nguong hardware (500 dps)')

# Gyro RMS
plot_log_hist(rms_all, axes[1, 1], '#7f7f7f', "Vận tốc góc tổng hợp (Gyro RMS)", "Vận tốc góc (dps)", "Log10(Số lượng mẫu)", 500.0, 'Nguong hardware (500 dps)')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("gyro_distribution_2x2.png", dpi=300)
plt.close()
print("Saved gyro_distribution_2x2.png")
