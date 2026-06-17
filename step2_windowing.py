import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

def extract_window(df, center, window_size=200):
    """Trích xuất 1 window xung quanh center, xử lý padding nếu bị tràn viền."""
    start = center - window_size // 2
    end = center + window_size // 2
    
    # Ép viền nếu bị sát vách
    if start < 0:
        start = 0
        end = window_size
    if end > len(df):
        end = len(df)
        start = max(0, end - window_size)
        
    return df.iloc[start:end]

def classify_idle(win_df):
    """Phân loại window tĩnh thành StandSit hoặc Lie dựa trên trọng lực (g)"""
    # Lấy trung bình trị tuyệt đối của từng trục gia tốc
    ax_m = win_df['ax'].abs().mean()
    ay_m = win_df['ay'].abs().mean()
    az_m = win_df['az'].abs().mean()

    # Nếu Y chịu trọng lực nhiều nhất -> Đang đứng/ngồi
    if ay_m > ax_m and ay_m > az_m:
        return 'Idle_StandSit'
    else:
        return 'Idle_Lie'

# Chỉ AUGMENT Trans cho subject thuộc tập TRAIN -> tránh thổi phồng Val/Test.
TRAIN_SUBJECTS = {f"SA{i:02d}" for i in range(1, 19)} | {f"SE{i:02d}" for i in range(1, 9)}
TRANS_AUG_SHIFTS = [-20, 0, 20]   # dịch trước/sau 20 mẫu -> x3 mẫu Trans

def save_trans(df, peak, out_file_dir, filename, subject_id, tag):
    """Lưu 1 cửa sổ Trans tại peak; augment x3 (dịch +-20) nếu subject thuộc TRAIN."""
    shifts = TRANS_AUG_SHIFTS if subject_id in TRAIN_SUBJECTS else [0]
    n = 0
    for shift in shifts:
        win = extract_window(df, peak + shift)
        if len(win) == 200:
            out_name = out_file_dir / f"{filename}_Trans_{tag}_s{shift:+d}.csv"
            win.to_csv(out_name, index=False)
            n += 1
    return n

def process_file(file_path, output_dir):
    df = pd.read_csv(file_path)
    if len(df) < 200:
        return 0

    filename = file_path.stem
    is_fall = filename.startswith('F')
    subject_id = filename.split('_')[1]

    # Phân loại file HAR
    prefix = filename[:3]
    trans_har    = {'D07', 'D08', 'D11', 'D12', 'D13', 'D15', 'D17'}  # ngồi/đứng/nằm -> Trans + Idle
    cont_har     = {'D01', 'D02', 'D03', 'D04', 'D05'}                # đi/chạy liên tục (KHÔNG còn D18/D19)
    stumble_jump = {'D18', 'D19'}                                     # D18 vấp khi đi, D19 nhảy nhẹ

    windows_generated = 0

    # Tính SVM
    svm = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)

    out_file_dir = output_dir / file_path.parent.name
    out_file_dir.mkdir(parents=True, exist_ok=True)

    if is_fall:
        # Fall: Lấy 1 peak cao nhất (Impact), cắt 5 windows
        peak_idx = int(svm.argmax())
        shifts = [-60, -30, 0, 30, 60]

        for w_idx, shift in enumerate(shifts):
            win = extract_window(df, peak_idx + shift)
            if len(win) == 200:
                # Lưu window gốc
                out_name = out_file_dir / f"{filename}_Fall_W{w_idx:03d}.csv"
                win.to_csv(out_name, index=False)
                windows_generated += 1

    elif prefix in stumble_jump:
        # D18 (vấp khi đi) / D19 (nhảy nhẹ): CHỈ 1 cửa sổ Trans tại đỉnh SVM cao nhất.
        # Phần đi bộ còn lại BỎ HẾT (không Walk, không Idle) vì Walk đã quá nhiều.
        peak_idx = int(svm.argmax())
        windows_generated += save_trans(df, peak_idx, out_file_dir, filename, subject_id, 'P000')

    elif prefix in trans_har:
        # ADL Trans: Lấy 2 peak cao nhất
        peaks, props = find_peaks(svm, height=1.0, distance=200)

        if len(peaks) >= 2:
            top_2 = np.argsort(props['peak_heights'])[-2:]
            selected_peaks = np.sort(peaks[top_2])
        elif len(peaks) == 1:
            selected_peaks = peaks
        else:
            selected_peaks = [int(svm.argmax())] # Fallback

        # Cắt Trans Windows (mỗi đỉnh augment x3 nếu subject thuộc TRAIN)
        for i, peak in enumerate(selected_peaks):
            windows_generated += save_trans(df, int(peak), out_file_dir, filename, subject_id, f"W{i:03d}")

        # Cắt Idle Windows (những đoạn cách xa Peak >= 80 mẫu) - KHÔNG augment
        for start in range(0, len(df) - 200 + 1, 100):
            center = start + 100
            # Kiểm tra xem có cách xa TẤT CẢ các peak không
            is_idle = True
            for peak in selected_peaks:
                if abs(center - peak) < 80:
                    is_idle = False
                    break

            if is_idle:
                win = df.iloc[start:start+200]
                idle_label = classify_idle(win)
                w_idx = start // 100
                out_name = out_file_dir / f"{filename}_{idle_label}_W{w_idx:03d}.csv"
                win.to_csv(out_name, index=False)
                windows_generated += 1

    elif prefix in cont_har:
        # ADL Continuous: Cắt sliding window (step=100)
        for start in range(0, len(df) - 200 + 1, 100):
            win = df.iloc[start:start+200]
            w_idx = start // 100
            label = "Walk" if prefix in {'D01','D02','D05'} else "Run"
            out_name = out_file_dir / f"{filename}_{label}_W{w_idx:03d}.csv"
            win.to_csv(out_name, index=False)
            windows_generated += 1

    return windows_generated

def main():
    input_dir = Path('SisFall_dataset_Processed')
    output_dir = Path('windowed_new')
    
    if not input_dir.exists():
        print(f"Lỗi: Không tìm thấy thư mục {input_dir}")
        return
        
    all_files = list(input_dir.rglob('*.csv'))
    print(f"Bắt đầu cắt window cho {len(all_files)} files...")
    
    total_windows = 0
    for f in tqdm(all_files, desc="Windowing"):
        try:
            total_windows += process_file(f, output_dir)
        except Exception as e:
            print(f"Lỗi ở file {f.name}: {e}")
            
    print(f"\nHoàn tất! Tổng số windows được tạo ra: {total_windows}")
    print(f"Dữ liệu đã được lưu vào thư mục: {output_dir}")

if __name__ == '__main__':
    main()
