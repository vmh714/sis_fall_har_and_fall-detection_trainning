#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script chuẩn bị dataset để test inference trên ESP32.
Mục đích: Chỉ trích xuất các file CSV thuộc tập Kiểm thử (Test Set) 
từ thư mục Windowed Dataset đầy đủ (chứa cả Train, Val, Test).
Việc này giúp tránh Data Leakage (ESP32 dự đoán trên dữ liệu đã học).

Hỗ trợ lấy đường dẫn tương ứng với cấu trúc PIPELINES trong test_inference_uart.py
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from test_inference_uart import PIPELINES
except ImportError:
    print("[LỖI] Không thể import PIPELINES. Đảm bảo chạy script cùng cấp test_inference_uart.py.")
    PIPELINES = {}

TEST_SUBJECTS = {f"SA{i:02d}" for i in range(22, 24)} | {f"SE{i:02d}" for i in range(12, 16)}
TARGET_HAR = {'D01', 'D02', 'D03', 'D04', 'D05', 'D07', 'D08', 'D11', 'D12', 'D13', 'D15', 'D17', 'D18', 'D19'}
TRANS_HAR = {'D07', 'D08', 'D11', 'D12', 'D13', 'D15', 'D17'}
STUMBLE_JUMP = {'D18', 'D19'}
CONT_HAR = {'D01', 'D02', 'D03', 'D04', 'D05'}

ACCEL_SCALE = 32.0 / 8192.0
GYRO_SCALE = 4000.0 / 65536.0 # dps (không x pi/180)
GYRO_RMS_TH = 20.0

def rolling_rms(x, w=21):
    return np.sqrt(np.convolve(x**2, np.ones(w)/w, mode='same'))

def extract_window(df, center, window_size=200):
    start, end = center - window_size // 2, center + window_size // 2
    if start < 0: start, end = 0, window_size
    if end > len(df): end, start = len(df), max(0, len(df) - window_size)
    return df.iloc[start:end]

def classify_idle(win_df):
    ax_m, ay_m, az_m = win_df['ax'].abs().mean(), win_df['ay'].abs().mean(), win_df['az'].abs().mean()
    return 'Idle_StandSit' if (ay_m > ax_m and ay_m > az_m) else 'Idle_Lie'

def gyro_event_peaks(df, min_len=10, gap=10):
    gmag = np.sqrt(df['gx'].values**2 + df['gy'].values**2 + df['gz'].values**2)
    grms = rolling_rms(gmag, 21)
    idx = np.where(grms > GYRO_RMS_TH)[0]
    peaks = []
    if len(idx):
        for grp in np.split(idx, np.where(np.diff(idx) > gap)[0] + 1):
            if len(grp) >= min_len:
                peaks.append(int(grp[0] + np.argmax(grms[grp[0]:grp[-1]+1])))
    return peaks

def process_file_v30(file_path, out_dir):
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
                
        if not data: return 0
        df = pd.DataFrame(data, columns=['ax', 'ay', 'az', 'gx', 'gy', 'gz'])
        df[['ax', 'ay', 'az']] *= ACCEL_SCALE
        df[['gx', 'gy', 'gz']] *= GYRO_SCALE
        df = df.iloc[::2, :].reset_index(drop=True)
        
        if len(df) < 200: return 0
        
        filename = file_path.stem
        prefix = filename[:3]
        windows_generated = 0
        
        if filename.startswith('F'):
            svm = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
            peak_idx = int(svm.values.argmax())
            for w_idx, shift in enumerate([-60, -30, 0, 30, 60]):
                win = extract_window(df, peak_idx + shift)
                if len(win) == 200:
                    win.to_csv(out_dir / f"{filename}_Fall_W{w_idx:03d}.csv", index=False)
                    windows_generated += 1
        elif prefix in STUMBLE_JUMP:
            gmag = np.sqrt(df['gx'].values**2 + df['gy'].values**2 + df['gz'].values**2)
            peak = int(rolling_rms(gmag, 21).argmax())
            win = extract_window(df, peak)
            if len(win) == 200:
                win.to_csv(out_dir / f"{filename}_Trans_P000.csv", index=False)
                windows_generated += 1
        elif prefix in TRANS_HAR:
            peaks = gyro_event_peaks(df)
            for i, p in enumerate(peaks):
                win = extract_window(df, p)
                if len(win) == 200:
                    win.to_csv(out_dir / f"{filename}_Trans_E{i:03d}.csv", index=False)
                    windows_generated += 1
            for start in range(0, len(df)-200+1, 100):
                center = start + 100
                if all(abs(center - p) >= 80 for p in peaks):
                    win = df.iloc[start:start+200]
                    idle_label = classify_idle(win)
                    win.to_csv(out_dir / f"{filename}_{idle_label}_W{start//100:03d}.csv", index=False)
                    windows_generated += 1
        elif prefix in CONT_HAR:
            for start in range(0, len(df)-200+1, 100):
                win = df.iloc[start:start+200]
                label = "Walk" if prefix in {'D01', 'D02', 'D05'} else "Run"
                win.to_csv(out_dir / f"{filename}_{label}_W{start//100:03d}.csv", index=False)
                windows_generated += 1
                
        return windows_generated
    except Exception as e:
        print(f"Lỗi khi xử lý {file_path.name}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Sinh trực tiếp dataset TEST từ raw txt theo chuẩn v30.")
    parser.add_argument("--pipeline", required=True, choices=list(PIPELINES.keys()), 
                        help="Tên pipeline (v30, v25...). Hiện tại hỗ trợ sinh data trực tiếp cho v30.")
    parser.add_argument("--workspace", type=str, 
                        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        help="Thư mục gốc của dự án (mặc định: tự phát hiện)")
    args = parser.parse_args()

    pipeline_cfg = PIPELINES.get(args.pipeline)
    
    if args.pipeline != "v30":
        print(f"[!] Hiện tại script này đã được tối ưu để sinh trực tiếp dữ liệu cho v30.")
        print(f"[!] Nếu muốn test {args.pipeline}, xin đảm bảo logic khớp v30, hoặc tải thư mục windowed cũ về.")
        
    src_folder = pipeline_cfg["folder"]
    dst_dir = Path(args.workspace) / f"{src_folder}_TEST"
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    raw_dir = Path(args.workspace) / "SisFall_dataset"
    if not raw_dir.exists():
        print(f"[LỖI] Không tìm thấy thư mục dữ liệu thô: {raw_dir}")
        return
        
    print(f"[*] Đang quét các file txt thô trong {raw_dir} (chỉ lấy tập TEST)...")
    all_files = list(raw_dir.rglob('*.txt'))
    
    files_to_process = []
    for f in all_files:
        filename = f.name
        parts = filename.split('_')
        if len(parts) < 2:
            continue
        subject_id = parts[1].replace('.txt', '')
        if subject_id in TEST_SUBJECTS:
            if filename.startswith('F') or filename[:3] in TARGET_HAR:
                files_to_process.append(f)
                
    print(f"[*] Đã tìm thấy {len(files_to_process)} file thô thuộc tập TEST. Bắt đầu xử lý...")
    
    total_windows = 0
    for f in tqdm(files_to_process, desc="Tiền xử lý + Windowing"):
        total_windows += process_file_v30(f, dst_dir)
        
    print("\n" + "=" * 60)
    print(f"[+] Đã TẠO thành công {total_windows} file CSV window cho tập TEST!")
    print(f"[*] Thư mục đầu ra: '{dst_dir}'")
    print(f"    Lệnh test: python test_inference_uart.py --pipeline {args.pipeline} --folder \"{dst_dir}\" --samples 100")
    print("=" * 60)

if __name__ == "__main__":
    import sys
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
    main()
