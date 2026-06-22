#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sinh CACHE TEST cho pipeline v32 (window 128/256) để bắn xuống ESP32 qua test_inference_uart.py.

- CHỈ lấy tập TEST (SA22-23 + SE12-15) -> tránh data leakage.
- Tiền xử lý + windowing GIỐNG HỆT notebook train_v32 (accel*32/8192 g; gyro*4000/65536 dps;
  downsample ::2 -> 100Hz; Fall cắt event-centered theo đỉnh accel-SVM, căn lệch FALL_LEFT_RATIO;
  Trans theo sự kiện gyro-RMS>20dps; Walk/Run/Idle sliding stride WINDOW//2).
- Output: <workspace>/SisFall_dataset_Windowed_v32_w<W>_TEST  (CSV cột ax,ay,az,gx,gy,gz; W dòng/ file).

Ví dụ:
  python prepare_test_dataset_v32.py --window 256
  python test_inference_uart.py --pipeline v32_w256 --samples 100
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):  # fallback nếu chưa cài tqdm
        return it

TEST_SUBJECTS = {f"SA{i:02d}" for i in range(22, 24)} | {f"SE{i:02d}" for i in range(12, 16)}
TARGET_HAR = {'D01', 'D02', 'D03', 'D04', 'D05', 'D07', 'D08', 'D11', 'D12', 'D13', 'D15', 'D17', 'D18', 'D19'}
TRANS_HAR = {'D07', 'D08', 'D11', 'D12', 'D13', 'D15', 'D17'}
STUMBLE_JUMP = {'D18', 'D19'}
CONT_HAR = {'D01', 'D02', 'D03', 'D04', 'D05'}

ACCEL_SCALE = 32.0 / 8192.0
GYRO_SCALE = 4000.0 / 65536.0   # dps (KHÔNG x pi/180 — khớp firmware)
GYRO_RMS_TH = 20.0

# Mặc định theo từng window (khớp cell cấu hình notebook v32)
WIN_DEFAULTS = {
    128: dict(shifts=[-24, -12, 0, 12, 24], left_ratio=0.45),
    256: dict(shifts=[-60, -30, 0, 30, 60], left_ratio=0.5),
}


def rolling_rms(x, w=21):
    return np.sqrt(np.convolve(x ** 2, np.ones(w) / w, mode='same'))


def extract_window(df, center, window_size, left_ratio=0.5):
    left = int(round(window_size * left_ratio))
    start, end = center - left, center - left + window_size
    if start < 0:
        start, end = 0, window_size
    if end > len(df):
        end, start = len(df), max(0, len(df) - window_size)
    return df.iloc[start:end]


def classify_idle(win_df):
    ax_m, ay_m, az_m = win_df['ax'].abs().mean(), win_df['ay'].abs().mean(), win_df['az'].abs().mean()
    return 'Idle_StandSit' if (ay_m > ax_m and ay_m > az_m) else 'Idle_Lie'


def gyro_event_peaks(df, min_len=10, gap=10):
    gmag = np.sqrt(df['gx'].values ** 2 + df['gy'].values ** 2 + df['gz'].values ** 2)
    grms = rolling_rms(gmag, 21)
    idx = np.where(grms > GYRO_RMS_TH)[0]
    peaks = []
    if len(idx):
        for grp in np.split(idx, np.where(np.diff(idx) > gap)[0] + 1):
            if len(grp) >= min_len:
                peaks.append(int(grp[0] + np.argmax(grms[grp[0]:grp[-1] + 1])))
    return peaks


def process_file(file_path, out_dir, W, shifts, left_ratio):
    idle_stride = W // 2
    idle_dist = int(round(W * 0.4))
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        data = []
        for line in lines:
            line = line.strip().rstrip(';')
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 6:
                data.append([float(x) for x in parts[:6]])
        if not data:
            return 0
        df = pd.DataFrame(data, columns=['ax', 'ay', 'az', 'gx', 'gy', 'gz'])
        df[['ax', 'ay', 'az']] *= ACCEL_SCALE
        df[['gx', 'gy', 'gz']] *= GYRO_SCALE
        df = df.iloc[::2, :].reset_index(drop=True)
        if len(df) < W:
            return 0

        filename = file_path.stem
        prefix = filename[:3]
        n = 0

        if filename.startswith('F'):
            svm = np.sqrt(df['ax'] ** 2 + df['ay'] ** 2 + df['az'] ** 2)
            peak_idx = int(svm.values.argmax())
            for w_idx, shift in enumerate(shifts):
                win = extract_window(df, peak_idx + shift, W, left_ratio)
                if len(win) == W:
                    win.to_csv(out_dir / f"{filename}_Fall_W{w_idx:03d}.csv", index=False)
                    n += 1
        elif prefix in STUMBLE_JUMP:
            gmag = np.sqrt(df['gx'].values ** 2 + df['gy'].values ** 2 + df['gz'].values ** 2)
            peak = int(rolling_rms(gmag, 21).argmax())
            win = extract_window(df, peak, W)
            if len(win) == W:
                win.to_csv(out_dir / f"{filename}_Trans_P000.csv", index=False)
                n += 1
        elif prefix in TRANS_HAR:
            peaks = gyro_event_peaks(df)
            for i, p in enumerate(peaks):
                win = extract_window(df, p, W)
                if len(win) == W:
                    win.to_csv(out_dir / f"{filename}_Trans_E{i:03d}.csv", index=False)
                    n += 1
            for start in range(0, len(df) - W + 1, idle_stride):
                center = start + W // 2
                if all(abs(center - p) >= idle_dist for p in peaks):
                    win = df.iloc[start:start + W]
                    win.to_csv(out_dir / f"{filename}_{classify_idle(win)}_W{start // idle_stride:03d}.csv", index=False)
                    n += 1
        elif prefix in CONT_HAR:
            for start in range(0, len(df) - W + 1, idle_stride):
                win = df.iloc[start:start + W]
                label = "Walk" if prefix in {'D01', 'D02', 'D05'} else "Run"
                win.to_csv(out_dir / f"{filename}_{label}_W{start // idle_stride:03d}.csv", index=False)
                n += 1
        return n
    except Exception as e:
        print(f"Lỗi khi xử lý {file_path.name}: {e}")
        return 0


def main():
    ap = argparse.ArgumentParser(description="Sinh cache TEST cho pipeline v32 (window 128/256).")
    ap.add_argument("--window", type=int, default=256, choices=[128, 256], help="Kích thước window (mặc định 256).")
    ap.add_argument("--left-ratio", type=float, default=None, help="Ghi đè FALL_LEFT_RATIO (mặc định theo window).")
    ap.add_argument("--workspace", type=str,
                    default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help="Thư mục gốc dự án (mặc định: tự phát hiện).")
    args = ap.parse_args()

    W = args.window
    cfg = WIN_DEFAULTS[W]
    shifts = cfg["shifts"]
    left_ratio = cfg["left_ratio"] if args.left_ratio is None else args.left_ratio

    ws = Path(args.workspace)
    raw_dir = ws / "SisFall_dataset"
    if not raw_dir.exists():
        print(f"[LỖI] Không tìm thấy dữ liệu thô: {raw_dir}")
        return
    dst_dir = ws / f"SisFall_dataset_Windowed_v32_w{W}_TEST"
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] window={W}, FALL_SHIFTS={shifts}, FALL_LEFT_RATIO={left_ratio}")
    print(f"[*] Quét file thô TEST trong {raw_dir} ...")
    files = []
    for f in raw_dir.rglob('*.txt'):
        parts = f.name.split('_')
        if len(parts) < 2:
            continue
        sid = parts[1].replace('.txt', '')
        if sid in TEST_SUBJECTS and (f.name.startswith('F') or f.name[:3] in TARGET_HAR):
            files.append(f)
    print(f"[*] {len(files)} file TEST. Bắt đầu windowing...")

    total = 0
    for f in tqdm(files, desc=f"Windowing v32_w{W} TEST"):
        total += process_file(f, dst_dir, W, shifts, left_ratio)

    print("\n" + "=" * 60)
    print(f"[+] Đã tạo {total} cửa sổ ({W} mẫu) cho tập TEST.")
    print(f"[*] Output: {dst_dir}")
    print(f"    Lệnh test: python test_inference_uart.py --pipeline v32_w{W} --samples 100")
    print("=" * 60)


if __name__ == "__main__":
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
    main()
