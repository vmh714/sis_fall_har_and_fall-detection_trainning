#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sinh CACHE TEST cho pipeline v25 (ResNet-1D, window 200) để bắn xuống ESP32 qua test_inference_uart.py.

Tái lập CHÍNH XÁC tiền xử lý + windowing của notebook v25 (SisFall_KFold_Experiments):
  - CHỈ lấy tập TEST (SA22-23 + SE12-15) -> tránh data leakage.
  - accel *= 32/8192 (g) ; gyro *= (4000/65536)*(pi/180) -> rad/s (KHÁC v32 dùng dps!).
  - downsample 200->100Hz bằng scipy decimate(q=2, IIR Chebyshev, zero_phase) chống aliasing.
  - window=200, CĂN GIỮA (center-100 .. center+100).
  - Fall: đỉnh accel-SVM, shifts [-60,-30,0,30,60].
  - D18/D19 (vấp/nhảy near-fall): đỉnh SVM, shifts [-30,-15,0,15,30] -> Trans (KHÔNG sinh Idle).
  - trans_har (D07,D08,D09,D10,D11,D12,D13,D14,D15,D17): find_peaks(SVM,height=1,distance=200) lấy
    top-2 đỉnh -> Trans (shift -20/0/20); vùng tĩnh (>=80 cách đỉnh) -> Idle (stride 100).
  - cont (D01,D02,D03,D04,D05): sliding stride 100 -> Walk (D01/D02/D05) / Run (D03/D04).

LƯU Ý FIRMWARE: gyro trong CSV là rad/s -> firmware phải chuẩn hoá gyro `/2000` (đặt
GYRO_NORM_DIV2000=1 trong tflite_wrapper.cpp), KHÔNG dùng clip(±500)/500 của v30/v32.

Output: <workspace>/SisFall_dataset_Windowed_v25_TEST  (CSV cột ax,ay,az,gx,gy,gz; 200 dòng/file).

Ví dụ:
  python prepare_test_dataset_v25.py
  python test_inference_uart.py --pipeline v25 --samples 100
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import decimate, find_peaks
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

TEST_SUBJECTS = {f"SA{i:02d}" for i in range(22, 24)} | {f"SE{i:02d}" for i in range(12, 16)}

# === Bộ nhãn HAR đúng theo notebook v25 (có D09/D10/D14, D18/D19) ===
TARGET_HAR = {'D01', 'D02', 'D03', 'D04', 'D05',
              'D07', 'D08', 'D09', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D17', 'D18', 'D19'}
TRANS_HAR = {'D07', 'D08', 'D09', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D17'}
NEARFALL_HAR = {'D18', 'D19'}
CONT_HAR = {'D01', 'D02', 'D03', 'D04', 'D05'}

ACCEL_SCALE = 32.0 / 8192.0
# DPS (= rad/s * 180/pi, tức KHÔNG *pi/180). Kiểm chứng thực tế: dps cho accuracy ĐÚNG
# (acc ~0.92, Trans ~0.86, Fall ~0.91); rad/s làm gyro ~0 -> Trans sập. -> v25 chuẩn hoá gyro dps/2000.
GYRO_SCALE = 4000.0 / 65536.0
WINDOW_SIZE = 200


def extract_window(df, center, window_size=WINDOW_SIZE):
    """Cửa sổ CĂN GIỮA quanh center, kẹp biên (giống extract_window notebook v25)."""
    start, end = center - window_size // 2, center + window_size // 2
    if start < 0:
        start, end = 0, window_size
    if end > len(df):
        end, start = len(df), max(0, len(df) - window_size)
    return df.iloc[start:end]


def classify_idle(win_df):
    ax_m, ay_m, az_m = win_df['ax'].abs().mean(), win_df['ay'].abs().mean(), win_df['az'].abs().mean()
    return 'Idle_StandSit' if (ay_m > ax_m and ay_m > az_m) else 'Idle_Lie'


def load_and_preprocess(file_path):
    """Đọc raw .txt -> scale (accel g, gyro rad/s) -> decimate q=2 (200->100Hz). Trả DataFrame hoặc None."""
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
        return None
    df = pd.DataFrame(data, columns=['ax', 'ay', 'az', 'gx', 'gy', 'gz'])
    df[['ax', 'ay', 'az']] *= ACCEL_SCALE
    df[['gx', 'gy', 'gz']] *= GYRO_SCALE
    # decimate q=2 cần đủ mẫu cho padlen của IIR; thiếu thì rơi về lấy mẫu cách 2.
    if len(df) > 27:
        arr = decimate(df.to_numpy(), q=2, axis=0, ftype='iir', zero_phase=True)
        df = pd.DataFrame(arr.astype(np.float32), columns=df.columns)
    else:
        df = df.iloc[::2, :].reset_index(drop=True)
    return df


def process_file(file_path, out_dir):
    try:
        df = load_and_preprocess(file_path)
        if df is None or len(df) < WINDOW_SIZE:
            return 0
        filename = file_path.stem
        prefix = filename[:3]
        svm = np.sqrt(df['ax'] ** 2 + df['ay'] ** 2 + df['az'] ** 2)
        n = 0

        if filename.startswith('F'):
            peak_idx = int(svm.values.argmax())
            for w_idx, shift in enumerate([-60, -30, 0, 30, 60]):
                win = extract_window(df, peak_idx + shift)
                if len(win) == WINDOW_SIZE:
                    win.to_csv(out_dir / f"{filename}_Fall_W{w_idx:03d}.csv", index=False)
                    n += 1

        elif prefix in NEARFALL_HAR:
            peak_idx = int(svm.values.argmax())
            for w_idx, shift in enumerate([-30, -15, 0, 15, 30]):
                win = extract_window(df, peak_idx + shift)
                if len(win) == WINDOW_SIZE:
                    win.to_csv(out_dir / f"{filename}_Trans_W{w_idx:03d}.csv", index=False)
                    n += 1
            # KHÔNG sinh Idle cho D18/D19 (vùng ngoài peak là đi/đứng -> sẽ thành rác).

        elif prefix in TRANS_HAR:
            peaks, props = find_peaks(svm, height=1.0, distance=200)
            if len(peaks) >= 2:
                selected_peaks = np.sort(peaks[np.argsort(props['peak_heights'])[-2:]])
            else:
                selected_peaks = peaks
            for i, peak in enumerate(selected_peaks):
                for shift_val in [-20, 0, 20]:
                    win = extract_window(df, peak + shift_val)
                    if len(win) == WINDOW_SIZE:
                        sname = "m20" if shift_val == -20 else "p20" if shift_val == 20 else "0"
                        win.to_csv(out_dir / f"{filename}_Trans_W{i:03d}_{sname}.csv", index=False)
                        n += 1
            for start in range(0, len(df) - WINDOW_SIZE + 1, 100):
                center = start + 100
                if all(abs(center - p) >= 80 for p in selected_peaks):
                    win = df.iloc[start:start + WINDOW_SIZE]
                    win.to_csv(out_dir / f"{filename}_{classify_idle(win)}_W{start // 100:03d}.csv", index=False)
                    n += 1

        elif prefix in CONT_HAR:
            for start in range(0, len(df) - WINDOW_SIZE + 1, 100):
                win = df.iloc[start:start + WINDOW_SIZE]
                label = "Walk" if prefix in {'D01', 'D02', 'D05'} else "Run"
                win.to_csv(out_dir / f"{filename}_{label}_W{start // 100:03d}.csv", index=False)
                n += 1
        return n
    except Exception as e:
        print(f"Lỗi khi xử lý {file_path.name}: {e}")
        return 0


def main():
    ap = argparse.ArgumentParser(description="Sinh cache TEST cho pipeline v25 (ResNet-1D, window 200).")
    ap.add_argument("--workspace", type=str,
                    default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help="Thư mục gốc dự án (mặc định: tự phát hiện).")
    args = ap.parse_args()

    ws = Path(args.workspace)
    raw_dir = ws / "SisFall_dataset"
    if not raw_dir.exists():
        print(f"[LỖI] Không tìm thấy dữ liệu thô: {raw_dir}")
        return
    dst_dir = ws / "SisFall_dataset_Windowed_v25_TEST"
    dst_dir.mkdir(parents=True, exist_ok=True)

    print("[*] v25: gyro rad/s (*pi/180) + decimate q=2; window=200 căn giữa.")
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
    for f in tqdm(files, desc="Windowing v25 TEST"):
        total += process_file(f, dst_dir)

    print("\n" + "=" * 60)
    print(f"[+] Đã tạo {total} cửa sổ (200 mẫu) cho tập TEST.")
    print(f"[*] Output: {dst_dir}")
    print("    Lệnh test: python test_inference_uart.py --pipeline v25 --samples 100")
    print("    NHỚ: firmware đặt GYRO_NORM_DIV2000=1 (gyro rad/s /2000).")
    print("=" * 60)


if __name__ == "__main__":
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
    main()
