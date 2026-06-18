#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script kết hợp:
1. Bắn dữ liệu (Inference via UART) xuống ESP32.
2. Tự động đánh giá hiệu năng (Performance Evaluation) của model TCN v24 (5 lớp) ngay sau khi chạy xong.
3. Tự động kiểm tra và cài đặt thư viện còn thiếu (pyserial, pandas, numpy, matplotlib, seaborn, scikit-learn).
4. Hỗ trợ chế độ `--eval-only` để chỉ đánh giá trên file CSV có sẵn mà không cần UART.
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
        pass

# 1. Tự động kiểm tra và cài đặt thư viện thiếu
REQUIRED_PACKAGES = {
    "serial": "pyserial",
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
import serial
from serial import Serial
import serial.tools.list_ports
import time
import pandas as pd
import struct
import numpy as np
import argparse
import glob
import random
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import re

# Thiết lập hiển thị cho Matplotlib
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# CÁC PIPELINE DATA — mỗi model train theo pipeline khác nhau (windowed folder + ĐƠN VỊ GYRO khác).
# Khi test PHẢI chọn đúng --pipeline khớp FIRMWARE đang flash, nếu không gyro sai đơn vị (rad/s vs dps,
# lệch ~57x) -> model dự đoán sai (nhất là Trans). 'gen' = cách sinh ra windowed folder (cache) đó.
# ============================================================
PIPELINES = {
    "v30": {
        "folder": "SisFall_dataset_Windowed_v30",
        "gyro_unit": "dps (raw*4000/65536, KHONG *pi/180)",
        "firmware_norm": "accel clip(+-8g)/8 ; gyro clip(+-500 dps)/500",
        "trans_cut": "su kien gyro rolling-RMS > 20 dps",
        "models": "v30, v30_lstm32 (v31 4-kenh: firmware tu tinh gyro_mag tu 6 kenh CSV)",
        "gen": "Chay cell windowing notebook train_v30 -> SisFall_dataset_Windowed_v30 (+ cache_v30)",
    },
    "resize": {
        "folder": "SisFall_dataset_Windowed_new",
        "gyro_unit": "rad/s (raw*4000/65536*pi/180)",
        "firmware_norm": "accel clip(+-8g)/8 ; gyro /2000 (v29: /34.9)",
        "trans_cut": "dinh accel SVM (cu)",
        "models": "resize_64_32/32_16/32, v27, v28, v29",
        "gen": "Chay cell windowing notebook resize_* -> SisFall_dataset_Windowed_new (+ cache_resize_64_32)",
    },
    "v3kf": {
        "folder": "SisFall_dataset_Windowed_v3kf",
        "gyro_unit": "dps + decimate (chong aliasing)",
        "firmware_norm": "accel clip(+-8g)/8 ; gyro clip(+-500 dps)/500",
        "trans_cut": "su kien gyro rolling-RMS > 20 dps",
        "models": "KFold v3 (cross-population)",
        "gen": "Chay cell windowing notebook SisFall_KFold_Experiments_v3 -> SisFall_dataset_Windowed_v3kf",
    },
    "v25": {
        "folder": "tool_for_new_dataset/SisFall_dataset_Windowed",
        "gyro_unit": "rad/s (cu)",
        "firmware_norm": "gyro /2000 rad/s",
        "trans_cut": "dinh accel + decimate + them D09/D10/D14",
        "models": "v25 ResNet, v22 TCN",
        "gen": "pipeline v25/v22 (decimate) -> tool_for_new_dataset/SisFall_dataset_Windowed",
    },
}

def print_pipelines():
    print("="*74)
    print("CAC PIPELINE DATA  (chon bang --pipeline <ten>; PHAI khop firmware dang flash)")
    print("="*74)
    for name, p in PIPELINES.items():
        print(f"\n[{name}]  folder = {p['folder']}")
        print(f"   gyro_unit : {p['gyro_unit']}")
        print(f"   firmware  : {p['firmware_norm']}")
        print(f"   Trans cut : {p['trans_cut']}")
        print(f"   models    : {p['models']}")
        print(f"   gen cache : {p['gen']}")
    print("\n" + "-"*74)
    print("[!] CSV gyro phai DUNG don vi firmware mong doi. Gui nham folder -> gyro lech ~57x")
    print("    -> model du doan sai. Folder windowed nao chua co thi chay cell windowing tuong ung.")
    print("="*74)

def get_class_from_filename(filename):
    basename = os.path.basename(filename)
    if basename.startswith('F'):
        return "Fall"
    elif "_Trans_" in basename:
        return "Transition"
    elif "_StandSit_" in basename:
        return "Idle_StandSit"
    elif "_Lie_" in basename:
        return "Idle_Lie"
    elif basename.startswith(('D01', 'D02', 'D05', 'D06')):
        return "Walk"
    elif basename.startswith(('D03', 'D04')):
        return "Run"
    return "Unknown"

def send_sample_to_esp32(csv_path, ser):
    if not os.path.exists(csv_path):
        return None

    expected_class = get_class_from_filename(csv_path)
    
    df = pd.read_csv(csv_path)
    required_columns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    
    for col in required_columns:
        if col not in df.columns:
            return None

    features = df[required_columns].head(200).values.astype(np.float32)
    
    if features.shape[0] < 200:
        pad_size = 200 - features.shape[0]
        padding = np.zeros((pad_size, len(required_columns)), dtype=np.float32)
        features = np.vstack((features, padding))
    
    flat_features = features.flatten()
    byte_data = struct.pack(f'<{len(flat_features)}f', *flat_features)
    
    if ser.in_waiting > 0:
        ser.read(ser.in_waiting)
    
    # --- 1. GIAO TIẾP HANDSHAKE ---
    try:
        # Bỏ ser.flush() để tránh lỗi Write Timeout trên Windows
        ser.write(b'SYNC\n') 
    except serial.SerialException as e:
        print(f"\n[LỖI] Lỗi gửi SYNC: {e}")
        return None
        
    # Chờ mạch báo đã sẵn sàng nhận data
    handshake_ok = False
    start_hs = time.time()
    hs_buffer = ""
    while time.time() - start_hs < 2.0: # Timeout 2s
        if ser.in_waiting > 0:
            try:
                raw_data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                hs_buffer += raw_data
                
                # In ra các log khác (có thể là log Crash/Panic của mạch) để dễ debug
                lines = raw_data.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and "RDY" not in line and "SYNC" not in line:
                        print(f"\n[ESP32 BOOT/CRASH LOG] {line}")
                        
                if "RDY" in hs_buffer:
                    handshake_ok = True
                    break
            except Exception:
                pass
        time.sleep(0.01)
        
    if not handshake_ok:
        print(f"\n[LỖI] Handshake thất bại! Không nhận được RDY từ ESP32.")
        return None

    # --- 2. BẮN DATA (Chia chunk nhỏ để CH340/CP2102 không bị tràn FIFO) ---
    bytes_sent = 0
    try:
        chunk_size = 256
        for i in range(0, len(byte_data), chunk_size):
            chunk = byte_data[i:i+chunk_size]
            bytes_sent += ser.write(chunk)
            time.sleep(0.002) # Delay 2ms cực nhỏ để chip USB xả kịp buffer
    except Exception as e:
        print(f"\n[LỖI] Lỗi ghi data: {e}")
        return None
        
    # --- 3. ĐỢI JSON TRẢ VỀ ---
    start_wait = time.time()
    buffer = ""
    while time.time() - start_wait < 3:
        time.sleep(0.05) 
        
        if ser.in_waiting > 0:
            try:
                raw_data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                buffer += raw_data
            except Exception:
                pass
                
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('{') and line.endswith('}'):
                    try:
                        data = json.loads(line)
                        return {
                            "File": os.path.basename(csv_path),
                            "Expected_Class": expected_class,
                            "Time_us": data.get("time_us", 0),
                            "Is_Stand_Sit": data.get("is_stand_sit"),
                            "Probs": data.get("probs", [])
                        }
                    except json.JSONDecodeError:
                        print(f"\n[ESP32 JSON ERROR] {line}")
                else:
                    if "RDY" not in line: 
                        print(f"\n[ESP32 LOG] {line}")
                        
    print("\n[LỖI] Timeout! ESP32 xử lý data xong nhưng không trả về JSON.")
    return None

def run_evaluation(csv_path, arena_used=None, model_name="model"):
    if not os.path.exists(csv_path):
        print(f"\n[LỖI] Không tìm thấy file '{csv_path}' để đánh giá hiệu năng!")
        return

    print(f"\n[*] Đang đọc dữ liệu từ '{csv_path}' để tiến hành đánh giá hiệu năng...")
    df = pd.read_csv(csv_path)
    
    # 5 lớp chuyên biệt theo ResNet-1D v25
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
    
    time_info = ""
    if 'Time_ms' in df_valid.columns:
        avg_time = df_valid['Time_ms'].mean()
        max_time = df_valid['Time_ms'].max()
        min_time = df_valid['Time_ms'].min()
        time_info += f"- Thời gian Inference trung bình: {avg_time:.2f} ms\n"
        time_info += f"- Thời gian Inference Max/Min: {max_time:.2f} / {min_time:.2f} ms\n"
        
    arena_info = ""
    if arena_used is not None:
        arena_info = f"- Tensor Arena (RAM) sử dụng: {arena_used} bytes\n"

    # Tạo chuỗi báo cáo
    report_str = "\n" + "="*50 + "\n"
    report_str += f"BÁO CÁO PHÂN LOẠI TẬP KIỂM THỬ TRÊN FIRMWARE - {model_name}\n"
    report_str += f"Dựa trên file: {csv_path}\n"
    report_str += f"- Tổng số mẫu test thành công: {total_samples}\n"
    report_str += time_info
    report_str += arena_info
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
    
    # Xác định thư mục lưu (luôn lưu ở thư mục chứa file script này)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Đường dẫn file đầu ra chính
    out_txt_path = os.path.join(base_dir, f"report_{model_name}_firmware.txt")
    out_img_path = os.path.join(base_dir, f"confusion_matrix_{model_name}_firmware.png")
    
    with open(out_txt_path, 'w', encoding='utf-8') as rf:
        rf.write(report_str)
    print(f"[+] Đã lưu báo cáo chi tiết vào file: '{out_txt_path}'")
    
    # Đồng thời lưu vào train_v25_kq nếu thư mục đó tồn tại
    train_kq_dir = os.path.join(base_dir, "train_v25_kq")
    if os.path.exists(train_kq_dir):
        out_txt_path_kq = os.path.join(train_kq_dir, "report_v25_firmware.txt")
        with open(out_txt_path_kq, 'w', encoding='utf-8') as rf:
            rf.write(report_str)
        print(f"[+] Đã lưu bản sao báo cáo chi tiết vào: '{out_txt_path_kq}'")
    
    # Vẽ Confusion Matrix bằng Seaborn (giao diện premium)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={"size": 14, "weight": "bold"})
    
    plt.title(f'Confusion Matrix - {model_name} on Firmware (5 Classes)', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Nhãn Thực Tế (Ground Truth)', fontsize=12, fontweight='bold')
    plt.xlabel('Nhãn Dự Đoán (Firmware Predict)', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    
    plt.savefig(out_img_path, dpi=300)
    print(f"[+] Đã vẽ và lưu ma trận nhầm lẫn thành file: '{out_img_path}'")
    
    # Đồng thời lưu vào train_v25_kq nếu thư mục đó tồn tại
    if os.path.exists(train_kq_dir):
        out_img_path_kq = os.path.join(train_kq_dir, "confusion_matrix_v25_firmware.png")
        plt.savefig(out_img_path_kq, dpi=300)
        print(f"[+] Đã lưu bản sao ma trận nhầm lẫn vào: '{out_img_path_kq}'")
        
    plt.close()
    print("\n>>> Đánh giá hiệu năng hoàn tất thành công! Bạn có thể xem các tệp kết quả ngay bây giờ.")

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(CURRENT_DIR, "inference_results.csv")

    print("\n" + "="*60)
    print("[!] CẢNH BÁO QUAN TRỌNG TRÊN WINDOWS:")
    print("Vui lòng đảm bảo đã TẮT chế độ 'QuickEdit Mode' của Terminal.")
    print("Nếu bạn lỡ click chuột trái vào Terminal, tiến trình Python sẽ bị ĐÓNG BĂNG!")
    print("Để thoát khỏi trạng thái đóng băng, hãy bấm phím ESC hoặc Enter.")
    print("="*60 + "\n")

    parser = argparse.ArgumentParser(description="Tool bắn dữ liệu inference xuống ESP32 và tự động đánh giá hiệu năng v25")
    parser.add_argument("--file", "-f", type=str, help="Đường dẫn đến 1 file CSV")
    parser.add_argument("--pipeline", type=str, default="v30", choices=list(PIPELINES.keys()),
                        help="Chọn PIPELINE data khớp firmware đang flash (mặc định: v30). Xem --list-pipelines")
    parser.add_argument("--list-pipelines", action="store_true", help="In bảng các pipeline data + cache cần gen rồi thoát")
    parser.add_argument("--model-name", type=str, default=None, help="Tên model để đặt tên report (mặc định = tên pipeline)")
    parser.add_argument("--folder", "-d", type=str, default=None, help="Thư mục CSV windowed (mặc định: tự lấy theo --pipeline)")
    parser.add_argument("--samples", "-n", type=int, default=100, help="Số lượng file muốn bốc bừa CHO MỖI NHÃN")
    parser.add_argument("--port", "-p", type=str, help="Cổng COM (ví dụ: COM3). Để trống sẽ TỰ ĐỘNG TÌM.")
    parser.add_argument("--baud", "-b", type=int, default=115200, help="Tốc độ Baudrate")
    parser.add_argument("--eval-only", action="store_true", help="Chỉ chạy đánh giá hiệu năng từ file CSV có sẵn, không thực hiện bắn UART")
    
    args = parser.parse_args()

    # Liệt kê pipeline rồi thoát
    if args.list_pipelines:
        print_pipelines()
        sys.exit(0)

    # Phân giải pipeline -> folder + model_name
    pcfg = PIPELINES[args.pipeline]
    if args.folder is None:
        args.folder = pcfg["folder"]
    if args.model_name is None:
        args.model_name = args.pipeline
    print("="*74)
    print(f"[*] PIPELINE = '{args.pipeline}'  ->  folder CSV = {args.folder}")
    print(f"    gyro      : {pcfg['gyro_unit']}")
    print(f"    firmware  : {pcfg['firmware_norm']}")
    print(f"    [!] DAM BAO firmware dang flash dung pipeline nay (gyro dung don vi), neu khong se sai!")
    print("="*74 + "\n")

    # Chế độ Eval-only
    if args.eval_only:
        print("\n[*] Đang chạy ở chế độ CHỈ ĐÁNH GIÁ (Eval-only)...")
        run_evaluation(out_file, model_name=args.model_name)
        sys.exit(0)
    
    # --- TỰ ĐỘNG TÌM CỔNG COM ---
    if not args.port:
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            print("Lỗi: Không tìm thấy cổng COM nào! Vui lòng cắm cáp kết nối ESP32.")
            sys.exit(1)
        elif len(ports) == 1:
            args.port = ports[0].device
            print(f"[*] Tự động nhận diện thiết bị duy nhất: {args.port} ({ports[0].description})")
        else:
            print("[*] Tìm thấy nhiều thiết bị. Vui lòng chọn cổng ESP32:")
            for i, p in enumerate(ports):
                print(f"  {i}: {p.device} - {p.description}")
            choice = input(f"Chọn (0-{len(ports)-1}) [Nhấn Enter để chọn 0]: ").strip()
            idx = int(choice) if choice.isdigit() and int(choice) < len(ports) else 0
            args.port = ports[idx].device
            print(f"[*] Đã chọn: {args.port}")
            
    if not args.file and not os.path.exists(args.folder):
        print(f"Không tìm thấy thư mục {args.folder}. Vui lòng kiểm tra lại!")
        sys.exit(1)
        
    try:
        # Bật lại write_timeout=3 để ĐẢM BẢO script không bao giờ treo cứng Terminal của bạn!
        ser = Serial(args.port, args.baud, timeout=0.1, write_timeout=3, rtscts=False, dsrdtr=False, xonxoff=False)
        print(f"Connected to {args.port} at {args.baud} bps\n")

        # BẮT BUỘC phải có 2 dòng này để nhả chân EN/BOOT, nếu không ESP32 sẽ tịt ngòi (không boot)
        ser.setDTR(False)
        ser.setRTS(False)

        print("Đang chờ ESP32 khởi động và nạp TFLite Model...")
        
        # Đọc log boot cho đến khi thấy chữ "Waiting for Python tool"
        boot_log_full = ""
        ready = False
        start_boot_wait = time.time()
        while time.time() - start_boot_wait < 5: # Chờ tối đa 5 giây
            if ser.in_waiting > 0:
                chunk = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                boot_log_full += chunk
                if "Waiting for Python tool" in boot_log_full:
                    ready = True
                    break
            time.sleep(0.05)
            
        print("-" * 40)
        print("[ESP32 BOOT LOG]:")
        print(boot_log_full.strip())
        print("-" * 40)
        
        # Lấy thông tin Tensor Arena từ log boot
        arena_used = None
        arena_match = re.search(r"Actual Arena Used:\s*(\d+)\s*bytes", boot_log_full, re.IGNORECASE)
        if arena_match:
            arena_used = arena_match.group(1)
        
        if not ready:
            print("\n[!] QUÁ THỜI GIAN CHỜ ESP32 KHỞI ĐỘNG! Có thể mạch đang bị crash (Panic).")
            if "failed" in boot_log_full.lower() or "error" in boot_log_full.lower() or "panic" in boot_log_full.lower():
                print("\n[!] PHÁT HIỆN LỖI NGAY LÚC KHỞI ĐỘNG FIRMWARE! Vui lòng kiểm tra lại C code.")
            ser.close()
            sys.exit(1)
            
        print("\n[*] ESP32 đã sẵn sàng nhận dữ liệu!")
        # An toàn xóa buffer thủ công
        if ser.in_waiting > 0:
            ser.read(ser.in_waiting)
        time.sleep(0.1) # Nghỉ một nhịp trước khi bắn data
    except Exception as e:
        print(f"Lỗi khi mở cổng {args.port}: {e}")
        print("Gợi ý: Đảm bảo bạn đã tắt ESP-IDF Monitor!")
        sys.exit(1)
        
    all_results = []
    selected_files = []
    
    if args.file:
        selected_files.append(args.file)
        
    if args.folder:
        csv_files = glob.glob(os.path.join(args.folder, "**", "*.csv"), recursive=True)
        if not csv_files:
            print(f"Không tìm thấy file CSV nào trong {args.folder}")
        else:
            class_dict = {}
            unknown_count = 0
            for f in csv_files:
                cls = get_class_from_filename(f)
                if cls == "Unknown":
                    unknown_count += 1
                    continue
                if cls not in class_dict:
                    class_dict[cls] = []
                class_dict[cls].append(f)
            
            if unknown_count > 0:
                print(f"[*] CẢNH BÁO: Bỏ qua {unknown_count} file không thể nhận diện nhãn (Unknown).")
                print("    Nguyên nhân: Tên file không chứa '_Trans_', '_StandSit_' hoặc '_Lie_', và không thuộc nhóm hành động Walk/Run/Fall cơ bản.\n")
                
            for cls, files in class_dict.items():
                num_to_test = min(args.samples, len(files))
                selected_files.extend(random.sample(files, num_to_test))
                print(f"Nhãn [{cls}]: Chọn ngẫu nhiên {num_to_test}/{len(files)} samples.")
                
            random.shuffle(selected_files)
            print(f"\nTổng cộng bốc ra {len(selected_files)} samples để valid...\n")
            
    # Bắt đầu test
    total_files = len(selected_files)
    print("\nBắt đầu test. Nhấn Ctrl + C bất cứ lúc nào để dừng và lưu kết quả hiện tại.")
    
    consecutive_errors = 0 # Cơ chế Fail-safe
    
    try:
        for i, f in enumerate(selected_files):
            print(f"Đang xử lý mẫu {i+1}/{total_files} - {os.path.basename(f)}...", end=" ")
            
            res = send_sample_to_esp32(f, ser)
            if res:
                print("OK")
                all_results.append(res)
                consecutive_errors = 0
            else:
                print("FAIL")
                consecutive_errors += 1
                
            if consecutive_errors >= 3:
                print("\n[FAIL-SAFE TRIGGERED] Đã có 3 lỗi gửi liên tiếp! ESP32 dường như đã bị treo hoặc ngắt kết nối.")
                print("Chủ động dừng script để không làm đơ Terminal...")
                break
                
            time.sleep(0.05) # Dừng 50ms giữa các sample
    except KeyboardInterrupt:
        print("\n\n[!] Bạn đã chủ động dừng chương trình (Ctrl+C).")
        print("[!] Sẽ tiến hành lưu các kết quả đã test được tính đến hiện tại...")
                
    ser.close()
    print("\n\nHoàn tất Test! Đang tổng hợp kết quả...")
    
    if all_results:
        # Làm phẳng list Probs thành các cột riêng rẽ
        flat_results = []
        for r in all_results:
            row = {
                "File": r["File"],
                "Expected_Class": r["Expected_Class"],
                "Time_ms": r["Time_us"] / 1000.0,
                "Is_Stand_Sit_Firmware": r.get("Is_Stand_Sit")
            }
            # --- Logic Threshold & Mapping ---
            CLASS_NAMES = ['Walk', 'Run', 'Idle', 'Trans', 'Fall']
            FALL_IDX = 4
            IDLE_IDX = 2
            
            if r["Probs"]:
                pred_idx = np.argmax(r["Probs"])
                # Fall 25% threshold
                if len(r["Probs"]) > FALL_IDX and r["Probs"][FALL_IDX] >= 0.25:
                    pred_idx = FALL_IDX
                
                # Ánh xạ thành tên nhãn
                pred_class_name = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"Class_{pred_idx}"
                
                # Split Idle thành StandSit và Lie dựa vào hardware heuristic
                if pred_idx == IDLE_IDX:
                    is_stand_sit = r.get("Is_Stand_Sit", True)
                    if is_stand_sit:
                        pred_class_name = "Idle_StandSit"
                    else:
                        pred_class_name = "Idle_Lie"
                
                row["Predicted_Class_Index"] = pred_idx
                row["Predicted_Class_Name"] = pred_class_name
                
            flat_results.append(row)
            
        df_results = pd.DataFrame(flat_results)
        
        # Lưu file tổng hợp
        df_results.to_csv(out_file, index=False)
        
        print("\n" + "="*50)
        print("TÓM TẮT KẾT QUẢ (SUMMARY)")
        print("="*50)
        print(f"- Đã lưu chi tiết vào file: {out_file}")
        print(f"- Tổng số mẫu test thành công: {len(df_results)}")
        print(f"- Thời gian Inference trung bình: {df_results['Time_ms'].mean():.2f} ms")
        print(f"- Thời gian Inference Max/Min: {df_results['Time_ms'].max():.2f} / {df_results['Time_ms'].min():.2f} ms")
        print("="*50)
        
        # In thêm 5 mẫu đầu để xem qua
        print("\nPreview 5 mẫu đầu tiên:")
        print(df_results.head(5).to_string())
        
        # Tự động gọi run_evaluation
        run_evaluation(out_file, arena_used=arena_used, model_name=args.model_name)
    else:
        print("Không có kết quả nào được trả về hợp lệ từ ESP32.")
