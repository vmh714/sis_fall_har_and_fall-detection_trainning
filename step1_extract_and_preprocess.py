import os
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def preprocess_sisfall(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Danh sách các hành động HAR cần lấy theo yêu cầu của user
    target_har = {
        'D01', 'D02', 'D03', 'D04', 'D05', 
        'D07', 'D08', 'D11', 'D12', 'D13', 
        'D15', 'D17', 'D18', 'D19'
    }
    
    # Tìm tất cả các file .txt trong dataset
    all_files = list(input_path.rglob('*.txt'))
    print(f"Tổng số file tìm thấy trong dataset gốc: {len(all_files)}")
    
    files_to_process = []
    for f in all_files:
        filename = f.name
        prefix = filename[:3]
        if filename.startswith('F') or prefix in target_har:
            files_to_process.append(f)
            
    print(f"Số lượng file cần xử lý (Fall + HAR đã chọn): {len(files_to_process)}")
    
    # Các hệ số chuyển đổi
    # Gia tốc (ADXL345): (2 * 16) / (2^13) = 32 / 8192 = 1/256 = 0.00390625 g
    ACCEL_SCALE = 32.0 / 8192.0
    
    # Góc quay (ITG3200): (2 * 2000) / (2^16) = 4000 / 65536 = 0.06103515625 deg/s
    # Chuyển sang rad/s: nhân thêm pi/180
    GYRO_SCALE = (4000.0 / 65536.0) * (np.pi / 180.0)
    
    # Tạo thư mục output
    output_path.mkdir(parents=True, exist_ok=True)
    
    for file_path in tqdm(files_to_process, desc="Processing files"):
        try:
            # Đọc dữ liệu thô (bỏ dấu ';' ở cuối mỗi dòng)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            data = []
            for line in lines:
                line = line.strip().rstrip(';')
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 6:
                    # Lấy 6 trục đầu tiên (ax, ay, az, gx, gy, gz)
                    data.append([float(x) for x in parts[:6]])
                    
            if not data:
                continue
                
            df = pd.DataFrame(data, columns=['ax', 'ay', 'az', 'gx', 'gy', 'gz'])
            
            # Áp dụng hệ số chuyển đổi
            df[['ax', 'ay', 'az']] = df[['ax', 'ay', 'az']] * ACCEL_SCALE
            df[['gx', 'gy', 'gz']] = df[['gx', 'gy', 'gz']] * GYRO_SCALE
            
            # Downsample từ 200Hz xuống 100Hz (Lấy mẫu cách đoạn 2)
            df = df.iloc[::2, :].reset_index(drop=True)
            
            # Xây dựng đường dẫn output (giữ nguyên cấu trúc thư mục subject)
            rel_path = file_path.relative_to(input_path)
            out_file = output_path / rel_path.parent / (file_path.stem + '.csv')
            out_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Lưu ra CSV
            df.to_csv(out_file, index=False)
            
        except Exception as e:
            print(f"Lỗi khi xử lý {file_path.name}: {e}")

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    INPUT_DIR = 'SisFall_dataset'
    OUTPUT_DIR = 'SisFall_dataset_Processed'
    preprocess_sisfall(INPUT_DIR, OUTPUT_DIR)
