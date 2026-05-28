import pandas as pd
import numpy as np
from scipy import signal
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore') # Ẩn các cảnh báo không cần thiết

def preprocess_sisfall_file(file_path, original_fs=200, target_fs=100):
    # 1. Load dữ liệu
    df = pd.read_csv(file_path, header=None, sep=',').iloc[:, :6]
    df.columns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    
    # Loại bỏ ký tự lạ ở cuối dòng
    df = df.replace(';', '', regex=True).astype(float)

    # 2. Chuyển đổi sang đơn vị chuẩn (Readme.txt)
    acc_scale = (2 * 16) / (2**13)
    df[['ax', 'ay', 'az']] = df[['ax', 'ay', 'az']] * acc_scale

    gyro_scale = (2 * 2000) / (2**16)
    df[['gx', 'gy', 'gz']] = df[['gx', 'gy', 'gz']] * gyro_scale

    # 3. Tính số lượng mẫu mới
    num_samples = len(df)
    new_num_samples = int(num_samples * (target_fs / original_fs))
    
    # 4. Downsampling (200Hz -> 100Hz)
    resampled_data = signal.resample(df, new_num_samples)
    
    df_resampled = pd.DataFrame(resampled_data, columns=['ax', 'ay', 'az', 'gx', 'gy', 'gz'])
    
    return df_resampled

def process_entire_dataset(input_dir='SisFall_dataset', output_dir='SisFall_dataset_Processed'):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Tìm toàn bộ các file .txt (bỏ qua Readme.txt)
    all_files = list(input_path.rglob('*.txt'))
    all_files = [f for f in all_files if f.name != 'Readme.txt']
    
    total_files = len(all_files)
    print(f"[*] Đã tìm thấy {total_files} file để xử lý.")
    
    for i, file_path in enumerate(all_files):
        try:
            # Giữ nguyên cấu trúc thư mục con (ví dụ: SA01, SE01...)
            relative_path = file_path.relative_to(input_path)
            out_file_dir = output_path / relative_path.parent
            out_file_dir.mkdir(parents=True, exist_ok=True)
            
            # Chuyển đổi tên file từ .txt sang .csv
            out_file_name = out_file_dir / (file_path.stem + '.csv')
            
            # Nếu file đã tồn tại thì bỏ qua (dùng để có thể dừng/chạy tiếp nếu bị gián đoạn)
            if out_file_name.exists():
                continue
                
            # Chạy pipeline xử lý
            df_processed = preprocess_sisfall_file(str(file_path))
            
            # Lưu ra file csv
            df_processed.to_csv(out_file_name, index=False)
            
            # In tiến độ định kỳ
            if (i + 1) % 100 == 0:
                print(f"[ Tiến độ ] Đã xử lý {i + 1}/{total_files} file...")
                
        except Exception as e:
            print(f"[ Lỗi ] Không thể xử lý file {file_path.name}: {e}")
            
    print("[*] Quá trình xử lý dữ liệu hoàn tất!")

if __name__ == '__main__':
    # Lấy thư mục hiện tại của file script
    current_dir = Path(__file__).parent
    input_directory = current_dir / 'SisFall_dataset'
    output_directory = current_dir / 'SisFall_dataset_Processed'
    
    process_entire_dataset(input_dir=input_directory, output_dir=output_directory)