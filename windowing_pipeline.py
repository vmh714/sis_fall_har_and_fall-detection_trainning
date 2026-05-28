import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

def extract_fall_windows(df):
    """Trích xuất các window xung quanh đỉnh gia tốc (Peak Detection) cho Fall data.
    Shift mỗi window 50 mẫu (phù hợp với hardware FIFO 50 mẫu)."""
    # Tính SVM sử dụng cột ADXL_X, ADXL_Y, ADXL_Z từ file csv đã preprocess
    svm = np.sqrt(df['ADXL_X']**2 + df['ADXL_Y']**2 + df['ADXL_Z']**2)
    peak_idx = svm.argmax()
    
    windows = []
    # Lấy các window xung quanh điểm ngã, tịnh tiến mỗi 50 mẫu
    # Các độ lệch (shift) so với center (-100, -50, 0, 50, 100)
    shifts = [-100, -50, 0, 50, 100]
    
    for shift in shifts:
        start = peak_idx - 100 + shift
        end = start + 200
        if start >= 0 and end <= len(df):
            windows.append(df.iloc[start:end])
            
    return windows

def extract_adl_windows(df, window_size=200, step_size=50):
    """Trích xuất sliding window (200 mẫu, overlap 150 mẫu do dịch 50 mẫu/lần) cho ADL data"""
    windows = []
    for start in range(0, len(df) - window_size + 1, step_size):
        windows.append(df.iloc[start:start + window_size])
    return windows

def process_all_windows():
    # Load biến môi trường từ file .env
    load_dotenv()
    
    # Lấy đường dẫn gốc của project từ .env, nếu không có thì dùng thư mục chứa script
    env_root = os.environ.get('PROJECT_ROOT')
    if env_root and os.path.exists(env_root):
        current_dir = Path(env_root)
    else:
        current_dir = Path(__file__).parent
        
    input_dir = current_dir / 'SisFall_dataset_Processed'
    output_dir = current_dir / 'SisFall_dataset_Windowed'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    all_files = list(input_dir.rglob('*.csv'))
    
    total = len(all_files)
    print(f"[*] Tìm thấy {total} files dữ liệu để cắt window.")
    
    windows_generated = 0
    
    for i, file_path in enumerate(all_files):
        try:
            rel_path = file_path.relative_to(input_dir)
            out_file_dir = output_dir / rel_path.parent
            out_file_dir.mkdir(parents=True, exist_ok=True)
            
            df = pd.read_csv(file_path)
            if len(df) < 200:
                continue # Bỏ qua các file quá ngắn không đủ 1 window
                
            filename = file_path.stem
            is_fall = filename.startswith('F')
            
            if is_fall:
                windows = extract_fall_windows(df)
            else:
                windows = extract_adl_windows(df)
                
            for w_idx, win_df in enumerate(windows):
                # Lưu định dạng giống gốc: F01_SA01_R01_W000.csv
                out_name = out_file_dir / f"{filename}_W{w_idx:03d}.csv"
                if not out_name.exists():
                    win_df.to_csv(out_name, index=False)
                windows_generated += 1
                    
            if (i + 1) % 100 == 0:
                print(f"[ Tiến độ ] Đã xử lý {i + 1}/{total} files... (Tạo ra {windows_generated} windows)")
                
        except Exception as e:
            print(f"Lỗi ở file {file_path.name}: {e}")
            
    print(f"[*] Quá trình hoàn tất! Tổng số window được tạo ra: {windows_generated}")

if __name__ == '__main__':
    # Fix unicode lỗi trên Windows console
    if sys.stdout.encoding.lower() != 'utf-8':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
        
    process_all_windows()
