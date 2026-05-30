import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys

warnings.filterwarnings('ignore')

def extract_fall_windows(df):
    """Trích xuất 3 window xung quanh đỉnh gia tốc (Peak Detection) cho Fall data"""
    svm = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    peak_idx = svm.argmax()
    
    windows = []
    # W1: Center
    w1_s, w1_e = peak_idx - 100, peak_idx + 100
    if w1_s >= 0 and w1_e <= len(df):
        windows.append(df.iloc[w1_s:w1_e])
        
    # W2: Left shift
    w2_s, w2_e = peak_idx - 50, peak_idx + 150
    if w2_s >= 0 and w2_e <= len(df):
        windows.append(df.iloc[w2_s:w2_e])
        
    # W3: Right shift
    w3_s, w3_e = peak_idx - 150, peak_idx + 50
    if w3_s >= 0 and w3_e <= len(df):
        windows.append(df.iloc[w3_s:w3_e])
        
    return windows

def extract_adl_windows(df, window_size=200, step_size=100):
    """Trích xuất sliding window (200 mẫu, overlap 100 mẫu) cho ADL data"""
    windows = []
    for start in range(0, len(df) - window_size + 1, step_size):
        windows.append(df.iloc[start:start + window_size])
    return windows

def remove_odd_adl_windows():
    current_dir = Path(__file__).parent
    # Hỗ trợ cả tên thư mục cũ và mới
    windowed_dir = current_dir / 'SisFall_dataset_Windowed'
    if not windowed_dir.exists():
        windowed_dir = current_dir / 'SisFall_dataset_Window'
        
    if not windowed_dir.exists():
        print(f"Không tìm thấy thư mục {windowed_dir}")
        return
        
    all_files = list(windowed_dir.rglob('*.csv'))
    total_files = len(all_files)
    removed = 0
    
    print(f"[*] Đang quét {total_files} file trong {windowed_dir.name} để xóa các window lẻ...")
    
    for file_path in all_files:
        filename = file_path.stem
        is_fall = filename.startswith('F')
        
        # Chỉ xử lý các file không phải là ngã (ADL)
        if not is_fall:
            try:
                # Cắt chuỗi để lấy số W (ví dụ từ D01_SA01_R01_W001 lấy ra 1)
                w_idx_str = filename.split('_W')[-1]
                w_idx = int(w_idx_str)
                
                # Nếu là số lẻ (1, 3, 5...) thì xóa file
                if w_idx % 2 != 0:
                    file_path.unlink()
                    removed += 1
            except Exception as e:
                pass
                
    print(f"[*] Hoàn tất! Đã xóa thành công {removed} file ADL có số window lẻ.")

def process_all_windows():
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
                windows = extract_adl_windows(df, step_size=50)
                
            for w_idx, win_df in enumerate(windows):
                # Lọc bỏ các window số chẵn của dữ liệu không phải Fall
                if not is_fall and w_idx % 2 == 0:
                    continue
                    
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
        
    # Thay vì process_all_windows() (tạo mới), bây giờ ta gọi hàm xóa file lẻ trực tiếp trong folder đã có
    remove_odd_adl_windows()
