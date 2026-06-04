import os
import ast
import glob
import time
from pathlib import Path

try:
    from google import genai
except ImportError:
    print("Vui lòng cài đặt thư viện trước khi chạy: pip install google-genai")
    exit(1)

def extract_from_file(file_path):
    """Trích xuất code từ file giống như script extract_info.py cũ"""
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
        
    extracted_source = []
    
    for node in tree.body:  
        if isinstance(node, ast.FunctionDef):
            name = node.name
            if name in ['parse_filename_info', 'load_single_csv', 'prepare_dataset']:
                src = ast.get_source_segment(source, node)
                if src:
                    extracted_source.append(f"# --- LOGIC GẮN NHÃN & PREPARE DATA ---\n" + src)
            elif name.startswith('build_') or 'block' in name or 'model' in name:
                src = ast.get_source_segment(source, node)
                if src:
                    extracted_source.append(f"# --- KIẾN TRÚC MÔ HÌNH (ARCHITECTURE) ---\n" + src)
            elif name == 'main':
                main_parts = []
                for child in node.body:
                    if isinstance(child, (ast.Assign, ast.For, ast.Expr, ast.AnnAssign)):
                        child_src = ast.get_source_segment(source, child)
                        if child_src:
                            keywords = ['X_train', 'X_val', 'X_test', 'np.clip', 'class_weight', 'to_categorical', 'X[:,']
                            if any(k in child_src for k in keywords) and 'model.fit' not in child_src:
                                main_parts.append(child_src)
                if main_parts:
                    prep_code = "def main_preprocessing_logic():\n    " + "\n    ".join(main_parts).replace('\n', '\n    ')
                    extracted_source.append(f"# --- TIỀN XỬ LÝ (PREPROCESSING) TRƯỚC KHI TRAIN ---\n" + prep_code)
                    
    return "\n\n".join(extracted_source)

def generate_doc_for_file(client, file_path):
    out_path = Path(file_path).parent / "instruction.md"
    
    # Bỏ qua nếu file đã tồn tại để tiếp tục (resume) tiến trình từ chỗ bị lỗi
    if out_path.exists():
        print(f"[*] Đã tồn tại, bỏ qua: {out_path}")
        return False
        
    print(f"[*] Đang xử lý: {file_path}")
    source_code = extract_from_file(file_path)
    
    if not source_code or len(source_code.strip().split('\n')) < 5:
        print(f"    -> Bỏ qua, file không chứa đủ thông tin.")
        return False
        
    prompt = f"""
Bạn là một chuyên gia về AI và Machine Learning.
Dưới đây là mã nguồn trích xuất (đã được lọc bỏ phần thừa) chứa logic tiền xử lý dữ liệu, gán nhãn và kiến trúc mô hình của một phiên bản Model (viết bằng Keras/TensorFlow).

Nhiệm vụ của bạn: Viết một tài liệu `instruction.md` ngắn gọn, súc tích bằng tiếng Việt để báo cáo về phiên bản model này.

YÊU CẦU ĐỊNH DẠNG (Bắt buộc dùng Markdown):
# Tài liệu Instruction: Phiên bản [Lấy tên file hoặc suy luận từ code]

## 1. Tổng quan Model
- Mục đích và quy mô số lớp (classes) đầu ra.

## 2. Chiến lược Data & Gắn nhãn
- Các class được gộp/chia thế nào (ví dụ: Fall, Idle, Walk...). (Liệt kê rõ từ code).

## 3. Tiền xử lý (Preprocessing)
- Thông số cắt gọt (clip) IMU (ví dụ: 8g).
- Thông số chia tỉ lệ (scale) cho Gyro và Accel.

## 4. Kiến trúc mô hình (Architecture)
- Loại mạng (ví dụ TCN, Conv1D, SE Block, Pooling).
- Kỹ thuật tối ưu (Label smoothing, class weights...).

DƯỚI ĐÂY LÀ MÃ NGUỒN:
```python
{source_code}
```
"""
    try:
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite', 
            contents=prompt,
        )
        
        doc_content = response.text
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(doc_content)
            
        print(f"    -> Đã tạo thành công: {out_path}")
        return True
    except Exception as e:
        print(f"    -> LỖI KHI GỌI API: {e}")
        # Nếu bị limit, dừng tại đây để báo hiệu
        if "429" in str(e):
            raise e
        return False

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("LỖI: Chưa tìm thấy GEMINI_API_KEY.")
        return
        
    client = genai.Client(api_key=api_key)
    
    # Tìm file
    search_pattern = 'train_*_kq*/train_*.py'
    files = glob.glob(search_pattern)
    files.extend(glob.glob('train_*_kq/*/*.py'))
    files = sorted(list(set(files)))
    
    print(f"Tìm thấy {len(files)} file cần xử lý. Bắt đầu gọi API...")
    for file_path in files:
        if 'export_' in file_path or 'test_' in file_path:
            continue
            
        try:
            if generate_doc_for_file(client, file_path):
                print("    -> Tạm nghỉ 4.5 giây để tránh lỗi Rate Limit (Limit: 15 RPM)...")
                time.sleep(4.5)
        except Exception as e:
            if "429" in str(e):
                print("Đã gặp lỗi Rate Limit 429. Dừng chương trình. Bạn có thể chạy lại lệnh để tiếp tục từ chỗ lỗi.")
                break
        
    print("\nHOÀN THÀNH!")

if __name__ == '__main__':
    main()
