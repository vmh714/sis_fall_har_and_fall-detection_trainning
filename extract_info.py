import os
import ast
import glob
from pathlib import Path

def extract_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
        
    extracted_source = []
    
    # Header comment
    extracted_source.append(f"# {'='*50}\n# TRÍCH XUẤT TỪ FILE: {Path(file_path).name}\n# {'='*50}")
    
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            name = node.name
            # Trích xuất Labeling & Data Prep
            if name in ['parse_filename_info', 'load_single_csv', 'prepare_dataset']:
                src = ast.get_source_segment(source, node)
                if src:
                    extracted_source.append(f"# --- LOGIC GẮN NHÃN & PREPARE DATA ---\n" + src)
                    
            # Trích xuất Architecture
            elif name.startswith('build_') or 'block' in name or 'model' in name:
                src = ast.get_source_segment(source, node)
                if src:
                    extracted_source.append(f"# --- KIẾN TRÚC MÔ HÌNH (ARCHITECTURE) ---\n" + src)
                    
            # Trích xuất Preprocessing trong main()
            elif name == 'main':
                main_parts = []
                for child in node.body:
                    if isinstance(child, (ast.Assign, ast.For, ast.Expr, ast.AnnAssign)):
                        child_src = ast.get_source_segment(source, child)
                        if child_src:
                            # Bộ lọc từ khóa để lấy đúng các thao tác trên dữ liệu Numpy
                            keywords = ['X_train', 'X_val', 'X_test', 'np.clip', 'class_weight', 'to_categorical', 'X[:,']
                            if any(k in child_src for k in keywords) and 'model.fit' not in child_src:
                                main_parts.append(child_src)
                                
                if main_parts:
                    prep_code = "def main_preprocessing_logic():\n    " + "\n    ".join(main_parts).replace('\n', '\n    ')
                    extracted_source.append(f"# --- TIỀN XỬ LÝ (PREPROCESSING) TRƯỚC KHI TRAIN ---\n" + prep_code)
                    
    return "\n\n".join(extracted_source)

def process_all():
    out_dir = Path('exported_ml_info')
    out_dir.mkdir(exist_ok=True)
    
    # Tìm tất cả file train_*.py trong các thư mục train_v*
    search_pattern = 'train_*_kq*/train_*.py'
    files = glob.glob(search_pattern)
    files.extend(glob.glob('train_*_kq/*/*.py')) # in subfolders if any
    
    # Loại bỏ duplicate và sắp xếp
    files = sorted(list(set(files)))
    
    for file_path in files:
        # Bỏ qua nếu không phải file train chính
        if 'export_' in file_path or 'test_' in file_path:
            continue
            
        print(f"Processing {file_path}...")
        content = extract_from_file(file_path)
        
        if content and len(content.strip().split('\n')) > 5:
            # Lưu ra file riêng
            parent_dir = Path(file_path).parent.name
            out_name = f"{parent_dir}_{Path(file_path).name}"
            out_path = out_dir / out_name
            
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(content)

if __name__ == '__main__':
    process_all()
