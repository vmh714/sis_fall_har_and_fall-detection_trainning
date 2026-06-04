import os
import glob
import re

def extract_version(folder_name):
    # Lấy số hiệu version từ tên thư mục (vd: train_v12_kq -> 12, train_2 -> 2)
    match = re.search(r'(?:v|_|^)(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return 999  

def main():
    search_pattern = 'train_*_kq*/instruction.md'
    files = glob.glob(search_pattern)
    files.extend(glob.glob('train_*_kq*/*/instruction.md'))
    
    files = list(set(files))
    if not files:
        print("Không tìm thấy file instruction.md nào.")
        return
        
    files.sort(key=lambda x: extract_version(os.path.dirname(x)))
    output_file = 'All_Models_Instructions_Summary.md'
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write("# TỔNG HỢP TÀI LIỆU CÁC PHIÊN BẢN MODEL\n\n")
        outfile.write("*Tài liệu này được tự động tổng hợp từ các file `instruction.md` trong từng thư mục huấn luyện.*\n\n")
        outfile.write("---\n\n")
        
        for file_path in files:
            folder_name = os.path.basename(os.path.dirname(file_path))
            
            with open(file_path, 'r', encoding='utf-8') as infile:
                content = infile.read()
                # Xóa code block rác
                content = content.replace("```markdown", "").replace("```", "")
                
                # Xóa dòng tiêu đề cũ do AI tự sinh (bắt đầu bằng # Tài liệu Instruction) để thay bằng tiêu đề chuẩn
                content = re.sub(r'^# Tài liệu Instruction:.*?\n', '', content, flags=re.MULTILINE)
                
                # Chèn tiêu đề kèm tên Thư mục / Version rõ ràng vào
                outfile.write(f"# 🏷️ Phiên bản: `{folder_name}`\n\n")
                outfile.write(content.strip() + "\n\n")
                
            outfile.write("---\n\n")
            
    print(f"Da tong hop thanh cong {len(files)} file vao {output_file}")

if __name__ == '__main__':
    main()
