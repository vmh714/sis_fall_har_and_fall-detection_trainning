import pypandoc
import os
import re
import subprocess

# Đường dẫn tương đối dựa trên thư mục chạy script
current_dir = os.path.dirname(os.path.abspath(__file__))
md_file = os.path.join(current_dir, "Bao_Cao_Tong_The_v30_v31.md")
tex_file = os.path.join(current_dir, "Bao_Cao_Tong_The_v30_v31.tex")
pdf_file = os.path.join(current_dir, "Bao_Cao_Tong_The_v30_v31.pdf")

# 1. Định nghĩa Preamble chuyên nghiệp
PREAMBLE = r"""\documentclass[12pt,a4paper]{article}

% --- CÀI ĐẶT FONT CHỮ VÀ NGÔN NGỮ ---
\usepackage{fontspec}
\setmainfont{Times New Roman}
\usepackage{amsmath}

% --- CĂN LỀ TRANG ---
\usepackage[top=2cm, bottom=2cm, left=2.5cm, right=2cm]{geometry}

% --- LÀM ĐẸP ĐOẠN VĂN (PARAGRAPH) ---
\usepackage{parskip} % Tự động tạo khoảng cách giữa các đoạn, bỏ lùi đầu dòng
\usepackage{microtype} % Căn chữ (justify) đều và đẹp hơn, tránh bị giãn khoảng trắng

% --- LÀM ĐẸP TIÊU ĐỀ (HEADINGS) ---
\usepackage{titlesec}
\usepackage{xcolor}
\titleformat{\section}{\Large\bfseries\color{blue!50!black}}{\thesection.}{1em}{}
\titleformat{\subsection}{\large\bfseries\color{blue!60!black}}{\thesubsection.}{1em}{}
\titleformat{\subsubsection}{\normalsize\bfseries\color{blue!70!black}}{\thesubsubsection.}{1em}{}

% --- LÀM ĐẸP BẢNG & HÌNH ẢNH ---
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{graphicx}
\usepackage{float}
\usepackage{caption}
\captionsetup{labelfont=bf, textfont=it, margin=1cm}

% --- HYPERLINK ---
\usepackage[colorlinks=true, linkcolor=blue, urlcolor=blue, citecolor=blue]{hyperref}

% Định nghĩa lại pandocbounded nếu pandoc 3.x sử dụng nó
\providecommand{\pandocbounded}[1]{#1}

% Định nghĩa \tightlist để tránh lỗi biên dịch list của Pandoc
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}

% Định nghĩa counter giả 'none' để sửa lỗi tương thích longtable/caption của Pandoc
\newcounter{none}

\begin{document}
"""

def optimize_tables_balanced(content):
    r"""
    Sử dụng thuật toán cân bằng ngoặc nhọn để tìm định nghĩa cột của \begin{longtable}[]
    và thay thế bằng định nghĩa cột tối ưu hóa tùy theo số lượng cột.
    Đồng thời áp dụng \begingroup \small \setlength{\tabcolsep}{4pt} để chống tràn bảng.
    """
    pos = 0
    while True:
        # Tìm \begin{longtable}[]
        idx = content.find(r"\begin{longtable}[]", pos)
        if idx == -1:
            break
        
        # Tìm dấu mở ngoặc nhọn '{' của định nghĩa cột ngay sau đó
        start_bracket = content.find("{", idx + len(r"\begin{longtable}[]"))
        if start_bracket == -1:
            pos = idx + 1
            continue
            
        # Quét cân bằng ngoặc nhọn
        balance = 1
        i = start_bracket + 1
        while balance > 0 and i < len(content):
            if content[i] == '{':
                balance += 1
            elif content[i] == '}':
                balance -= 1
            i += 1
            
        if balance == 0:
            # Lấy được định nghĩa cột bao gồm cả cặp ngoặc nhọn ngoài cùng
            col_def = content[start_bracket:i]
            
            # Đếm số lượng cột trong col_def
            num_cols = 0
            if "p{" in col_def:
                num_cols = col_def.count("p{")
            elif "l" in col_def or "c" in col_def or "r" in col_def:
                # Đếm số chữ l, c, r đơn giản
                num_cols = col_def.count("l") + col_def.count("c") + col_def.count("r")
            
            # Xác định định dạng cột mới và giãn cách
            new_col_def = None
            if num_cols == 4:
                new_col_def = "{@{} l c c c @{}}"
            elif num_cols == 7:
                new_col_def = "{@{} l c c c c c c @{}}"
            elif num_cols == 5:
                # Ép cột 2 (Cấu trúc & Kỹ thuật) về p{5cm} thay vì p{6.5cm} để tránh tràn lề
                new_col_def = "{@{} l p{5cm} c c c @{}}"
                
            if new_col_def:
                # Thay thế định nghĩa cột cũ bằng định nghĩa mới, thêm giãn dòng và bóp nhỏ font/khoảng cách cột
                new_segment = (
                    f"\\begingroup\n"
                    f"\\small\n"
                    f"\\setlength{{\\tabcolsep}}{{4pt}}\n"
                    f"\\renewcommand{{\\arraystretch}}{{1.3}}\n"
                    f"\\begin{{longtable}}{new_col_def}"
                )
                content = content[:idx] + new_segment + content[i:]
                
                # Tìm và chèn \endgroup sau \end{longtable} tương ứng của bảng này
                end_idx = content.find(r"\end{longtable}", idx + len(new_segment))
                if end_idx != -1:
                    end_segment = "\\end{longtable}\n\\endgroup"
                    content = content[:end_idx] + end_segment + content[end_idx + len(r"\end{longtable}"):]
                    pos = end_idx + len(end_segment)
                else:
                    pos = idx + len(new_segment)
            else:
                pos = i
        else:
            pos = start_bracket + 1
            
    return content

def main():
    # Bước 1: Dịch Markdown sang LaTeX standalone
    print("Buoc 1: Dang sinh tep LaTeX standalone tu file Markdown nguon...")
    try:
        pypandoc.convert_file(
            md_file,
            'latex',
            outputfile=tex_file,
            extra_args=['-s']
        )
        print("-> Sinh file LaTeX thanh cong!")
    except Exception as e:
        print(f"Loi khi goi Pandoc: {e}")
        return

    # Bước 2: Đọc file LaTeX và tiến hành tối ưu hóa cấu trúc
    print("\nBuoc 2: Dang toi uu hoa dinh dang trong file LaTeX...")
    if not os.path.exists(tex_file):
        print(f"Loi: Khong tim thay file {tex_file} vua sinh.")
        return

    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 2.1. Thay thế Preamble mặc định của Pandoc bằng Preamble chuyên nghiệp
    if r"\begin{document}" in content:
        parts = content.split(r"\begin{document}", 1)
        content = PREAMBLE + parts[1]
        print("   [+] Da nang cap Preamble thanh cong.")
    else:
        print("   [!] Khong tim thay the \\begin{document} de thay the Preamble.")

    # 2.2. Tối ưu cấu trúc Bảng (Tables) bằng thuật toán Balanced Brace Parser và Grouping
    content = optimize_tables_balanced(content)
    print("   [+] Da toi uu hoa dinh nghia va gian cach dong cho tat ca cac bang.")

    # 2.3. Căn giữa và chuẩn hoá kích thước Hình ảnh (Images)
    pattern_img = r'\\pandocbounded\{\\includegraphics\[([^\]]+)\]\{([^\}]+)\}\}'
    
    def replace_img(match):
        options = match.group(1)
        path = match.group(2)
        # Thêm width=0.85\textwidth vào options để vừa vặn khung trang
        new_options = f"width=0.85\\textwidth, {options}"
        return f"\\begin{{figure}}[H]\n    \\centering\n    \\pandocbounded{{\\includegraphics[{new_options}]{{{path}}}}}\n\\end{{figure}}"
        
    content = re.sub(pattern_img, replace_img, content)
    print("   [+] Da cau hinh can giua va co dan anh cho tat ca hinh ve.")

    # 2.4. Xóa mã lỗi minipage trong tiêu đề bảng của Pandoc
    content = re.sub(
        r'\\begin\{minipage\}\[b\]\{\\linewidth\}\\raggedright\s*(.*?)\s*\\end\{minipage\}',
        r'\1',
        content,
        flags=re.DOTALL
    )
    print("   [+] Da don dep ma loi minipage trong tieu de bang.")

    # Ghi đè lại tệp .tex đã được làm đẹp
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print("-> Da cap nhat va luu file LaTeX hoan chinh.")

    # Bước 3: Biên dịch file .tex sang PDF bằng xelatex của hệ thống (chạy 2 lần để cập nhật cross-reference)
    print("\nBuoc 3: Dang bien dich file LaTeX sang PDF bang xelatex...")
    try:
        # Chạy xelatex lần 1
        print("   [+] Dang chay xelatex lan 1...")
        subprocess.run(
            ['xelatex', '-interaction=nonstopmode', 'Bao_Cao_Tong_The_v30_v31.tex'],
            cwd=current_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            check=True
        )
        
        # Chạy xelatex lần 2
        print("   [+] Dang chay xelatex lan 2...")
        result = subprocess.run(
            ['xelatex', '-interaction=nonstopmode', 'Bao_Cao_Tong_The_v30_v31.tex'],
            cwd=current_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            check=True
        )
        
        if result.returncode == 0:
            print("-> Bien dich PDF THANH CONG!")
            # Dọn dẹp các tệp phụ sinh ra từ quá trình dịch LaTeX (.aux, .log, .out)
            for ext in ['.aux', '.log', '.out']:
                aux_file = os.path.join(current_dir, f"Bao_Cao_Tong_The_v30_v31{ext}")
                if os.path.exists(aux_file):
                    os.remove(aux_file)
            print("   [+] Da don dep cac tep phu (.aux, .log, .out).")
        else:
            print("Loi khi chay xelatex:")
            print(result.stdout[:2000])
            print(result.stderr[:2000])
    except Exception as e:
        print(f"Loi he thong khi goi xelatex: {e}")

if __name__ == "__main__":
    main()
