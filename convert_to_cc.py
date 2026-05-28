import os
import sys
from pathlib import Path

# Cấu hình UTF-8 cho stdout trên Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Đường dẫn tệp đầu vào và đầu ra
TFLITE_PATH = Path(r'c:\Users\hung.vumanh2\Documents\SisFall-PreProcessing\model_dynamic_range.tflite')
CC_PATH = Path(r'c:\Users\hung.vumanh2\Documents\SisFall-PreProcessing\model_data.cc')
H_PATH = Path(r'c:\Users\hung.vumanh2\Documents\SisFall-PreProcessing\model_data.h')

def convert_tflite_to_cc():
    if not TFLITE_PATH.exists():
        print(f"[!] Lỗi: Không tìm thấy file TFLite tại: {TFLITE_PATH}")
        print("    Vui lòng chắc chắn bạn đã chạy convert_to_tflite.py trước.")
        return

    print(f"[*] Đang đọc file TFLite: {TFLITE_PATH.name}...")
    with open(TFLITE_PATH, 'rb') as f:
        tflite_data = f.read()
    
    data_len = len(tflite_data)
    print(f"[*] Kích thước mô hình: {data_len} bytes (~{data_len / 1024:.2f} KB)")

    # 1. Tạo nội dung cho file Header (.h)
    print(f"[*] Đang tạo file Header: {H_PATH.name}...")
    h_content = f"""#ifndef MODEL_DATA_H_
#define MODEL_DATA_H_

// Khai báo mảng byte chứa mô hình TFLite và độ dài của nó
extern const unsigned char g_model_data[];
extern const unsigned int g_model_data_len;

#endif // MODEL_DATA_H_
"""
    with open(H_PATH, 'w', encoding='utf-8') as f:
        f.write(h_content)
    print(f"    -> Đã lưu thành công: {H_PATH.name}")

    # 2. Tạo nội dung cho file Source C++ (.cc)
    print(f"[*] Đang tạo file Source C++: {CC_PATH.name}...")
    
    # Định dạng mảng byte dạng hex (ví dụ: 0x1c, 0x00, ...)
    hex_bytes = []
    # In tối đa 12 byte trên một dòng để file code đẹp và gọn gàng
    bytes_per_line = 12
    
    for i, byte in enumerate(tflite_data):
        hex_bytes.append(f"0x{byte:02x}")
        
    # Tạo các dòng định dạng thụt lề
    lines = []
    for i in range(0, len(hex_bytes), bytes_per_line):
        chunk = hex_bytes[i:i + bytes_per_line]
        lines.append("  " + ", ".join(chunk))
        
    array_content = ",\n".join(lines)

    cc_content = f"""#include "model_data.h"

// Align 16-byte phù hợp cho tối ưu hóa phần cứng trên vi điều khiển (ARM Cortex-M, ESP32, v.v.)
alignas(16) const unsigned char g_model_data[] = {{
{array_content}
}};

const unsigned int g_model_data_len = {data_len};
"""
    
    with open(CC_PATH, 'w', encoding='utf-8') as f:
        f.write(cc_content)
    print(f"    -> Đã lưu thành công: {CC_PATH.name}")

    print("\n" + "="*50)
    print(" 🎉 XUẤT FILE C++ THÀNH CÔNG CHO TINYML!")
    print("="*50)
    print(f" - File Header: {H_PATH.name}")
    print(f" - File Source: {CC_PATH.name}")
    print(f" - Tên mảng:    g_model_data")
    print(f" - Độ dài mảng: {data_len} bytes")
    print("="*50)
    print("[*] Hướng dẫn sử dụng:")
    print("  1. Copy 2 file 'model_data.h' và 'model_data.cc' vào project C/C++ trên STM32, Arduino, hoặc ESP32.")
    print("  2. Gọi mô hình trong code C++ bằng cách sử dụng: ")
    print("     tflite::GetModel(g_model_data)")
    print("="*50)

if __name__ == "__main__":
    convert_tflite_to_cc()
