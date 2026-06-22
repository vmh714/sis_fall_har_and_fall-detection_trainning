# Tài liệu Instruction: Phiên bản TCN-MCU (Temporal Convolutional Network for MCU)

## 1. Tổng quan Model
- **Mục đích:** Phân loại hành vi người dùng từ dữ liệu cảm biến IMU (6-axis).
- **Quy mô:** Hệ thống phân loại 5 lớp (5-class classification).

## 2. Chiến lược Data & Gắn nhãn
Dữ liệu được gán nhãn dựa trên tiền tố của tên file CSV, được chia thành 5 nhóm hành vi cụ thể:
- **Fall:** Các file có tiền tố bắt đầu bằng 'F'.
- **Walk:** Các file có tiền tố 'D01', 'D02', 'D05', 'D06'.
- **Run:** Các file có tiền tố 'D03', 'D04'.
- **Idle:** Bao gồm các trạng thái tĩnh ('_StandSit_', '_Lie_').
- **Trans:** Các file chuyển động trung gian ('_Trans_').
*Lưu ý: Dữ liệu được giới hạn cố định ở độ dài cửa sổ (window size) là 200 mẫu.*

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc dữ liệu:** Sử dụng 6 features đầu vào (dữ liệu thô từ cảm biến).
- **Phân chia tập dữ liệu:** Sử dụng chiến lược tách theo chủ thể (Subject-based split) thành 3 tập: Train, Validation, Test.
- **Cân bằng lớp:** Sử dụng kỹ thuật `class_weight='balanced'` để xử lý vấn đề mất cân bằng dữ liệu trong quá trình huấn luyện.
- **Cache:** Hệ thống tự động lưu/tải dữ liệu qua file `.npy` để tăng tốc độ khởi tạo.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Temporal Convolutional Network (TCN) được tùy biến cho thiết bị nhúng (MCU).
- **Cấu trúc lớp:**
    - Sử dụng 2 stack, mỗi stack bao gồm 4 lớp `Conv1D` với các dilation rate tương ứng [1, 2, 4, 8].
    - `Kernel size = 5` nhằm mở rộng trường tiếp nhận (receptive field).
    - Có sử dụng **Residual Connection** với kỹ thuật `Cropping1D` để khớp chiều dữ liệu sau khi convolution.
    - `BatchNormalization` và `Dropout` (0.2 - 0.4) được áp dụng tại các lớp để chống overfitting.
- **Tối ưu đầu ra:** 
    - Kết hợp cả `GlobalAveragePooling1D` và `GlobalMaxPooling1D` (Concatenate) để trích xuất đặc trưng trung bình và đặc trưng mạnh nhất (chống spikes).
- **Compile:** Sử dụng hàm loss `sparse_categorical_crossentropy` với bộ tối ưu `Adam` (learning rate 1e-3).