# Tài liệu Instruction: Phiên bản TCN-MCU Human Activity Recognition

## 1. Tổng quan Model
- **Mục đích:** Nhận diện hoạt động con người (HAR) dựa trên dữ liệu cảm biến IMU (cảm biến gia tốc và con quay hồi chuyển).
- **Quy mô:** Phân loại 5 lớp hoạt động (Classes).

## 2. Chiến lược Data & Gắn nhãn
Dữ liệu được gắn nhãn tự động dựa trên tên tệp tin:
- **Fall:** Các tệp bắt đầu bằng ký tự 'F'.
- **Walk:** Các tệp bắt đầu bằng mã 'D01', 'D02', 'D05', 'D06'.
- **Run:** Các tệp bắt đầu bằng mã 'D03', 'D04'.
- **Idle:** Các tệp có chứa '_StandSit_' hoặc '_Lie_'.
- **Trans:** Các tệp có chứa '_Trans_'.
- **Định dạng dữ liệu:** Cửa sổ trượt (window) có độ dài cố định 200 mẫu (samples) cho mỗi tệp.

## 3. Tiền xử lý (Preprocessing)
- **Cắt gọt (Clipping):** Dữ liệu cảm biến gia tốc (3 trục đầu) được giới hạn trong khoảng [-8.0, 8.0]g để loại bỏ nhiễu vượt ngưỡng vật lý.
- **Scaling:**
    - **Accelerometer (0:3):** Chia cho 8.0.
    - **Gyroscope (3:6):** Chia cho 2000.0.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Residual Conv1D (phong cách TCN) kết hợp Squeeze-and-Excitation (SE) Block.
- **Chi tiết cấu trúc:**
    - Sử dụng 4 khối Residual với Conv1D (kernel size 7), xen kẽ với các lớp BatchNormalization và Dropout (0.2).
    - Sử dụng **SE Block** (ratio=4) để tối ưu hóa trọng số kênh (channel attention).
    - Giảm chiều dữ liệu bằng `strides=2` thay vì Dilation.
    - Đầu ra kết hợp `GlobalAveragePooling1D` và `GlobalMaxPooling1D` (Concatenate) để tối ưu đặc trưng.
- **Kỹ thuật tối ưu:**
    - **Label Smoothing:** 0.1 nhằm tăng khả năng tổng quát hóa.
    - **Class Weights:** Sử dụng trọng số cân bằng (`class_weight='balanced'`) để xử lý bài toán dữ liệu mất cân bằng.
    - **Activation:** `relu6` (phù hợp cho các thiết bị nhúng).
    - **Regularization:** L2 regularization (1e-4) trên các lớp Conv/Dense.