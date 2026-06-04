Dưới đây là tài liệu hướng dẫn kỹ thuật cho phiên bản model dựa trên mã nguồn bạn đã cung cấp.

# Tài liệu Instruction: Phiên bản TCN-MCU (Time Convolutional Network)

## 1. Tổng quan Model
- **Mục đích:** Phân loại hành động dựa trên dữ liệu chuỗi thời gian từ cảm biến IMU (thường là Accel/Gyro).
- **Quy mô:** Đầu ra gồm 4 lớp (n_classes = 4).

## 2. Chiến lược Data & Gắn nhãn
- **Cơ chế:** Dữ liệu được gán nhãn tự động thông qua tiền tố (prefix) của tên tệp tin (ví dụ: `[label_code]_[subject_id].csv`).
- **Phân chia tập dữ liệu:** Dữ liệu được phân chia theo `subject_id` (người thực hiện) vào các tập Train, Validation và Test để đảm bảo tính độc lập của dữ liệu.
- **Xử lý mất cân bằng:** Sử dụng kỹ thuật `class_weight='balanced'` trong quá trình huấn luyện để tính toán trọng số lớp, đảm bảo mô hình không bị thiên lệch bởi các class có số lượng mẫu ít.

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc dữ liệu:** Mỗi mẫu đầu vào là một cửa sổ thời gian (window) cố định với độ dài chính xác là 200 đơn vị (timestep).
- **Định dạng:** Dữ liệu được lưu trữ và tải thông qua tệp `.npy` để tăng tốc độ huấn luyện (cache system).
- **Lưu ý:** Mã nguồn hiện tại tập trung vào việc chuẩn hóa cấu trúc dữ liệu theo `subject_id`, đảm bảo tính nhất quán của dữ liệu đầu vào trước khi đưa vào mạng.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Temporal Convolutional Network (TCN) với 2 lớp stack (mỗi lớp gồm các khối giãn nở - dilation rates 1, 2, 4, 8).
- **Thành phần chính:**
    - `BatchNormalization`: Ổn định quá trình học.
    - `Conv1D`: Trích xuất đặc trưng không gian-thời gian.
    - `Residual Connection (Add)`: Kết nối tắt để tránh mất mát thông tin (kèm theo `Cropping1D` để đồng bộ chiều dữ liệu sau tích chập).
    - `GlobalAveragePooling1D`: Giảm chiều dữ liệu trước lớp phân loại.
    - `Dropout` (0.2 - 0.3): Chống quá khớp (overfitting).
- **Tối ưu hóa:**
    - **Loss:** `sparse_categorical_crossentropy`.
    - **Optimizer:** Adam (Learning rate = 1e-3).
    - **Cấu hình:** Đã loại bỏ *Label Smoothing* và quay về cấu trúc mặc định để tối ưu hóa hiệu suất trên MCU.