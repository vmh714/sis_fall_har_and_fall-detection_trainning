# Tài liệu Instruction: Phiên bản Human Activity Recognition (TCN-based)

## 1. Tổng quan Model
- **Mục đích:** Phân loại hành động của con người dựa trên dữ liệu IMU (cảm biến gia tốc và con quay hồi chuyển).
- **Quy mô:** Mô hình phân loại thành **5 lớp** hành động đầu ra (Softmax).

## 2. Chiến lược Data & Gắn nhãn
Dữ liệu được gán nhãn dựa trên thông tin tên file (filename parsing):
- **Fall**: Các file có tiền tố bắt đầu bằng 'F'.
- **Walk**: Tiền tố 'D01', 'D02', 'D05', 'D06'.
- **Run**: Tiền tố 'D03', 'D04'.
- **Idle**: Các trạng thái tĩnh ('StandSit', 'Lie').
- **Trans**: Các trạng thái chuyển tiếp ('Trans').

## 3. Tiền xử lý (Preprocessing)
- **Cửa sổ dữ liệu (Windowing):** Dữ liệu được chuẩn hóa thành các clip có độ dài cố định là **200 mẫu (samples)** cho mỗi file.
- **Tính năng đầu vào:** 6 features (3 trục gia tốc `ax, ay, az` và 3 trục con quay hồi chuyển `gx, gy, gz`).
- **Lưu trữ:** Sử dụng cơ chế Cache (file `.npy`) để tăng tốc độ tải dữ liệu cho các lần chạy sau.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Sử dụng kiến trúc **TCN (Temporal Convolutional Network)** với các khối Residual nối tiếp.
- **Cấu trúc chi tiết:**
    - Gồm 2 stack, mỗi stack chứa các lớp `Conv1D` với `dilation_rate` lần lượt là 1, 2, 4, 8 để mở rộng trường nhìn (receptive field).
    - Sử dụng `BatchNormalization` sau các lớp tích chập và `Dropout` (0.2 - 0.3) để chống overfitting.
    - Kết nối tắt (Residual connection) với `Cropping1D` để khớp kích thước tensor sau các lớp giãn nở.
    - Lớp cuối: `GlobalAveragePooling1D` kết hợp với `Dense` layer (Softmax).
- **Kỹ thuật tối ưu:**
    - **Class Weights:** Sử dụng trọng số cân bằng (`class_weight='balanced'`) để xử lý vấn đề mất cân bằng dữ liệu giữa các class.
    - **Optimizer:** Adam (learning rate = 1e-3).
    - **Loss Function:** `sparse_categorical_crossentropy`.