# Tài liệu Instruction: Phiên bản TCN-MCU Human Activity Recognition

## 1. Tổng quan Model
- **Mục đích**: Nhận diện hoạt động (Human Activity Recognition) dựa trên dữ liệu cảm biến IMU (Accel/Gyro).
- **Quy mô**: Phân loại theo 4 lớp đầu ra (`n_classes=4`).

## 2. Chiến lược Data & Gắn nhãn
- **Gắn nhãn**: Dựa trên tiền tố của tên tệp tin (`label_code`) thông qua dictionary `LABEL_MAP`.
- **Phân tách**: Dữ liệu được chia tập huấn luyện (train), kiểm thử (val) và đánh giá (test) dựa trên `subject_id` (ID người tham gia), đảm bảo không bị rò rỉ dữ liệu giữa các tập.
- **Cấu trúc**: Mỗi mẫu dữ liệu được yêu cầu cố định ở độ dài 200 time-steps.

## 3. Tiền xử lý (Preprocessing)
- **Cửa sổ thời gian**: Cố định `window size = 200`.
- **Cân bằng dữ liệu**: Sử dụng kỹ thuật `class_weight='balanced'` để tính toán trọng số cho từng lớp, giúp xử lý vấn đề mất cân bằng dữ liệu trong quá trình huấn luyện.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng**: Temporal Convolutional Network (TCN) tùy chỉnh cho MCU.
- **Thành phần chính**:
    - **Stacking**: 2 khối (stacks), mỗi khối gồm 4 lớp `Conv1D` với các `dilation_rate` lần lượt là [1, 2, 4, 8].
    - **Cơ chế**: Sử dụng `padding='valid'` kết hợp `Cropping1D` để căn chỉnh kích thước `residual` thay vì dùng `causal padding`, tối ưu hóa cho việc chuyển đổi sang định dạng TFLite (tránh toán tử `Pad`/`SpaceToBatchND`).
    - **Regularization**: Sử dụng `BatchNormalization` sau mỗi lớp `Conv1D` và `Dropout` (0.2 trong block, 0.3 ở lớp Global Pooling) để giảm overfitting.
    - **Pooling**: `GlobalAveragePooling1D` để nén đặc trưng trước lớp phân loại.
- **Tối ưu**: Sử dụng `Adam` optimizer với learning rate `1e-3` và hàm mất mát `sparse_categorical_crossentropy`.