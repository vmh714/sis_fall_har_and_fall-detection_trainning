# Tài liệu Instruction: Phiên bản TCN-MCU Human Activity Recognition

## 1. Tổng quan Model
- **Mục đích**: Nhận diện hoạt động (Human Activity Recognition) dựa trên dữ liệu cảm biến IMU (Accel & Gyro).
- **Quy mô đầu ra**: Mô hình phân loại 4 lớp (n_classes=4).

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc dữ liệu**: Mỗi sample là một cửa sổ (window) có độ dài cố định 200 đơn vị thời gian.
- **Phân chia tập dữ liệu**: Dữ liệu được chia theo `subject_id` (Train/Val/Test) để đảm bảo tính độc lập giữa các đối tượng người dùng.
- **Gắn nhãn**: Sử dụng `LABEL_MAP` để ánh xạ mã nhãn từ tên file (dạng `labelCode_subjectId.csv`).

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc trục**: Dữ liệu đầu vào ban đầu gồm Accel và Gyro, sau đó được bổ sung thêm kênh SVM (Signal Vector Magnitude) để tăng cường đặc trưng (tổng cộng 7 trục dữ liệu).
- **Định dạng**: Dữ liệu được kiểm tra độ dài nghiêm ngặt (chỉ chấp nhận 200 timestep).
- **Tối ưu hóa**: Dữ liệu được lưu trữ dạng Cache (`.npy`) để tăng tốc độ tải trong các lần huấn luyện tiếp theo.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng**: Kiến trúc **TCN (Temporal Convolutional Network)** tối ưu cho thiết bị nhúng (MCU).
- **Thành phần**:
    - Sử dụng các lớp `Conv1D` với cơ chế **Dilation** (tăng dần từ 1, 2, 4, 8) qua 2 stacks.
    - **Cấu trúc Residual**: Kết hợp các nhánh nối tắt (skip connections) với `Cropping1D` để đồng bộ kích thước tensor.
    - **Regularization**: Ứng dụng `BatchNormalization` và `Dropout` (tỉ lệ 0.2 trong các block và 0.3 trước lớp Output).
    - **Pooling**: `GlobalAveragePooling1D` được sử dụng để giảm số lượng tham số trước khi đưa qua lớp Dense cuối cùng.
- **Kỹ thuật tối ưu**:
    - **Loss Function**: `sparse_categorical_crossentropy`.
    - **Class Weighting**: Sử dụng chiến lược `class_weight='balanced'` để xử lý vấn đề mất cân bằng dữ liệu giữa các lớp.
    - **Optimizer**: Adam với learning rate 1e-3.