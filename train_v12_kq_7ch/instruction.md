# Tài liệu Instruction: Phiên bản TCN-MCU (Temporal Convolutional Network for MCU)

## 1. Tổng quan Model
- **Mục đích**: Nhận diện hoạt động (Activity Recognition) dựa trên dữ liệu cảm biến IMU.
- **Quy mô**: Phân loại 4 lớp (n_classes=4).

## 2. Chiến lược Data & Gắn nhãn
- **Dữ liệu đầu vào**: Dữ liệu IMU gồm 6 trục gốc (3 Accel, 3 Gyro).
- **Kỹ thuật bổ sung**: Tích hợp thêm kênh **SVM (Signal Vector Magnitude)** từ 6 trục gốc, tạo thành tập dữ liệu đầu vào 7 trục.
- **Cân bằng dữ liệu**: Sử dụng `class_weight='balanced'` trong quá trình huấn luyện để xử lý vấn đề mất cân bằng giữa các lớp.

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc**: Input được chuẩn hóa thông qua `BatchNormalization` ngay tại lớp đầu vào.
- **Nhiễu**: Áp dụng `GaussianNoise(0.01)` để tăng tính ổn định cho mô hình, đặc biệt bảo vệ các đặc trưng của kênh SVM vốn rất nhạy cảm.
- **Lưu ý**: Dữ liệu được nạp từ các file `.npy` đã được lưu cache trước đó.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng**: Temporal Convolutional Network (TCN) với cơ chế Residual Connection.
- **Cấu hình lớp Conv1D**:
    - **Stacking**: 2 stack, mỗi stack bao gồm các lớp với `dilation_rate` lần lượt là [1, 2, 4, 8].
    - **Filter**: 64 filters cho lớp đầu tiên, 32 filters cho các lớp tiếp theo.
    - **Kernel size**: 3, padding 'valid'.
- **Kỹ thuật tối ưu**:
    - **Residual Connection**: Sử dụng `Cropping1D` để căn chỉnh kích thước tensor trước khi cộng (Add) vào nhánh residual.
    - **Regularization**: Sử dụng `Dropout` (0.2 trong các block và 0.3 trước lớp đầu ra) để chống overfitting.
    - **Pooling**: `GlobalAveragePooling1D` để nén dữ liệu trước khi đưa vào lớp Dense cuối cùng.
    - **Optimizer**: Adam với learning rate = 1e-3, hàm mất mát `sparse_categorical_crossentropy`.