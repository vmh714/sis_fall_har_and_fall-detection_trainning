# Tài liệu Instruction: Phiên bản TCN-IMU-6Classes

## 1. Tổng quan Model
- **Mục đích**: Phân loại hoạt động người dùng (Human Activity Recognition - HAR) dựa trên dữ liệu cảm biến quán tính (IMU).
- **Quy mô**: Mô hình đầu ra gồm **6 lớp (classes)**.

## 2. Chiến lược Data & Gắn nhãn
Dữ liệu được gán nhãn dựa trên thông tin tên file (`filename_info`):
- **Fall**: Các file có tiền tố bắt đầu bằng 'F'.
- **Run**: Các file có tiền tố 'D03', 'D04'.
- **Walk**: Các file có tiền tố 'D01', 'D02', 'D05', 'D06'.
- **Trans**: Các file chứa nhãn '_Trans_'.
- **StandSit**: Các file chứa nhãn '_StandSit_'.
- **Lie**: Các file chứa nhãn '_Lie_'.
*Lưu ý: Dữ liệu được cắt cố định thành các đoạn (windows) có chiều dài 200 mẫu (samples).*

## 3. Tiền xử lý (Preprocessing)
- **Định dạng input**: Dữ liệu bao gồm 6 features (ax, ay, az, gx, gy, gz).
- **Cấu trúc**: Mỗi instance có kích thước (200, 6).
- **Chuẩn hóa**: Sử dụng `BatchNormalization` ngay tại lớp đầu vào để ổn định phân phối dữ liệu (gián tiếp xử lý các giá trị scale của Accel/Gyro).

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng**: **Temporal Convolutional Network (TCN)** kết hợp với khối residual.
- **Chi tiết cấu trúc**:
    - Sử dụng 2 stack, mỗi stack bao gồm các lớp `Conv1D` với dilation rate là [1, 2, 4, 8] để mở rộng trường nhìn (receptive field).
    - Áp dụng `BatchNormalization` và `Dropout (0.2)` sau mỗi lớp tích chập.
    - Sử dụng kết nối tắt (**Residual connections**) với `Cropping1D` để khớp kích thước tensor.
    - Lớp gộp: `GlobalAveragePooling1D` để giảm chiều dữ liệu trước khi đưa vào lớp phân loại.
- **Kỹ thuật tối ưu**:
    - **Class Weights**: Sử dụng chiến lược `balanced` để giải quyết vấn đề mất cân bằng dữ liệu giữa các lớp.
    - **Tối ưu hóa**: Sử dụng bộ tối ưu `Adam` (learning rate 1e-3) với hàm mất mát `sparse_categorical_crossentropy`.
    - **Regularization**: Sử dụng `Dropout (0.3)` trước lớp Dense cuối cùng để ngăn chặn overfitting.