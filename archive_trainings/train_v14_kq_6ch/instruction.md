# Tài liệu Instruction: Phiên bản TCN-MCU Human Activity Recognition

## 1. Tổng quan Model
- **Mục đích:** Nhận diện hoạt động dựa trên dữ liệu cảm biến IMU (Accel + Gyro).
- **Quy mô:** Thiết kế cho các thiết bị nhúng (MCU), phân loại đầu ra với `n_classes=4`.

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc dữ liệu:** Dữ liệu được chia theo cửa sổ (window) cố định với độ dài 200 điểm mẫu/file.
- **Phân chia tập dữ liệu:** Sử dụng kỹ thuật phân tách theo đối tượng thực hiện (`subject_id`) để đảm bảo tính độc lập giữa tập Train, Val và Test (tránh rò rỉ dữ liệu).
- **Gắn nhãn:** Dựa trên mã nhãn (`label_code`) được trích xuất trực tiếp từ tên tệp tin (sử dụng `LABEL_MAP`).

## 3. Tiền xử lý (Preprocessing)
- **Cắt gọt (Clipping):** Dữ liệu đầu vào được chọn lọc bằng cách giữ lại 6 trục cảm biến chính (`X[:, :, :6]`), tương ứng với 3 trục Gia tốc (Accel) và 3 trục Con quay (Gyro).
- **Định dạng đầu vào:** Dữ liệu được chuẩn hóa thông qua `BatchNormalization` ngay tại lớp đầu tiên của mô hình.
- **Cache:** Sử dụng cơ chế lưu trữ cache (`.npy`) để tối ưu hóa quá trình tải dữ liệu lặp lại.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Temporal Convolutional Network (TCN) với các đặc điểm:
    - **Cấu trúc:** 2 khối (stack), mỗi khối gồm 4 lớp `Conv1D` với các tốc độ giãn nở (dilation rates) lần lượt là [1, 2, 4, 8].
    - **Cơ chế:** Sử dụng kết nối tắt (Residual Connection) với `Add` layer sau mỗi khối `Conv1D` để tránh triệt tiêu gradient.
    - **Pooling:** Sử dụng `GlobalAveragePooling1D` để giảm chiều dữ liệu trước khi đưa vào lớp phân loại cuối.
- **Kỹ thuật tối ưu:**
    - **Regularization:** Sử dụng `Dropout` (0.2 trong block, 0.3 trước đầu ra) và `BatchNormalization` để ổn định huấn luyện.
    - **Cân bằng lớp:** Sử dụng `class_weight='balanced'` để xử lý vấn đề mất cân bằng dữ liệu trong quá trình huấn luyện.
    - **Optimizer:** `Adam` với learning rate 1e-3, hàm mất mát `sparse_categorical_crossentropy`.