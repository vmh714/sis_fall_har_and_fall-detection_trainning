# Tài liệu Instruction: Phiên bản TCN_MCU_IMU_Classifier

## 1. Tổng quan Model
- **Mục đích:** Phân loại hành động dựa trên dữ liệu cảm biến IMU (Accel & Gyro).
- **Quy mô:** Thiết kế cho bài toán đa lớp (mặc định `n_classes=4`), phù hợp với các thiết bị biên (MCU).

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc dữ liệu:** Dữ liệu đầu vào được cắt thành các cửa sổ (window) cố định với độ dài **200 mẫu (samples)**.
- **Phân chia tập dữ liệu:** Sử dụng chiến lược tách theo người thực hiện (`subject_id`) thành 3 tập riêng biệt: `Train`, `Val` và `Test` để đảm bảo tính tổng quát hóa.
- **Gán nhãn:** Dựa trên tiền tố của tên file (`label_code` trong `LABEL_MAP`).

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc dữ liệu:** Chỉ giữ lại 6 trục đầu tiên (tương ứng với 3 trục Accelerometer và 3 trục Gyroscope).
- **Bộ nhớ đệm:** Sử dụng cơ chế `cache` (.npy files) để tăng tốc độ tải dữ liệu trong các lần huấn luyện sau.
- **Chuẩn hóa:** Sử dụng `BatchNormalization` ngay tại lớp đầu vào của mô hình để ổn định phân phối dữ liệu đầu vào.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Temporal Convolutional Network (TCN) với 2 lớp xếp chồng (stacks), mỗi lớp sử dụng các dilation rate [1, 2, 4, 8] để mở rộng trường tiếp nhận (receptive field).
- **Cấu trúc bổ trợ:**
    - Sử dụng **Residual Connection** (Add layer) để giảm thiểu hiện tượng mất mát gradient.
    - **Cropping1D:** Xử lý sự chênh lệch kích thước do sử dụng `padding='valid'` trong các lớp Convolution.
- **Kỹ thuật tối ưu:**
    - **Class Weights:** Sử dụng `class_weight='balanced'` để xử lý vấn đề mất cân bằng dữ liệu giữa các lớp.
    - **Regularization:** Kết hợp `Dropout` (0.2 - 0.3) và `GlobalAveragePooling1D` để chống overfitting.
    - **Trình tối ưu:** Adam optimizer (learning rate 1e-3) với hàm mất mát `sparse_categorical_crossentropy`.