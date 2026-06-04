# Tài liệu Instruction: Phiên bản TCN-MCU (Time Convolutional Network for Microcontrollers)

## 1. Tổng quan Model
- **Mục đích:** Phân loại hành vi (Human Activity Recognition - HAR) từ dữ liệu IMU.
- **Quy mô:** Mô hình được thiết kế với đầu ra gồm **4 lớp (classes)**, tối ưu hóa để triển khai trên các thiết bị nhúng (MCU) thông qua TFLite Micro.

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc dữ liệu:** Dữ liệu đầu vào dạng cửa sổ (windowed) với độ dài cố định **200 mẫu/window**.
- **Phân chia tập dữ liệu:** Dựa trên `subject_id` (người thực hiện) để tách biệt hoàn toàn tập Train, Validation và Test, tránh tình trạng rò rỉ dữ liệu (data leakage).
- **Gắn nhãn:** Dựa trên tiền tố của tên file (được định nghĩa trong `LABEL_MAP`).
- **Cân bằng lớp:** Sử dụng kỹ thuật `class_weight='balanced'` để xử lý dữ liệu mất cân bằng (có đề cập đến việc áp dụng trọng số tăng cường cho lớp 'Fall').

## 3. Tiền xử lý (Preprocessing)
- **Chuẩn hóa:** Dữ liệu sau khi nạp được xử lý qua `BatchNormalization` ở lớp đầu tiên của mô hình.
- **Tối ưu Pipeline:**
    - Sử dụng `ThreadPoolExecutor` (8 workers) để nạp dữ liệu song song.
    - Cơ chế lưu Cache dữ liệu (`.npy`) giúp giảm thiểu thời gian đọc file trong các lần chạy sau.
- **Lưu ý kỹ thuật:** Mặc dù code trích xuất không liệt kê cụ thể hệ số nhân cho Accel/Gyro, nhưng logic `prepare_dataset` đảm bảo tính đồng nhất (200 mẫu) trước khi đưa vào mô hình.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** **Temporal Convolutional Network (TCN)** với cấu trúc 2 stacks, mỗi stack gồm 4 block sử dụng `dilation_rate` [1, 2, 4, 8] để mở rộng trường nhìn (receptive field) theo thời gian.
- **Đặc điểm thiết kế:**
    - Sử dụng **Residual Connections** (Add) để hỗ trợ huấn luyện mạng sâu.
    - Sử dụng **Causal Padding** để đảm bảo tính nhân quả (phù hợp với dữ liệu chuỗi thời gian).
    - **Pooling:** `GlobalAveragePooling1D` để nén đặc trưng trước lớp phân loại.
    - **Dropout:** Tỷ lệ 0.2 trong các block và 0.3 ở lớp Fully Connected để giảm Overfitting.
- **Kỹ thuật tối ưu:**
    - Tối ưu hóa cho TFLite (Full INT8 Quantization).
    - Optimizer: Adam (Learning rate 1e-3).
    - Loss function: `sparse_categorical_crossentropy`.