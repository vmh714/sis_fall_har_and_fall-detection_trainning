# Tài liệu Instruction: Phiên bản TCN-MCU (Temporal Convolutional Network)

## 1. Tổng quan Model
- **Mục đích:** Phân loại hành vi (Activity Recognition) dựa trên dữ liệu cảm biến quán tính.
- **Quy mô:** Mô hình phân loại 4 lớp (n_classes = 4) đầu ra thông qua hàm kích hoạt Softmax.

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc:** Sử dụng bộ `LABEL_MAP` (được trích xuất từ tiền tố tên file, ví dụ: `label_code_subject_id`).
- **Gắn nhãn:** Dữ liệu đầu vào yêu cầu độ dài cố định là 200 điểm dữ liệu (time steps).
- **Cân bằng dữ liệu:** Sử dụng kỹ thuật `class_weight='balanced'` để tự động tính toán trọng số cho các lớp, đặc biệt ưu tiên xử lý dữ liệu mất cân bằng cho class "Fall" (ngã).

## 3. Tiền xử lý (Preprocessing)
- **Cắt gọt dữ liệu:** Chỉ giữ lại 6 trục cảm biến (3 trục Accelerometer + 3 trục Gyroscope).
- **Định dạng:** Dữ liệu nạp từ cache dưới dạng `.npy` với kích thước đầu vào `(samples, 200, 6)`.
- **Lưu ý:** Quy trình yêu cầu tiền xử lý đồng nhất cho tập Train, Val và Test trước khi đưa vào mô hình.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Temporal Convolutional Network (TCN) với 2 stacks, mỗi stack gồm 4 lớp Conv1D sử dụng dilation rate lần lượt là `[1, 2, 4, 8]`.
- **Kỹ thuật tối ưu:**
    - **Data Augmentation:** Thêm `GaussianNoise(0.05)` ở tầng đầu vào để tăng tính ổn định.
    - **Residual Connections:** Sử dụng kết nối tắt (shortcut) có điều chỉnh shape qua lớp `Cropping1D` và `Conv1D(1)` để duy trì luồng dữ liệu.
    - **Cải tiến Convolution:** Lớp Conv1D đầu tiên (stack 0, d=1) tăng lên 64 filters, các lớp còn lại sử dụng 32 filters.
    - **Regularization:** Sử dụng `BatchNormalization` sau mỗi lớp Conv và `Dropout(0.2)` trong block, `Dropout(0.3)` sau lớp `GlobalAveragePooling1D`.
    - **Loss Function:** `sparse_categorical_crossentropy` với Optimizer Adam (learning rate = 1e-3).