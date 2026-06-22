# Tài liệu Instruction: Phiên bản TCN_MCU_v15

## 1. Tổng quan Model
- **Mục đích:** Phân loại hoạt động dựa trên dữ liệu cảm biến IMU (Cảm biến quán tính).
- **Quy mô:** Thiết kế cho hệ thống nhúng (MCU), mô hình đầu ra phân loại 4 lớp (classes).

## 2. Chiến lược Data & Gắn nhãn
- **Cơ chế:** Gán nhãn dựa trên tiền tố của tên tệp (filename).
- **Phân chia tập dữ liệu:** Dữ liệu được chia theo `subject_id` (Train/Val/Test) để đảm bảo tính độc lập giữa người thực hiện.
- **Độ dài cửa sổ:** Cố định 200 mẫu (samples) cho mỗi tệp đầu vào.
- **Cân bằng dữ liệu:** Sử dụng kỹ thuật `class_weight='balanced'` để xử lý sự mất cân bằng giữa các lớp trong quá trình huấn luyện.

## 3. Tiền xử lý (Preprocessing)
- **Định dạng dữ liệu:** Cắt gọt dữ liệu thành các window có độ dài 200.
- **Chuẩn hóa:** Dữ liệu được đưa vào mô hình dưới dạng `float32`. (Lưu ý: Logic scale cho Gyro/Accel được thực hiện trước đó trong quá trình windowing, dữ liệu đầu vào hiện tại đã qua xử lý sẵn).

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Temporal Convolutional Network (TCN) với các lớp `Conv1D` tích hợp cơ chế Dilated Convolutions (dilation rates: 1, 2, 4, 8) và kết nối tắt (Residual connections).
- **Thành phần chính:**
    - **Backbone:** 2 tầng TCN, mỗi tầng sử dụng `BatchNormalization` và `Dropout (0.2)` để chống overfitting.
    - **Pooling:** Sử dụng `GlobalAveragePooling1D` để trích xuất đặc trưng từ chuỗi thời gian.
    - **Head (v15):** Thêm lớp `Dense(64)` kèm `ReLU` và `BatchNormalization` để tăng cường khả năng phân biệt giữa các lớp có đặc điểm tương đồng (ví dụ: Walk vs Static).
- **Kỹ thuật tối ưu:**
    - **Optimizer:** Adam (learning rate = 1e-3).
    - **Loss:** `sparse_categorical_crossentropy`.
    - **Regularization:** `Dropout (0.3)` tại lớp Dense cuối, không sử dụng Label Smoothing trong phiên bản này.