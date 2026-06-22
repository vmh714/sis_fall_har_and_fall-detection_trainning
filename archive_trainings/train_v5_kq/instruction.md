# Tài liệu Instruction: Phiên bản HAR & Fall Detection Model

## 1. Tổng quan Model
- **Mục đích:** Mô hình lai (hybrid) tích hợp phát hiện ngã (Fall Detection) và nhận diện hoạt động thường ngày (HAR).
- **Quy mô:** Đầu ra gồm 4 lớp (4 classes) sử dụng hàm kích hoạt Softmax.

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc:** Sử dụng dữ liệu dạng cửa sổ (windowed data) với độ dài cố định 200 mẫu/file.
- **Phương pháp phân chia:** Sử dụng chiến lược **LSO (Leave-Subject-Out)**: chia tập Train/Val/Test dựa trên ID đối tượng (`subject_id`) để đảm bảo tính tổng quát hóa cho người mới.
- **Nhãn:** Được trích xuất từ tiền tố của tên file (`label_code`) thông qua từ điển `LABEL_MAP`.

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc dữ liệu:** Input bao gồm 6 kênh (3 trục gia tốc - Accel, 3 trục góc quay - Gyro).
- **Kiểm soát chất lượng:** Loại bỏ các file không đạt chuẩn kích thước (không đủ 200 mẫu).
- **Tăng tốc xử lý:** Sử dụng `ThreadPoolExecutor` để đọc dữ liệu I/O song song và cơ chế lưu trữ Cache (`.npy`) để tối ưu thời gian tải dữ liệu cho các lần chạy sau.

## 4. Kiến trúc mô hình (Architecture)
- **Kiến trúc:** **Two-Stream (Phân nhánh)**:
    - **Nhánh Fall Expert:** Sử dụng `Conv1D` kết hợp `GlobalMaxPooling1D` để trích xuất các đặc trưng đột biến mạnh (cú ngã).
    - **Nhánh HAR Expert:** Sử dụng `MaxPooling1D` kết hợp hai lớp `LSTM` để học các phụ thuộc theo thời gian (chuỗi nhịp điệu) của hoạt động.
    - **Hợp nhất:** Kết hợp đặc trưng từ hai nhánh qua lớp `Concatenate`.
- **Kỹ thuật tối ưu:**
    - **Regularization:** Sử dụng `L2 Regularization` (0.001) cho hầu hết các lớp Dense và Conv để tránh Overfitting.
    - **Class Weight:** Sử dụng kỹ thuật `compute_class_weight` với tham số `balanced` để giải quyết vấn đề mất cân bằng dữ liệu giữa các lớp.
    - **Normalization:** Áp dụng `BatchNormalization` để ổn định quá trình hội tụ.
    - **Optimizer:** Adam (learning_rate=1e-3) và Loss function là `sparse_categorical_crossentropy`.