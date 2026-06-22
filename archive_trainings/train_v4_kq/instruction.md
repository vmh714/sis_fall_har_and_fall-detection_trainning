# Tài liệu Instruction: Phiên bản HAR Hybrid CNN-LSTM

## 1. Tổng quan Model
- **Mục đích:** Nhận diện hoạt động con người (Human Activity Recognition - HAR) dựa trên dữ liệu cảm biến quán tính (IMU).
- **Đầu ra:** Mô hình phân loại 4 lớp (classes) hoạt động khác nhau.

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc:** Sử dụng chiến lược chia tập dữ liệu **LSO (Leave-Subject-Out)** dựa trên `subject_id` để đảm bảo mô hình có khả năng tổng quát hóa trên đối tượng mới.
- **Dữ liệu đầu vào:** Mỗi mẫu (window) là một file CSV có kích thước cố định là 200 bước thời gian (time-steps) với 6 đặc trưng (ax, ay, az, gx, gy, gz).
- **Gắn nhãn:** Dựa trên tiền tố `label_code` trong tên tệp tin (ví dụ: `D01_SA01_R01_W000` được ánh xạ thông qua `LABEL_MAP`).

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc dữ liệu:** Chuẩn hóa đầu vào về dạng `(200, 6)`.
- **Cơ chế tải:** Sử dụng `ThreadPoolExecutor` để đọc dữ liệu song song nhằm tối ưu hóa I/O, kết hợp lưu trữ dạng Cache (`.npy`) để tăng tốc độ khởi tạo.
- **Lưu ý:** Code hiện tại giả định dữ liệu đã được chuẩn hóa (scale) hoặc ở định dạng thô phù hợp với kiến trúc; các tham số clip (8g) và scale cụ thể sẽ được áp dụng tại bước chuẩn bị file CSV đầu vào (theo logic ngoài phạm vi hàm `prepare_dataset`).

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Kiến trúc lai **CNN-LSTM** (Sequential model).
    - **Feature Extraction:** 2 lớp `Conv1D` (kernel size 3) kết hợp `BatchNormalization` và `MaxPooling1D`.
    - **Sequence Modeling:** 2 lớp `LSTM` (64 và 32 units) để trích xuất phụ thuộc thời gian.
    - **Regularization:** Sử dụng `L2 Regularization` (0.001) cho tất cả các lớp và `Dropout` (0.3 - 0.4) để tránh quá mức (overfitting).
- **Kỹ thuật tối ưu:**
    - **Class Weights:** Sử dụng `compute_class_weight='balanced'` để xử lý tình trạng mất cân bằng dữ liệu giữa các lớp.
    - **Loss Function:** `sparse_categorical_crossentropy`.
    - **Optimizer:** `Adam` (learning rate 1e-3).