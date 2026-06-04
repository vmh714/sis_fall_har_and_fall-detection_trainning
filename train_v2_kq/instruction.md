# Tài liệu Instruction: Phiên bản HAR-Hybrid-CNN-LSTM

## 1. Tổng quan Model
- **Mục đích:** Nhận diện hoạt động con người (Human Activity Recognition - HAR) dựa trên dữ liệu cảm biến IMU.
- **Quy mô đầu ra:** Phân loại thành **4 lớp (classes)** với hàm kích hoạt `softmax` ở lớp cuối.

## 2. Chiến lược Data & Gắn nhãn
- **Phương pháp gán nhãn:** Dựa trên quy ước đặt tên file `Dxx_Sxx_Rxx_Wxxx` (trong đó `Dxx` là mã định danh nhãn).
- **Phân chia dữ liệu:** Sử dụng kỹ thuật **LSO (Leave-Subject-Out)**:
    - `TRAIN_SUBJECTS`: Dữ liệu dùng để huấn luyện.
    - `VAL_SUBJECTS`: Dữ liệu dùng để tinh chỉnh và kiểm tra validation.
    - `TEST_SUBJECTS`: Dữ liệu độc lập dùng để đánh giá mô hình cuối cùng.
- **Cấu trúc dữ liệu:** Mỗi mẫu dữ liệu là một cửa sổ (window) có kích thước cố định là **200 bước thời gian (timesteps)** với 6 đặc trưng (3 trục Accel + 3 trục Gyro).

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc dữ liệu đầu vào:** Dữ liệu được đọc từ các file CSV, mỗi file đại diện cho một window 200 mẫu.
- **Tối ưu hiệu năng:** 
    - Sử dụng `ThreadPoolExecutor` (8 workers) để tăng tốc độ tải dữ liệu I/O từ đĩa.
    - Cơ chế **Caching**: Dữ liệu đã xử lý được lưu dưới dạng file `.npy` để tăng tốc độ khởi tạo cho các lần chạy sau.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Hybrid CNN-LSTM.
    - **CNN (Spatial Extraction):** Gồm 2 lớp `Conv1D` (64 filters, kernel 3) kết hợp với `MaxPooling1D` để trích xuất đặc trưng không gian cục bộ.
    - **LSTM (Temporal Learning):** Gồm 2 lớp `LSTM` (128 và 64 units) để học mối quan hệ tuần tự/thời gian.
- **Kỹ thuật tối ưu:**
    - **Regularization:** Sử dụng `Dropout` (tỉ lệ 0.4 sau CNN và 0.5 sau LSTM) để chống overfitting.
    - **Xử lý mất cân bằng:** Sử dụng `class_weight='balanced'` để tính toán trọng số cho từng lớp, giúp mô hình tập trung vào các class thiểu số.
    - **Optimizer:** `Adam` với learning rate `1e-3`.
    - **Loss Function:** `sparse_categorical_crossentropy`.