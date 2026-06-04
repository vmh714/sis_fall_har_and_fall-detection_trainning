# Tài liệu Instruction: Phiên bản HAR Hybrid CNN-LSTM

## 1. Tổng quan Model
- **Mục đích:** Nhận dạng hành động người dùng (HAR - Human Activity Recognition) dựa trên dữ liệu cảm biến IMU (Gia tốc và Góc quay).
- **Quy mô đầu ra:** Phân loại 4 lớp (4 classes) hành động.

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc dữ liệu:** Dữ liệu được chia theo cửa sổ (window) cố định với 200 mẫu/window.
- **Phương pháp phân chia:** Sử dụng chiến lược **Leave-Subject-Out (LSO)**, dữ liệu được tách biệt hoàn toàn giữa các tập Train, Val và Test dựa trên ID đối tượng (`subject_id`) để đảm bảo tính tổng quát hóa cho mô hình.
- **Gắn nhãn:** Sử dụng file định dạng `D01_SA01_R01_W000`, trong đó `D01` tương ứng với mã nhãn (Label code) được ánh xạ qua `LABEL_MAP`.

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc input:** Dữ liệu bao gồm 6 cột (ax, ay, az, gx, gy, gz).
- **Định dạng dữ liệu:** Loại bỏ các window không đủ 200 mẫu để duy trì tính nhất quán đầu vào.
- **Tối ưu hóa I/O:** Sử dụng `ThreadPoolExecutor` để đọc dữ liệu song song và hệ thống `cache` (file `.npy`) để tăng tốc độ tải dữ liệu cho các lần huấn luyện tiếp theo.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Kiến trúc lai (Hybrid) kết hợp:
    - **CNN (Conv1D):** Hai lớp trích xuất đặc trưng cục bộ (64 filters, kernel size 3) kết hợp với `MaxPooling1D` và `Dropout(0.3)`.
    - **LSTM:** Hai lớp RNN (128 và 64 units) để học phụ thuộc tuần tự theo thời gian, kết hợp `Dropout(0.4)` để chống overfitting.
    - **Classifier:** Lớp Dense(32) dẫn tới đầu ra Softmax(4).
- **Kỹ thuật tối ưu:**
    - **Optimizer:** Adam (Learning rate = 5e-4).
    - **Class Weights:** Áp dụng `compute_class_weight` (chế độ 'balanced') để xử lý tình trạng mất cân bằng dữ liệu giữa các lớp nhãn.
    - **Loss function:** `sparse_categorical_crossentropy`.