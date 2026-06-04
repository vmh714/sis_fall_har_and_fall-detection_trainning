# Tài liệu Instruction: Phiên bản HAR & Fall Detection Trunk-and-Branches Model

## 1. Tổng quan Model
- **Mục đích:** Mô hình được thiết kế để phân loại hoạt động con người (HAR) và phát hiện ngã (Fall Detection) dựa trên dữ liệu cảm biến IMU (Gia tốc và Góc quay).
- **Quy mô đầu ra:** Phân loại thành **4 lớp (4 classes)**, được xác định thông qua hàm kích hoạt Softmax tại lớp cuối cùng.

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc dữ liệu:** Dữ liệu đầu vào là các cửa sổ thời gian (windows) với kích thước cố định **200 mẫu (samples)** cho mỗi file CSV.
- **Gắn nhãn:** Sử dụng hàm `parse_filename_info` để giải mã nhãn từ tên file (định dạng `D01_SA01_R01_W000`).
- **Phân chia dữ liệu:** Thực hiện theo phương pháp **LSO (Leave-Subject-Out)**, chia tập Train/Val/Test dựa trên định danh đối tượng (`subject_id`) để đảm bảo tính tổng quát hóa.

## 3. Tiền xử lý (Preprocessing)
- **Định dạng đầu vào:** Dữ liệu 6 trục (ax, ay, az, gx, gy, gz).
- **Tối ưu hóa:** Sử dụng `ThreadPoolExecutor` để đọc song song dữ liệu từ ổ đĩa và áp dụng cơ chế lưu **Cache (.npy)** giúp rút ngắn thời gian chuẩn bị dữ liệu cho các lần chạy tiếp theo.
- **Xử lý mất cân bằng:** Sử dụng kỹ thuật `compute_class_weight` với tham số `class_weight='balanced'` để tính toán trọng số cho các lớp, giúp mô hình tập trung hơn vào các lớp thiểu số.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Sử dụng kiến trúc **Trunk-and-Branches (Late-Branching)**:
    - **Trunk (Thân cây):** Gồm 2 lớp `Conv1D` kết hợp `BatchNormalization` và `MaxPooling1D` để trích xuất đặc trưng và lọc nhiễu chung.
    - **Fall Expert (Nhánh 1):** Sử dụng `GlobalMaxPooling1D` để nhận diện các thay đổi đột ngột (đặc trưng của việc ngã).
    - **HAR Expert (Nhánh 2):** Sử dụng kiến trúc 2 tầng `LSTM` để phân tích tính chuỗi thời gian của các hoạt động hàng ngày.
- **Hợp nhất (Merge):** Kết hợp kết quả từ hai nhánh qua lớp `Concatenate` trước khi đưa vào lớp phân loại cuối cùng.
- **Kỹ thuật tối ưu:**
    - **Regularization:** Sử dụng `L2 Regularization` (0.001) trên các lớp Dense/Conv1D và `Dropout` (0.3 - 0.4) để tránh quá mức (overfitting).
    - **Optimizer:** `Adam` với learning rate `1e-3`.
    - **Loss Function:** `sparse_categorical_crossentropy`.