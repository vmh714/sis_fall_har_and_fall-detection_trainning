# Tài liệu Instruction: Phiên bản HAR & Fall Detection Trunk-and-Branches Model

## 1. Tổng quan Model
- **Mục đích:** Phát hiện té ngã (Fall detection) và phân loại hoạt động thể chất (HAR).
- **Quy mô:** Mô hình đầu ra gồm 4 lớp (4 classes), sử dụng hàm kích hoạt Softmax để phân loại đa lớp.

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc nhãn:** Dữ liệu được gán nhãn dựa trên tiền tố của tên file (định dạng `D01_SA01_...`).
- **Phân chia:** Sử dụng chiến lược **LSO (Leave-Subject-Out)** để chia tập dữ liệu thành Train, Validation và Test dựa trên ID đối tượng (`subject_id`), đảm bảo mô hình không học vẹt đặc trưng cá nhân.

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc dữ liệu đầu vào:** Mỗi cửa sổ (window) có độ dài cố định là 200 mẫu.
- **Tính năng mở rộng:** Từ 6 cột dữ liệu thô ban đầu (ax, ay, az, gx, gy, gz), mô hình tính toán thêm 3 đặc trưng vật lý quan trọng:
    - **Pitch & Roll:** Tính từ dữ liệu gia tốc (góc nghiêng).
    - **SVM (Signal Vector Magnitude):** Tính từ gia tốc để đo cường độ vận động.
- **Kích thước vector:** Tổng cộng 9 đặc trưng (9 features) cho mỗi điểm thời gian.
- **Lưu trữ:** Hỗ trợ lưu trữ/tải cache dạng `.npy` để tăng tốc độ huấn luyện.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Kiến trúc **Trunk-and-Branches (Late-Branching)**:
    - **Thân cây (Trunk):** Sử dụng 2 lớp `Conv1D` kết hợp `BatchNormalization` và `MaxPooling1D` để trích xuất đặc trưng không gian và lọc nhiễu.
    - **Nhánh 1 (Fall Expert):** Sử dụng `GlobalMaxPooling1D` để nhận diện các biến đổi đột ngột (đặc trưng của té ngã).
    - **Nhánh 2 (HAR Expert):** Sử dụng 2 lớp `LSTM` để nắm bắt tính chuỗi thời gian và nhịp điệu của các hoạt động.
- **Hợp nhất:** Kết hợp kết quả từ hai nhánh qua `Concatenate` trước khi đưa vào lớp `Dense` phân loại cuối cùng.
- **Kỹ thuật tối ưu:**
    - **Class Weights:** Sử dụng `compute_class_weight` (balanced) để xử lý mất cân bằng dữ liệu giữa các lớp.
    - **Regularization:** Áp dụng `L2 Regularization` (0.001) trên các lớp Dense/Conv1D và `Dropout` (0.3 - 0.4) để tránh Overfitting.
    - **Trình tối ưu:** Adam (learning rate 1e-3).