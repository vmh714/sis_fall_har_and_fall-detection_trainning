# Tài liệu Instruction: Phiên bản TCN_MCU_IMU_Classifier

## 1. Tổng quan Model
- **Mục đích**: Phân loại các hoạt động dựa trên dữ liệu cảm biến IMU (Inertial Measurement Unit) theo chuỗi thời gian.
- **Quy mô**: Mô hình được thiết kế để phân loại 4 lớp (default `n_classes=4`).

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc dữ liệu**: Dữ liệu đầu vào được chia thành các cửa sổ (windows) cố định với độ dài 200 mẫu (samples).
- **Phân chia tập dữ liệu**: Dữ liệu được gán nhãn dựa trên tiền tố của tên file (`label_code`) và được phân tách theo `subject_id` thành 3 tập: Train, Validation và Test để đảm bảo tính độc lập giữa các đối tượng.
- **Cân bằng dữ liệu**: Sử dụng kỹ thuật tính toán `class_weight='balanced'` để xử lý tình trạng mất cân bằng giữa các lớp trong quá trình huấn luyện.

## 3. Tiền xử lý (Preprocessing)
- **Độ dài cửa sổ**: Cố định 200 điểm dữ liệu cho mỗi sample đầu vào.
- **Chuẩn hóa**: Dữ liệu được nạp trực tiếp qua các file đã được windowing, sau đó được mô hình hóa thành dạng `float32`.
- **Thông số kỹ thuật**: Mặc dù mã nguồn không trực tiếp hiển thị hệ số scale cụ thể, mô hình sử dụng `BatchNormalization` ngay từ lớp đầu vào để ổn định phân phối dữ liệu (giảm ảnh hưởng của việc chênh lệch giá trị cảm biến).

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng**: Kiến trúc **TCN (Temporal Convolutional Network)** tối ưu cho dữ liệu chuỗi thời gian.
- **Thành phần chính**:
    - **Initial Layer**: Một lớp `Conv1D` (kernel=1) để chuẩn hóa số kênh lên 32.
    - **TCN Blocks**: 2 khối (stacks), mỗi khối chứa 4 lớp tích chập với các tốc độ giãn nở (dilation rates) lần lượt là [1, 2, 4, 8] giúp mô hình học các phụ thuộc xa.
    - **Cơ chế Shortcut**: Sử dụng phép cộng `Add()` (Residual connection) để duy trì luồng gradient và tránh suy giảm tín hiệu qua các lớp sâu.
    - **Pooling**: Sử dụng `GlobalAveragePooling1D` để nén thông tin chuỗi trước khi đưa vào lớp phân loại cuối cùng.
- **Kỹ thuật tối ưu**:
    - **Regularization**: Sử dụng `Dropout` (0.2 trong các block TCN và 0.3 trước lớp Dense) để tránh Overfitting.
    - **Optimizer**: Adam (learning rate = 1e-3).
    - **Loss Function**: `sparse_categorical_crossentropy`.