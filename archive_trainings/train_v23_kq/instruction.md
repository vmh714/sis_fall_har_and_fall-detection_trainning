# Tài liệu Instruction: Phiên bản TCN-MCU-Human-Activity-Recognition

## 1. Tổng quan Model
- **Mục đích**: Nhận diện hoạt động người (Human Activity Recognition - HAR) dựa trên dữ liệu cảm biến IMU (6 trục).
- **Quy mô đầu ra**: Phân loại 5 lớp hoạt động (Classes).

## 2. Chiến lược Data & Gắn nhãn
Dữ liệu được gắn nhãn dựa trên tiền tố của tên tệp (filename) và cấu trúc thư mục, cụ thể:
- **Fall**: Các tệp có tiền tố bắt đầu bằng 'F'.
- **Walk**: Các tệp có tiền tố 'D01', 'D02', 'D05', 'D06'.
- **Run**: Các tệp có tiền tố 'D03', 'D04'.
- **Idle**: Các tệp chứa '_StandSit_' hoặc '_Lie_'.
- **Trans**: Các tệp chứa '_Trans_'.
- **Yêu cầu dữ liệu**: Window size cố định là 200 mẫu (samples) trên mỗi tệp CSV.

## 3. Tiền xử lý (Preprocessing)
- **Input**: Dữ liệu chuỗi thời gian 6 features.
- **Cấu trúc**: Phân chia tập Train/Val/Test dựa trên định danh chủ thể (`subject_id`).
- **Cân bằng dữ liệu**: Sử dụng kỹ thuật `compute_class_weight` với tham số `'balanced'` để xử lý sự mất cân bằng giữa các lớp.
- **Định dạng**: Chuyển đổi nhãn sang dạng One-Hot Encoding (`to_categorical`) cho quá trình huấn luyện.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng**: Kiến trúc dựa trên **Residual Conv1D** kết hợp với **Squeeze-and-Excitation (SE) Block**.
- **Đặc điểm nổi bật**:
    - **TCN-like**: Sử dụng 4 khối Residual Conv1D (kernel size 7), thay thế Dilation bằng Strides để giảm chiều không gian.
    - **SE Block**: Áp dụng attention cơ chế Squeeze-and-Excitation (tỉ lệ giảm 4) để tối ưu hóa trọng số kênh (phù hợp cho thiết bị MCU).
    - **Pooling**: Kết hợp `GlobalAveragePooling1D` và `GlobalMaxPooling1D` (Concatenate) ở lớp cuối để trích xuất đặc trưng đa dạng.
- **Kỹ thuật tối ưu**:
    - **Label Smoothing**: Áp dụng `label_smoothing=0.1` trong hàm mất mát `CategoricalCrossentropy` để chống overfitting.
    - **Regularization**: Sử dụng `BatchNormalization` và `Dropout` (0.2 - 0.4) giữa các lớp.
    - **Optimizer**: Adam với `learning_rate=1e-3`.