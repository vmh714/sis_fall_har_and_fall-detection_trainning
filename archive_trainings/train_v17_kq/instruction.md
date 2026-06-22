# Tài liệu Instruction: Phiên bản TCN-IMU-6-Class

## 1. Tổng quan Model
- **Mục đích**: Phân loại hoạt động con người (HAR - Human Activity Recognition) dựa trên dữ liệu cảm biến IMU (6 features).
- **Quy mô**: 6 lớp đầu ra (classes).

## 2. Chiến lược Data & Gắn nhãn
Dữ liệu được phân loại dựa trên hậu tố/tiền tố của tên tệp tin:
- **Trans**: Chuyển trạng thái (Transition).
- **StandSit**: Đứng hoặc ngồi.
- **Lie**: Nằm.
- **Walk**: Các tệp có tiền tố `D01, D02, D05, D06`.
- **Run**: Các tệp có tiền tố `D03, D04`.
- **Fall**: Các tệp có tiền tố `F`.
- **Độ dài cửa sổ**: Cố định 200 mẫu (samples) mỗi tệp tin.

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc dữ liệu**: Mỗi đầu vào có 6 features (được nạp trực tiếp từ file CSV sau khi kiểm tra độ dài 200 samples).
- **Phân chia dữ liệu**: Dữ liệu được chia theo `subject_id` (Train/Val/Test) để đảm bảo không bị rò rỉ dữ liệu giữa các tập.
- **Cache**: Có cơ chế lưu trữ cache dữ liệu (`.npy`) để tối ưu hóa quá trình tải dữ liệu lặp lại.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng**: Temporal Convolutional Network (TCN) sử dụng các lớp `Conv1D` với kỹ thuật **Dilation** (tăng dần từ 1, 2, 4, 8) và kết nối tắt (**Residual connections**) để giải quyết bài toán vanishing gradient.
- **Cấu trúc chi tiết**:
    - Sử dụng `BatchNormalization` và `Dropout` (0.2) sau các lớp Convolution.
    - Cấu trúc Residual gồm 2 stacks, mỗi stack có 4 lớp TCN.
    - `GlobalAveragePooling1D` ở tầng cuối trước lớp `Dense`.
- **Kỹ thuật tối ưu**:
    - **Class Weighting**: Sử dụng `class_weight='balanced'` để xử lý dữ liệu bị mất cân bằng giữa các lớp.
    - **Optimizer**: Adam với learning rate `1e-3`.
    - **Loss Function**: `sparse_categorical_crossentropy`.