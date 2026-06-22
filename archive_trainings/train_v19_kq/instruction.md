# Tài liệu Instruction: Phiên bản TCN-IMU Human Activity Recognition

## 1. Tổng quan Model
- **Mục đích:** Phân loại hoạt động con người dựa trên dữ liệu cảm biến IMU (cảm biến quán tính).
- **Quy mô:** Mô hình đầu ra phân loại 5 lớp (5 classes).

## 2. Chiến lược Data & Gắn nhãn
Dữ liệu được gán nhãn dựa trên tiền tố của tên file CSV, được chia thành 5 nhóm lớp chính:
- **Trans:** Dành cho các tệp có chứa chuỗi `_Trans_`.
- **Idle:** Dành cho các trạng thái tĩnh (`_StandSit_` hoặc `_Lie_`).
- **Walk:** Dành cho các hoạt động di chuyển (các tiền tố file: `D01`, `D02`, `D05`, `D06`).
- **Run:** Dành cho các hoạt động chạy (các tiền tố file: `D03`, `D04`).
- **Fall:** Dành cho các sự kiện té ngã (tiền tố file bắt đầu bằng `F`).

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc dữ liệu:** Mỗi mẫu (window) dữ liệu yêu cầu cố định độ dài là 200 điểm thời gian (time steps) với 6 features đầu vào.
- **Phân bổ:** Dữ liệu được chia theo `subject_id` (Train/Val/Test) để đảm bảo tính độc lập của người dùng trong quá trình huấn luyện và kiểm thử.
- **Định dạng:** Sử dụng cache (định dạng `.npy`) để tăng tốc độ tải dữ liệu trong các lần chạy sau.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Temporal Convolutional Network (TCN) sử dụng các khối Residual Conv1D với kỹ thuật Dilation (tăng dần từ 1, 2, 4, 8) qua 2 stack.
- **Cấu phần:**
    - Sử dụng `BatchNormalization` để ổn định training.
    - `Dropout` (0.2 - 0.3) được thêm vào sau lớp tích chập và lớp Global Average Pooling để chống overfitting.
    - `GlobalAveragePooling1D` được sử dụng thay vì Flatten để giảm số lượng tham số trước lớp Dense cuối cùng.
- **Kỹ thuật tối ưu:**
    - **Class Weights:** Sử dụng `compute_class_weight='balanced'` để xử lý vấn đề mất cân bằng dữ liệu giữa các lớp.
    - **Optimizer:** Adam với learning rate = 1e-3.
    - **Loss:** `sparse_categorical_crossentropy` cho phân loại đa lớp.