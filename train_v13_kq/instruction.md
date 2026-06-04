# Tài liệu Instruction: Phiên bản TCN_MCU_v13

## 1. Tổng quan Model
- **Mục đích:** Phân loại hành động (Activity Recognition) dựa trên dữ liệu cảm biến IMU.
- **Quy mô:** Thiết kế cho bài toán phân loại đa lớp (default: 4 lớp đầu ra).

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc:** Sử dụng các cửa sổ trượt (windowing) cố định với độ dài 200 mẫu (samples) mỗi tệp.
- **Phân chia:** Dữ liệu được chia theo `subject_id` (Train/Val/Test) để đảm bảo tính độc lập giữa các đối tượng thực hiện thử nghiệm.
- **Gắn nhãn:** Sử dụng ánh xạ `LABEL_MAP` (được đọc trực tiếp từ tiền tố của tên tệp tin).

## 3. Tiền xử lý (Preprocessing)
- **Chuẩn hóa:** Dữ liệu đầu vào được chuẩn hóa qua lớp `BatchNormalization` ngay tại lớp đầu tiên của mô hình.
- **Định dạng dữ liệu:** Chuyển đổi nhãn sang định dạng One-hot Encoding (`to_categorical`) phục vụ cho quá trình huấn luyện với `CategoricalCrossentropy`.
- **Cân bằng dữ liệu:** Tính toán trọng số lớp (`class_weight='balanced'`) để xử lý vấn đề mất cân bằng dữ liệu trong tập huấn luyện.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Sử dụng kiến trúc **TCN (Temporal Convolutional Network)** với 2 stacks, mỗi stack bao gồm các lớp `Conv1D` có độ trễ (dilation rate) tăng dần [1, 2, 4, 8].
- **Cơ chế đặc biệt:** 
    - Áp dụng **Residual Connections** để duy trì luồng thông tin qua các lớp convolution.
    - Sử dụng **Cropping1D** để khớp kích thước trong các khối residual.
    - **GlobalMaxPooling1D:** Được lựa chọn thay thế cho Average Pooling để ưu tiên giữ lại các đỉnh cường độ tín hiệu (đặc biệt hữu ích cho các hành động như đi bộ).
- **Kỹ thuật tối ưu:**
    - **Loss Function:** Sử dụng `CategoricalCrossentropy` kết hợp với **Label Smoothing (0.1)** để giảm overfitting.
    - **Regularization:** Sử dụng `Dropout` (0.2 trong các khối TCN và 0.3 ở lớp dense cuối) để tăng khả năng tổng quát hóa.
    - **Optimizer:** Adam (learning rate = 1e-3).