# Tài liệu Instruction: Phiên bản TCN-SE Fall Detection

## 1. Tổng quan Model
- **Mục đích:** Phân loại hoạt động của con người dựa trên dữ liệu cảm biến IMU (6 trục).
- **Quy mô:** Mô hình đầu ra gồm **5 lớp (classes)**.

## 2. Chiến lược Data & Gắn nhãn
Dữ liệu được quy đổi về 5 nhãn chính dựa trên tên file:
- **Fall:** Các file bắt đầu bằng tiền tố 'F'.
- **Walk:** Các file bắt đầu bằng 'D01', 'D02', 'D05', 'D06'.
- **Run:** Các file bắt đầu bằng 'D03', 'D04'.
- **Idle:** Các file chứa '_StandSit_' hoặc '_Lie_'.
- **Trans:** Các file chứa '_Trans_'.
- *Lưu ý:* Chỉ xử lý các cửa sổ dữ liệu có độ dài cố định là 200 mẫu.

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc:** Sử dụng dữ liệu đầu vào dạng windowed (200, 6) gồm 6 features từ cảm biến.
- **Cân bằng dữ liệu:** Tính toán `class_weights` dựa trên phân phối thực tế của tập huấn luyện (`class_weight='balanced'`) để xử lý mất cân bằng dữ liệu.
- **Định dạng:** Nhãn được chuyển sang dạng **One-hot encoding** phục vụ cho quá trình huấn luyện với hàm lỗi Categorical Crossentropy.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** 
    - Kiến trúc **TCN (Temporal Convolutional Network)** với các lớp `Conv1D` sử dụng `dilation_rate` (1, 2, 4, 8) nhằm tăng trường nhìn (receptive field) của mô hình.
    - Tích hợp **SE Block (Squeeze-and-Excitation)** để tăng cường khả năng chú ý (attention) trên các kênh đặc trưng.
    - Kết hợp `GlobalAveragePooling1D` và `GlobalMaxPooling1D` để tận dụng cả thông tin đặc trưng trung bình và cực đại.
- **Kỹ thuật tối ưu:**
    - **Regularization:** Sử dụng `BatchNormalization` sau mỗi lớp Conv và `Dropout` (0.2 - 0.4) để tránh overfitting.
    - **Loss function:** Sử dụng `CategoricalCrossentropy` kết hợp với kỹ thuật **Label Smoothing (0.1)** để cải thiện khả năng tổng quát hóa.
    - **Cấu trúc nhánh:** Sử dụng cơ chế `Residual Connection` (Add) để hỗ trợ hội tụ cho các mạng sâu.