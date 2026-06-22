# Tài liệu Instruction: Phiên bản TCN-MCU (Temporal Convolutional Network for Microcontrollers)

## 1. Tổng quan Model
- **Mục đích:** Nhận diện hoạt động dựa trên dữ liệu cảm biến IMU (cảm biến gia tốc và con quay hồi chuyển) trên thiết bị nhúng.
- **Quy mô:** Thiết kế cho bài toán phân loại đa lớp (mặc định `n_classes=4`), tối ưu hóa để chuyển đổi sang **TFLite Micro** với định dạng **Full INT8 Quantization**.

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc dữ liệu:** Dữ liệu được chia theo cửa sổ (windowed) với độ dài cố định là **200 mẫu (samples)** mỗi file.
- **Gán nhãn:** Nhãn được trích xuất trực tiếp từ tên file thông qua tiền tố (`label_code`) và được ánh xạ qua từ điển `LABEL_MAP`.
- **Phân chia tập:** Dữ liệu được chia tập Train/Val/Test dựa trên định danh người dùng (`subject_id`) để đảm bảo tính độc lập giữa các tập dữ liệu.
- **Xử lý mất cân bằng:** Sử dụng kỹ thuật `class_weight='balanced'` để tính toán trọng số, đặc biệt ưu tiên/phạt lỗi cho các lớp hiếm (như Fall - té ngã).

## 3. Tiền xử lý (Preprocessing)
- **Định dạng dữ liệu:** Input được chuẩn hóa về dạng `(200, features)`.
- **Đặc điểm kỹ thuật:**
    - Sử dụng `BatchNormalization` ngay tại lớp đầu vào để ổn định phân phối dữ liệu (giảm ảnh hưởng của việc lệch thang đo giữa các cảm biến).
    - Dữ liệu được lưu trữ dạng cache `.npy` để tối ưu hóa quy trình training và đảm bảo tính nhất quán của dữ liệu đầu vào giữa các phiên chạy.

## 4. Kiến trúc mô hình (Architecture)
- **Kiến trúc chính:** **Temporal Convolutional Network (TCN)** với 2 stacks, mỗi stack gồm 4 khối (tổng cộng 8 lớp tích chập).
- **Các thành phần kỹ thuật:**
    - **Dilated Conv1D:** Sử dụng `dilation_rate` lần lượt [1, 2, 4, 8] để mở rộng trường nhìn (receptive field) mà không làm tăng tham số đáng kể.
    - **Kết nối tắt (Residual Connection):** Sử dụng các nhánh cộng (Add) để tránh vấn đề triệt tiêu đạo hàm trong mạng sâu.
    - **Regularization:** Kết hợp `Dropout (0.2 - 0.3)` và `BatchNormalization` sau mỗi khối để chống quá khớp (overfitting).
    - **Pooling:** Sử dụng `GlobalAveragePooling1D` để nén đặc trưng trước khi đưa vào lớp phân loại `Dense`.
    - **Tối ưu hóa:** Sử dụng bộ tối ưu `Adam` với learning rate `1e-3` và hàm mất mát `sparse_categorical_crossentropy`.
    - **Triển khai:** Hỗ trợ xuất mô hình sang TFLite với cơ chế `Full INT8 Quantization` để tối ưu cho chip MCU.