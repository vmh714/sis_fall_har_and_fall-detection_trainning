# Tài liệu Instruction: Phiên bản TCN-MCU Human Activity Recognition

## 1. Tổng quan Model
- **Mục đích:** Phân loại hành động người dùng dựa trên dữ liệu cảm biến IMU (cảm biến chuyển động).
- **Quy mô:** Hỗ trợ 4 lớp đầu ra (n_classes=4).

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc:** Sử dụng các file CSV được cắt thành các đoạn (window) có độ dài cố định là 200 mẫu (samples).
- **Phân chia dữ liệu:** Dữ liệu được chia theo `subject_id` (ID người thực hiện) thành tập Huấn luyện (Train), Kiểm chứng (Val) và Kiểm thử (Test) để đảm bảo tính độc lập giữa các tập.
- **Ánh xạ:** Sử dụng từ điển `LABEL_MAP` (dựa trên tiền tố của tên file) để quy đổi nhãn văn bản sang nhãn số nguyên.

## 3. Tiền xử lý (Preprocessing)
- **Cắt gọt:** Dữ liệu đầu vào bắt buộc phải có chiều dài chính xác là 200 samples/window. Các dữ liệu không thỏa mãn điều kiện này sẽ bị loại bỏ trong quá trình nạp.
- **Định dạng:** Dữ liệu sau khi xử lý được cache dưới định dạng `.npy` để tăng tốc độ nạp cho các lần huấn luyện sau.
- **Chuẩn hóa:** Áp dụng `BatchNormalization` ngay lớp đầu vào của mô hình để ổn định phân phối dữ liệu (thay vì tiền xử lý thủ công trên file gốc).

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Temporal Convolutional Network (TCN) kết hợp kỹ thuật `Causal Conv1D` để xử lý dữ liệu chuỗi thời gian.
- **Cấu trúc lớp:** 
    - 2 chồng (stacks) convolution với các hệ số giãn nở (dilation rates: 1, 2, 4, 8) nhằm tăng trường nhìn (receptive field).
    - Sử dụng `Residual connection` (kết nối tắt) để tránh triệt tiêu gradient.
    - `GlobalAveragePooling1D` được dùng để nén đặc trưng trước khi đi vào lớp `Dense` đầu ra.
- **Kỹ thuật tối ưu:**
    - **Class weights:** Sử dụng `balanced` class weights để xử lý vấn đề mất cân bằng dữ liệu giữa các lớp.
    - **Regularization:** Sử dụng `Dropout` (0.2 - 0.3) và `BatchNormalization` xuyên suốt mô hình.
    - **Hậu xử lý:** Hỗ trợ chuyển đổi sang định dạng `TFLite (int8)` để tối ưu hóa việc triển khai trên vi điều khiển (MCU).