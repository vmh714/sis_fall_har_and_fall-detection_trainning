# Tài liệu Instruction: Phiên bản TCN_MCU_HumanActivityRecognition

## 1. Tổng quan Model
- **Mục đích:** Nhận diện hoạt động dựa trên dữ liệu cảm biến IMU (thời gian thực).
- **Quy mô:** Thiết kế mô hình với 4 lớp đầu ra (`n_classes=4`), tập trung vào sự tối ưu hóa cho các thiết bị nhúng (MCU) thông qua định dạng TFLite Micro và chuẩn bị cho Full INT8 Quantization.

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc dữ liệu:** Dữ liệu đầu vào là các "window" cố định có độ dài 200 mẫu (samples/window).
- **Phân bổ:** Dữ liệu được chia theo `subject_id` (người thực hiện) vào 3 tập riêng biệt: `Train`, `Val`, và `Test` để đảm bảo tính tổng quát hóa.
- **Xử lý nhãn:** Sử dụng `LABEL_MAP` để ánh xạ từ mã nhãn trong tên file (định dạng `LabelCode_SubjectID`) sang giá trị số nguyên.

## 3. Tiền xử lý (Preprocessing)
- **Cắt gọt (Clipped):** Dữ liệu được lọc theo độ dài cố định 200 mẫu/file. Các file không đạt chuẩn bị loại bỏ (`len(df) != 200`).
- **Scale:** Dữ liệu được tải dưới dạng mảng `float32`. Mặc dù code không hiện rõ thông số scale cụ thể, mô hình thực hiện `BatchNormalization` ngay tại lớp đầu vào để chuẩn hóa phân phối dữ liệu (trung bình và phương sai) trước khi đưa vào các khối tích chập.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Temporal Convolutional Network (TCN) bao gồm 2 stacks, mỗi stack chứa 4 khối `Conv1D` với các tốc độ giãn nở (dilation rates) khác nhau [1, 2, 4, 8].
- **Cấu trúc chi tiết:**
    - Sử dụng `Padding='causal'` đảm bảo tính nhân quả cho dữ liệu chuỗi thời gian.
    - Kết nối tắt (Residual connections) được áp dụng để tránh suy giảm gradient.
    - Sử dụng `GlobalAveragePooling1D` để nén đặc trưng trước lớp phân loại cuối cùng.
- **Kỹ thuật tối ưu:**
    - **Cân bằng lớp:** Sử dụng `class_weight='balanced'` để xử lý tình trạng mất cân bằng dữ liệu đầu vào.
    - **Điều chuẩn:** Áp dụng `Dropout(0.2)` trong các block TCN và `Dropout(0.3)` sau lớp pooling.
    - **Định dạng triển khai:** Mô hình được tối ưu hóa cho **TFLite INT8 Quantization**, giúp giảm kích thước và tăng tốc độ suy luận trên vi điều khiển.