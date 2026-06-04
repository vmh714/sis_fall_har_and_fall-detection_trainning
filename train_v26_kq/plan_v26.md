# Kế Hoạch Triển Khai & Tinh Chỉnh ResNet-1D v26 (Cho AI / GPU Machine)

Tài liệu này đóng gói toàn bộ bối cảnh, quyết định thiết kế và chiến lược của phiên bản `v26` nhằm mục đích chuyển giao cho môi trường huấn luyện GPU. Bất kỳ sự tinh chỉnh nào trong quá trình huấn luyện cũng phải bám sát các ràng buộc phần cứng được nêu dưới đây.

## 1. Bối cảnh & Mục tiêu
- **Kế thừa:** Phiên bản `v25` đã đạt độ chính xác 91%, thời gian suy luận (Inference Time) trên ESP32-S3 chỉ mất **70ms**, và Recall cho lớp Té Ngã (Fall) trên Firmware đạt 99%. 
- **Vấn đề còn tồn đọng:** Lớp `Trans` (chuyển đổi tư thế) có Precision thấp (79%) do thường bị nhầm lẫn qua lại với lớp `Idle` (tĩnh).
- **Mục tiêu v26:** Đẩy độ chính xác của lớp `Trans` và `Idle` lên mức cao nhất có thể **MÀ KHÔNG** làm tăng thời gian suy luận (giữ < 100ms) trên vi điều khiển ESP32-S3.

## 2. Ràng buộc Tối thượng từ Phần cứng (ESP-NN SIMD & INT8)
Khi AI bên máy GPU tiến hành tuning (tinh chỉnh) lại model, **tuyệt đối tuân thủ** các quy tắc sau để không phá vỡ khả năng tăng tốc phần cứng của chip ESP32-S3:
1. **Chỉ dùng Kernel = 3 hoặc 1:** Tập lệnh vector SIMD của ESP-NN tăng tốc gấp 5-14x cho các phép Conv/DepthwiseConv có `kernel=3` và `kernel=1`. Dùng kernel=5 hoặc 7 sẽ khiến ESP32 rớt về chạy C thuần, làm thời gian inference vọt lên 500ms.
2. **Filters phải là bội số của 8:** Cấu trúc mạng hiện tại là `[16, 32, 64, 96]`. Để Quantization INT8 hoạt động trơn tru trên thanh ghi SIMD mà không cần sinh padding, số lượng bộ lọc luôn phải chia hết cho 8.
3. **Tiền xử lý Fixed-Scale (Không dùng Z-Score):** Vi điều khiển không thể dễ dàng nạp Mean/Std động. Cảm biến đã được tiền xử lý cố định bằng việc chia theo giới hạn vật lý: `/ 8.0` cho Accel và `/ 2000.0` cho Gyro. Không thay đổi logic này.

## 3. Các thay đổi cốt lõi đã được Code sẵn trong v26
Hai file `ml_pipeline_v26.py` và `train_v26.py` đã bao gồm các nâng cấp sau:

- **Data Augmentation riêng cho lớp Trans:** Khi tạo tập Cache, tất cả các mẫu thuộc lớp `Trans` sẽ tự động được nhân bản 1 lần với việc thêm hệ số scale ngẫu nhiên từ `[0.9, 1.1]`. Điều này giúp nhân đôi dữ liệu `Trans` và ép mạng NN học hình dáng của sóng thay vì biên độ cố định.
- **Dynamic SE Block Ratio:** Squeeze-and-Excitation được đổi sang dùng ratio động: `ratio = max(2, filters // 8)`. Giúp Bottleneck luôn có tối thiểu 8 nơ-ron, cực kỳ có lợi cho việc giữ thông tin khi lượng tử hóa INT8.
- **Tăng Receptive Field (Block 5):** Thêm 1 Block (strides=1, filters=96) ở cuối mạng (độ phân giải 25 timestep). Việc xếp chồng thêm Block ở độ phân giải thấp giúp tăng Receptive Field mà chỉ tốn thêm 4-6ms.
- **Đổi Loss Function:** Thay vì `CategoricalCrossentropy` thông thường, đã áp dụng **CategoricalFocalCrossentropy (gamma=2.0)** để phạt nặng mạng NN khi nó đoán nhầm các ranh giới khó (Idle vs Trans).

## 4. Hướng dẫn tinh chỉnh (Tuning) cho AI ở máy GPU
Nếu kết quả huấn luyện lần đầu chưa đạt 93-94% Accuracy, AI ở máy GPU hãy thử các phép tinh chỉnh theo thứ tự ưu tiên sau:

1. **Tinh chỉnh Gamma của Focal Loss:** Thử giảm gamma xuống `1.5` hoặc tăng lên `2.5` nếu thấy Loss không hội tụ, hoặc mô hình bắt đầu quên lớp Walk/Run.
2. **Thử Augment thêm cho Idle:** Nếu `Idle` vẫn bị nhầm nhiều sang `Trans`, có thể vào `ml_pipeline_v26.py` (dòng 86) bổ sung logic Jittering (thêm nhiễu Gauss `np.random.normal(0, 0.05)`) vào dữ liệu `Idle`.
3. **Tuning Class Weights:** Thử tắt bỏ `class_weights` trong `model.fit()` để xem Focal Loss khi tự hoạt động một mình có cho ra kết quả phân phối tốt hơn sự kết hợp của cả hai không.

*Ký gửi từ Agent Máy Client - Chúc GPU Agent hoàn thành xuất sắc nhiệm vụ!*
