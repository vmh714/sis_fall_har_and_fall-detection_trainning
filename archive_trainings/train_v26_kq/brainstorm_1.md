# Brainstorming Ý Tưởng Tối Ưu Cho Phiên Bản v26 (Kế thừa từ ResNet-1D v25)

Dựa trên kết quả đánh giá của v25 (Accuracy 91.01%), lớp **Walk** (Recall 88.02%) và lớp **Trans** (Precision 79.18%) vẫn còn có thể cải thiện thêm. Việc mô hình nhầm lẫn giữa các hành động này cho thấy khả năng trích xuất đặc trưng chuỗi thời gian chưa đạt trạng thái hoàn hảo nhất.

Tuy nhiên, như đã phân tích về tập lệnh **SIMD của ESP32-S3 (ESP-NN)**, các phép toán tích chập (Conv/DepthwiseConv) được tối ưu cực kỳ mạnh mẽ (tốc độ tăng từ 5x đến 14x) khi sử dụng **kernel_size = 3**. Do đó, chúng ta tuyệt đối **không được thay đổi kernel_size sang 5 hay 7** để tránh phá vỡ tối ưu phần cứng. Thay vào đó, dưới đây là 3 mũi nhọn tối ưu hóa kiến trúc ResNet-1D cho phiên bản `v26` nhằm đẩy Accuracy lên ngưỡng 93-94% mà vẫn giữ trọn vẹn tốc độ 70ms:

## 1. Nới rộng Receptive Field bằng cách chồng thêm các lớp Conv(kernel=3)
Thay vì dùng 1 lớp `kernel=7` (vừa làm chậm vi điều khiển, vừa không tận dụng được ESP-NN), chúng ta có thể nới rộng "tầm nhìn" của mô hình bằng cách **xếp chồng liên tiếp các lớp có `kernel=3`**.
**Lý thuyết:** Một stack gồm hai lớp Conv1D `kernel=3` sẽ có Receptive Field tương đương một lớp `kernel=5`, ba lớp `kernel=3` sẽ tương đương `kernel=7`, nhưng số lượng tham số ít hơn và vi điều khiển chạy nhanh hơn gấp nhiều lần nhờ ESP-NN.
**Giải pháp**: Giữ nguyên lớp Stem `kernel=3` nhưng có thể thêm 1 lớp Conv1D(kernel=3) ngay sau Stem trước khi vào các khối ResNet Block, hoặc tăng số lượng ResNet Block lên 1 bậc (thêm 1 Block) để mô hình có thể bắt được Context dài hơn của `Trans` (chuyển đổi tư thế).

## 2. Tinh chỉnh lại Squeeze-and-Excitation (SE Block)
Trong `se_block_v25`, hệ số nén đang là `ratio=4`.
**Vấn đề**: Ở các khối ResNet đầu tiên, số lượng Filter chỉ là `16` hoặc `32`. Nếu dùng `ratio=4`, lớp bottleneck của SE Block sẽ bị ép xuống chỉ còn `16 // 4 = 4` nơ-ron. Quá ít nơ-ron ở đây sẽ tạo thành **"nút thắt cổ chai thông tin"**, khiến mạch logic bị đứt gãy.
**Giải pháp**: 
- Đổi `ratio=2` cho các Block đầu để đảm bảo các lớp có Filter thấp (như 16) vẫn giữ được ít nhất 8 nơ-ron để xử lý thông tin sự chú ý (attention). Hoặc có thể áp dụng SE Block chọn lọc ở các Block sâu (có 64 filters) thay vì đặt ở mọi Block.

## 3. Điều chỉnh quy mô màng lọc (Filters) một cách thận trọng
Cấu hình v25 hiện tại là `[16 -> 32 -> 64]`. Cấu hình này cực kỳ nhẹ.
**Giải pháp**: Nhích nhẹ số lượng filters lên thành `[24 -> 48 -> 96]` nếu dung lượng Flash/RAM của vi điều khiển (ESP-NN) vẫn còn dư dả. Việc tăng số lượng bộ lọc giúp mô hình biểu diễn phong phú hơn các ranh giới mong manh giữa `Trans` và `Idle`.

---
*Ghi chú: Nên ưu tiên thử nghiệm Cách 1 + Cách 2 trước, vì nó giúp thông minh hóa mô hình mà tận dụng tối đa sức mạnh của SIMD kernel=3 trên ESP32-S3. Bằng cách xếp chồng kernel=3, tốc độ nhúng (latency) sẽ gần như không thay đổi quá nhiều so với bản v25.*
