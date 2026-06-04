# Brainstorming Ý Tưởng Tối Ưu Cho Phiên Bản v26 (Kế thừa từ ResNet-1D v25)

Dựa trên kết quả đánh giá của v25 (Accuracy 91.01%), lớp **Walk** (Recall 88.02%) và lớp **Trans** (Precision 79.18%) vẫn còn có thể cải thiện thêm. Việc mô hình nhầm lẫn giữa các hành động này cho thấy khả năng trích xuất đặc trưng chuỗi thời gian chưa đạt trạng thái hoàn hảo nhất.

Dưới đây là 3 mũi nhọn tối ưu hóa kiến trúc ResNet-1D cho phiên bản `v26` nhằm đẩy Accuracy lên ngưỡng 93-94%:

## 1. Nới rộng "tầm nhìn" (Receptive Field) của lớp Stem
Hiện tại, lớp đầu tiên của v25 đang là:
```python
x = Conv1D(16, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
```
**Vấn đề**: Việc dùng `kernel_size=3` đi kèm `strides=2` ngay ở cổng vào khiến mạng "chớp giật" mất 50% dữ liệu gốc ngay lập tức, và mỗi filter chỉ "nhìn" được có 3 điểm (tương đương 0.06 giây). Với những động tác kéo dài như Walk hay Trans, điều này làm mất các đặc trưng sóng tuần hoàn ngay từ đầu.
**Giải pháp**: Tăng `kernel_size` ở lớp Stem lên `7` (hoặc `5`), giúp mạng "nhìn" được bức tranh chuyển động rộng hơn (0.14 giây) trước khi nén nó lại:
```python
x = Conv1D(16, kernel_size=7, strides=2, padding='same', use_bias=False)(inputs)
```

## 2. Tinh chỉnh lại Squeeze-and-Excitation (SE Block)
Trong `se_block_v25`, hệ số nén đang là `ratio=4`.
**Vấn đề**: Ở các khối ResNet đầu tiên, số lượng Filter chỉ là `16` hoặc `32`. Nếu dùng `ratio=4`, lớp bottleneck của SE Block sẽ bị ép xuống chỉ còn `16 // 4 = 4` nơ-ron. Quá ít nơ-ron ở đây sẽ tạo thành **"nút thắt cổ chai thông tin"**, khiến mạch logic bị đứt gãy.
**Giải pháp**: 
- Đổi `ratio=2` cho toàn mạng để đảm bảo các lớp có Filter thấp (như 16) vẫn giữ được ít nhất 8 nơ-ron để xử lý thông tin sự chú ý (attention).

## 3. Điều chỉnh quy mô màng lọc (Filters)
Cấu hình v25 hiện tại là `[16 -> 32 -> 64]`. Cấu hình này cực kỳ nhẹ (chỉ ~18k tham số). TCN ở `v24` có thể có số tham số lớn hơn nên nó học chi tiết hơn.
**Giải pháp**: Nhích nhẹ số lượng filters lên thành `[24 -> 48 -> 96]` hoặc `[32 -> 64 -> 128]` nếu dung lượng Flash/RAM của vi điều khiển (ESP-NN) vẫn còn dư dả. Việc tăng số lượng bộ lọc giúp mô hình biểu diễn phong phú hơn các ranh giới mong manh giữa `Trans` và `Idle`.

---
*Ghi chú: Nên ưu tiên thử nghiệm Cách 1 + Cách 2 trước, vì nó giúp thông minh hóa mô hình mà gần như không làm tăng kích thước hay độ trễ (latency). Nếu vẫn muốn ép giới hạn cao hơn, ta mới áp dụng Cách 3.*
