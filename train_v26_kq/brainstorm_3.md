PROMPT1:
REQUEST: các file đính kèm gồm report_v25, report_v25_firmware, confusion matrix v25 (2 phiên bản), source code train_v25.py, README.md của esp-nn
tôi đang muốn cải thiện model để cải thiện độ chính xác của 2 nhãn trans and idle, tôi đã add brainstorm và source code của v25, hãy đề xuất hướng cải tiến mà vẫn giữ nguyên độ chính xác và tốc độ của mô hình cũ, chỉ brainstorm thôi, không cần tạo file code. Đặc biệt đề xuất kiến trúc phù hợp với thư viện tăng tốc phần cứng của esp nn , không cần lo về dung lượng bộ nhớ nhưng phải đảm bảo tốc độ suy luận của mô hình dưới 100 ms như cách v25 đã làm (70ms)
RESPONSE:
# Brainstorm Cải Thiện v26: Tối Ưu Idle/Trans

Dựa trên phân tích toàn bộ tài liệu, đây là các hướng cải tiến ưu tiên theo thứ tự tác động:

---

## Vấn đề cốt lõi cần giải quyết

Nhìn vào report v25, `Trans` đang có **Precision thấp nhất (79.18%)** và `Walk` có **Recall thấp nhất (88.02%)**. Nguyên nhân chính là mô hình chưa bắt được đủ **ngữ cảnh thời gian dài** để phân biệt ranh giới mờ giữa Idle↔Trans↔Walk. Tất cả các hướng dưới đây đều nhắm vào điểm này mà **không vi phạm ràng buộc kernel=3 của ESP-NN**.

---

## Hướng 1: Tăng chiều sâu mạng tại tầng cuối (Ưu tiên cao nhất)

**Vấn đề:** Sau 4 ResNet Block, chuỗi thời gian đã bị downsample từ 200 → 25 mẫu. Chỉ có 1 khối xử lý ở độ phân giải thấp nhất này, nghĩa là mô hình chỉ có một "cái nhìn" tổng hợp đặc trưng bậc cao.

**Đề xuất:** Thêm **Block 5** ngay sau Block 4, giữ `strides=1` và tăng filters lên 96. Stack 2 SeparableConv(kernel=3) liên tiếp ở độ phân giải 25 sẽ cho Receptive Field tương đương kernel=7 ở tầng gốc, nhưng vẫn được ESP-NN tăng tốc tối đa.

```
Block 4: 50x32 → 25x64  (như cũ)
Block 5: 25x64 → 25x96  (thêm mới, strides=1)  ← điểm mấu chốt
```

**Ước tính tác động tốc độ:** Ở độ phân giải 25 timestep, phép tích chập rất nhẹ. Thêm 1 block ở đây chỉ tốn thêm ước tính 3–5ms, giữ tổng trong 75–80ms.

---

## Hướng 2: Sửa lỗi SE Block ratio cứng

**Vấn đề đã được xác nhận trong brainstorm.md:** Với `ratio=4` cố định, SE Block ở Block 1 (16 filters) bị nén xuống chỉ còn **4 nơ-ron** — quá ít để học attention weight có ý nghĩa.

**Đề xuất:** Truyền `ratio` động theo số filters:

| Block | Filters | ratio hiện tại | ratio đề xuất | Bottleneck nơ-ron |
|-------|---------|---------------|---------------|-------------------|
| 1 | 16 | 4 | **2** | 4 → **8** |
| 2 | 32 | 4 | **2** | 8 → **16** |
| 3 | 32 | 4 | **2** | 8 → **16** |
| 4 | 64 | 4 | 4 | 16 (giữ nguyên) |
| 5 (mới) | 96 | — | 4 | 24 |

Thay đổi này **không tốn thêm thời gian suy luận đáng kể** vì chỉ thêm vài nơ-ron ở Conv1D(1x1) — phép toán được ESP-NN tối ưu 14x.

---

## Hướng 3: Focal Loss thay thế CategoricalCrossentropy

**Vấn đề:** `CategoricalCrossentropy(label_smoothing=0.1)` phạt đều mọi lỗi. Mô hình đang "lười" sửa sai ở các case khó (Idle giả Trans) vì tổng loss vẫn thấp nhờ phân loại tốt các lớp dễ như Fall, Run.

**Đề xuất:** Áp dụng **Focal Loss với gamma=2.0, alpha cân bằng theo class weight**:

```
FL(p) = -alpha * (1 - p)^gamma * log(p)
```

Khi mô hình dự đoán sai với confidence cao (nhầm Idle thành Trans chắc chắn), `(1-p)^gamma` sẽ khuếch đại penalty lên gấp nhiều lần. Đây là thay đổi **thuần túy ở training**, không ảnh hưởng tốc độ inference trên firmware.

---

## Hướng 4: Chuẩn hóa độc lập Accel/Gyro

**Vấn đề:** Biên độ Accelerometer (~9.8 m/s² trọng lực) lớn hơn nhiều so với Gyroscope (~rad/s). Nếu Z-score toàn bộ 6 trục cùng nhau, tín hiệu Gyro bị "lu mờ". Nhưng chính **Gyro mới là chìa khóa** phân biệt Trans (vận tốc góc thay đổi mạnh) vs Idle (Gyro ≈ 0).

**Đề xuất trong DataPreprocessor:** Chuẩn hóa Z-score **riêng** cho 3 trục Accel (index 0–2) và **riêng** cho 3 trục Gyro (index 3–5). Đây là thay đổi ở tầng tiền xử lý, hoàn toàn không ảnh hưởng tốc độ inference (chuẩn hóa test sample cũng được thực hiện trước khi đưa vào mô hình).

---

## Hướng 5: Tăng Overlap cửa sổ cho lớp Trans khi tạo dataset

Như đã phân tích ở brainstorm.txt, với overlap 50% (step=100), hành động Trans ngắn (~1–1.5s) dễ bị cắt đôi. 

**Đề xuất:** Khi chạy DataPreprocessor, sinh thêm mẫu Trans bằng **step=50 (overlap 75%)** riêng cho nhãn Trans và Walk, trong khi Idle và Run vẫn dùng step=100. Điều này giải quyết đồng thời vấn đề mất cân bằng (Trans hiện chỉ có 1480 mẫu so với 3674 của Idle) và giúp mô hình thấy Trans xuất hiện ở nhiều vị trí khác nhau trong cửa sổ 2 giây.

---

## Tổng kết và thứ tự ưu tiên thử nghiệm

| Thứ tự | Hướng | Tác động dự kiến | Rủi ro tốc độ |
|--------|-------|-----------------|---------------|
| **1** | Sửa SE Block ratio động | Trung bình–Cao | Không có |
| **2** | Focal Loss | Cao | Không có |
| **3** | Chuẩn hóa Accel/Gyro độc lập | Trung bình | Không có |
| **4** | Tăng Overlap cho Trans/Walk | Cao | Không có |
| **5** | Thêm Block 5 (25x96) | Trung bình | +3–5ms |

**Khuyến nghị:** Nên thử **Hướng 1+2+3+4 trước** (không đụng đến kiến trúc, chỉ sửa training pipeline và preprocessing). Nếu vẫn chưa đạt mục tiêu thì mới thêm Block 5. Đây là cách tiếp cận kiểm soát tốt nhất vì mỗi thay đổi đều có thể đo lường tác động độc lập.

PROMPT2:
REQUEST:
data tôi đã chuẩn hóa về -1;1 cho cả accel và gyro, hướng 5 tôi cũng đã làm với dataset của mình, chắc bạn chỉ cần đóng gói 3 hướng đầu và nhớ rằng phải tôi ưu bằng các chạy INT8 với kiến trúc tôi ưu cho tập lệnh của esp nn
RESPONSE:
# Brainstorm v26: 3 Hướng Cải Tiến Tối Ưu ESP-NN INT8

---

## Bối cảnh quan trọng: Ràng buộc INT8 + ESP-NN

Trước khi đi vào từng hướng, cần nhắc lại một nguyên tắc thiết kế cốt lõi khi nhắm tới **Post-Training Quantization INT8** trên ESP-NN:

- **Tránh hoàn toàn** các phép toán không được ESP-NN tối ưu cho INT8: `LayerNormalization`, `MultiHeadAttention`, `LSTM`, `GRU` — các lớp này sẽ fallback về ANSI C và làm tăng latency đột biến
- **Ưu tiên tuyệt đối** các phép toán nằm trong bảng tối ưu của ESP-NN: `Conv1D (kernel=1, kernel=3)`, `SeparableConv1D (DepthwiseConv + Pointwise)`, `ReLU6`, `GlobalAveragePooling`, `Dense`
- **ReLU6** thay vì ReLU thông thường là lựa chọn đúng — nó clip giá trị trong `[0, 6]`, giúp quantization INT8 không bị mất thông tin ở đuôi phân phối

---

## Hướng 1: Sửa SE Block — Ratio động theo Filter depth

### Vấn đề
`se_block_v25` đang hardcode `ratio=4` cho tất cả các block. Ở Block 1 với 16 filters, bottleneck bị ép xuống còn **4 nơ-ron** — quá nhỏ để học attention weight có nghĩa. Nhưng vấn đề với INT8 còn sâu hơn: bottleneck càng nhỏ thì **quantization error** càng lớn vì khoảng giá trị được biểu diễn quá hẹp.

### Giải pháp
Dùng công thức `ratio` động đảm bảo bottleneck luôn có **tối thiểu 8 nơ-ron**:

```
ratio = max(2, filters // 8)
```

Kết quả theo từng block:

| Block | Filters | ratio cũ | ratio mới | Bottleneck cũ | Bottleneck mới |
|-------|---------|----------|----------|---------------|----------------|
| 1 | 16 | 4 | **2** | 4 nơ-ron | **8 nơ-ron** |
| 2, 3 | 32 | 4 | **4** | 8 nơ-ron | **8 nơ-ron** |
| 4 | 64 | 4 | **8** | 16 nơ-ron | **8 nơ-ron** |

> **Lưu ý INT8:** Conv1D(1x1) — tức pointwise — là phép toán được ESP-NN tăng tốc **14.24x** trên S3. Tăng nhẹ bottleneck từ 4 lên 8 nơ-ron ở Block 1 gần như không tác động đến latency.

---

## Hướng 2: Focal Loss — Ép mô hình học ranh giới khó

### Vấn đề
`CategoricalCrossentropy(label_smoothing=0.1)` phạt đồng đều mọi lỗi. Khi Fall và Run đã phân loại gần hoàn hảo, gradient từ các lớp dễ này **át đi** gradient học từ Idle/Trans. Mô hình hội tụ sớm nhưng không chịu "cố" thêm ở các ranh giới khó.

### Giải pháp
Thay bằng **Focal Loss (gamma=2.0)** kết hợp giữ nguyên `class_weight` như v25:

```
FL(p_t) = -(1 - p_t)^gamma * log(p_t)
```

Khi mô hình dự đoán sai với confidence cao (ví dụ: nhầm Idle→Trans với p=0.85), factor `(1-0.85)^2 = 0.0225` khuếch đại penalty lên gấp nhiều lần so với cross-entropy thông thường. Ngược lại, khi mô hình đã chắc chắn đúng (Fall với p=0.99), factor `(1-0.99)^2 ≈ 0` — loss gần bằng 0, không lãng phí gradient.

> **Lưu ý:** Focal Loss là thay đổi thuần túy ở training. Không ảnh hưởng bất kỳ gì đến kiến trúc và latency firmware.

**Điểm cần thực nghiệm:** Thử `gamma` trong khoảng `[1.5, 2.0, 2.5]`. Gamma quá cao sẽ khiến mô hình bỏ qua các mẫu dễ quá sớm và mất ổn định training.

---

## Hướng 3: Thêm Block 5 — Mở rộng Receptive Field tại tầng thấp nhất

### Vấn đề
Sau 4 block, chuỗi đã được downsample về 25 timestep. Chỉ có **một lần** mô hình xử lý đặc trưng bậc cao ở độ phân giải này, tức là chỉ có một "cái nhìn" tổng hợp cuối cùng trước GAP/GMP.

Stack hai `SeparableConv(kernel=3)` liên tiếp ở cùng độ phân giải 25 tương đương **Receptive Field = kernel 7** ở tầng gốc — nhưng được ESP-NN tối ưu đầy đủ thay vì dùng kernel=7 thực sự bị chậm.

### Giải pháp
Thêm Block 5 ngay sau Block 4, `strides=1`, tăng filters lên 96:

```
Block 4: 50x32 → 25x64   (giữ nguyên)
Block 5: 25x64 → 25x96   (thêm mới)  ← Receptive Field tăng gấp đôi
GAP + GMP → Concat(192)  (thay vì 128)
```

### Ước tính tác động latency

Ở độ phân giải 25 timestep, tổng số phép nhân tích chập của cả Block 5 rất nhỏ. Tham khảo bảng ESP-NN: DepthwiseConv trên S3 tăng tốc **6.3x**, pointwise **14.24x**. Ước tính block này chỉ thêm **4–6ms**, giữ tổng latency trong khoảng **75–80ms** — vẫn dưới ngưỡng 100ms.

> **Lưu ý INT8 quan trọng:** Khi tăng filters từ 64 lên 96, cần đảm bảo 96 chia hết cho 8 (✓) để ESP-NN SIMD xử lý trọn vẹn vector 8 phần tử INT8 mà không cần padding. Tương tự, tất cả filter counts trong toàn bộ mạng `[16, 32, 64, 96]` đều đã là bội số của 8 — đây là điều kiện cần để tránh overhead alignment khi quantize.

---

## Thứ tự thực hiện đề xuất

```
Hướng 1 + 2  →  train + eval  →  nếu chưa đủ  →  thêm Hướng 3
(không đụng kiến trúc)                           (mới mở rộng model)
```

Tách hai giai đoạn này giúp xác định chính xác phần nào mang lại cải thiện, tránh thay đổi đồng thời khiến việc debug kết quả trở nên khó kiểm soát.
