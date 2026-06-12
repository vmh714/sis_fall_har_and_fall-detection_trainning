# Phân tích khả năng triển khai firmware & tối ưu ESP-NN

Tài liệu này so sánh các phiên bản mô hình theo góc độ **tương thích ESP-NN** và **hiệu năng suy luận trên MCU (ESP32-S3)**.

---

## 1. Nền tảng: ESP-NN là gì?

**ESP-NN** là thư viện tăng tốc suy luận mạng nơ-ron của Espressif, khai thác tập lệnh **SIMD (Single Instruction, Multiple Data)** của Xtensa LX7 trên ESP32-S3.

Điều kiện để ESP-NN tăng tốc một lớp:

| Điều kiện | Lý do |
|-----------|-------|
| Dùng `SeparableConv1D` (depthwise + pointwise) | Tách phép tính → vectorize độc lập từng bước |
| `kernel_size` = 3 (bội số SIMD width) | Load/compute align với thanh ghi 128-bit |
| Số filter là bội số 8 hoặc 16 | Tránh padding/masking khi xử lý theo chunk |
| Activation là `ReLU6` | Clamp cứng → không cần branch, thân thiện INT8 |
| Không có dilation (`dilation_rate > 1`) | Dilated conv không có kernel ESP-NN → fallback CPU |

---

## 2. So sánh kiến trúc theo mức độ tối ưu ESP-NN

### v22 — TCN dilated (không dùng được ESP-NN)

**Kiến trúc:** TCN với dilated causal convolution, 2 stack × 4 lớp, dilation rate [1, 2, 4, 8], kernel_size=7, 32 filters.

**Vấn đề với ESP-NN:**
- `dilation_rate > 1` → **không có kernel ESP-NN** cho dilated conv → toàn bộ mô hình chạy trên generic CPU kernel
- `kernel_size=7` → không align tốt với SIMD width 128-bit
- Dùng `Conv1D` thông thường (không phải SeparableConv) → mất lợi thế depthwise 6.3×

**Kết quả:** Mô hình có hiệu năng **tốt nhất trên PC** (Trans F1 = 0.8554) nhưng khi triển khai lên firmware sẽ chạy hoàn toàn trên CPU, ước tính chậm hơn v25 nhiều lần. Chưa có kết quả firmware thực tế.

| Chỉ số | Giá trị |
|--------|---------|
| Acc (PC) | 91.56% |
| F1-Fall | 0.9786 |
| F1-Trans | **0.8554** |
| Macro-F1 | 0.9267 |
| Model C array | ~700 KB |
| Inference (firmware) | Chưa đo — ước tính >300ms |

---

### v23 — ResNet-1D Conv1D, stride-based (tối ưu một phần)

**Kiến trúc:** Loại bỏ hoàn toàn dilated conv. Thay bằng MaxPooling1D + 4 residual block Conv1D với strides [2,1,2,1], kernel_size=7, 32 filters, SE block.

**Cải tiến so với v22:**
- Không còn dilation → các lớp Conv1D có thể dùng kernel ESP-NN tiêu chuẩn
- Thêm `ReLU6` (quantization-friendly)

**Hạn chế còn lại:**
- Vẫn dùng `Conv1D` thông thường thay vì `SeparableConv1D` → bỏ lỡ tăng tốc depthwise ×6.3
- `kernel_size=7` → không phải kích thước tối ưu cho SIMD (k=3 tốt hơn)
- 32 filters → bội số 8 nhưng không phải 16 → tận dụng SIMD chưa đầy đủ

**Ghi chú lịch sử:** Bản firmware v23 đầu tiên ghi nhận accuracy **54.17%** — đây là kết quả của **bug preprocessing**: firmware nhận raw sensor data chưa normalize (accelerometer ±16g thay vì clip ±8g rồi chia 8.0), khiến tensor đầu vào INT8 bị bão hòa hoàn toàn. Đây **không phải** hiệu năng thực của kiến trúc.

| Chỉ số | Giá trị |
|--------|---------|
| Acc (PC) | 91.33% |
| F1-Fall | 0.9832 |
| F1-Trans | 0.8444 |
| Macro-F1 | 0.9273 |
| Model C array | ~638 KB |
| Acc (firmware, sau fix) | 90.33% |
| Fall Recall (firmware) | 99.00% |
| Inference (firmware) | Chưa đo riêng |

---

### v25 — ResNet-1D SeparableConv (tối ưu hoàn toàn cho ESP-NN)

**Kiến trúc:** Thiết kế lại toàn bộ với mục tiêu tối đa hóa ESP-NN:
- Stem: `Conv1D(16, k=3, stride=2)` — lớp đầu dùng standard conv để học đặc trưng thô
- Block 1–4: `SeparableConv1D(k=3) × 2 + SE Block (Conv1D 1×1) + Residual`
- Filters: [16, 32, 32, 64] — toàn bộ là bội số 16
- Activation: `ReLU6` xuyên suốt
- Head: `GAP + GMP → Concat → Dropout → Dense(5)`
- Tổng params: **19,253** (18,453 trainable)

**Tất cả điều kiện ESP-NN đều đạt:**

| Điều kiện ESP-NN | v25 |
|-----------------|-----|
| SeparableConv1D | ✅ (depthwise ×6.3) |
| kernel_size = 3 | ✅ |
| Filters bội số 16 | ✅ (16, 32, 64) |
| ReLU6 | ✅ |
| Không dilation | ✅ |
| Không Lambda layer | ✅ (TFLite-clean) |

| Chỉ số | Giá trị |
|--------|---------|
| Acc (PC) | 91.01% |
| F1-Fall | 0.9784 |
| F1-Trans | 0.8365 |
| Macro-F1 | 0.9238 |
| Model C array | ~504 KB |
| Tensor arena thực dùng | **28.2 KB** |
| Acc (firmware) | **90.33%** |
| Fall Recall (firmware) | **99.00%** |
| Inference trung bình | **70.23 ms** |
| Inference min/max | 70.07 / 70.32 ms |

Gap desktop→firmware chỉ **−0.68%** → pipeline INT8 ổn định.

---

### v27 — CNN-LSTM baseline (CPU thuần, không ESP-NN)

**Kiến trúc:** Giữ nguyên kiến trúc v1 (baseline), huấn luyện lại trên tập 5 nhãn.
`Conv1D(64)×2 → MaxPool → LSTM(128, return_seq) → LSTM(64) → Dense(32) → Dense(5)`

**Lý do không tận dụng ESP-NN:**
- `LSTM` sử dụng op `UnidirectionalSequenceLSTM` trong TFLite — không có kernel ESP-NN tương ứng
- 100 timestep × gate operations phải tính **tuần tự** trên CPU
- `Conv1D(64)` với k=3 được ESP-NN tăng tốc nhưng chiếm tỷ trọng nhỏ trong tổng thời gian

**Trạng thái:** Đã viết script, chưa train và chưa deploy firmware. Inference time ước tính **300–600ms** dựa trên kích thước LSTM.

> **Mục đích:** Baseline so sánh kiến trúc cổ điển vs MCU-aware trên cùng tập dữ liệu 5 nhãn.

---

## 3. Bảng tổng hợp so sánh

### 3.1 Kết quả trên PC (test set 9,135 mẫu)

| Ver | Kiến trúc | #Nhãn | Acc% | F1-Fall | F1-Trans | Macro-F1 |
|-----|-----------|-------|------|---------|----------|----------|
| v22 | TCN dilated (SE + DualPool) | 5 | 91.56 | 0.9786 | **0.8554** | 0.9267 |
| v23 | ResNet-1D (Conv1D, k=7, stride) | 5 | 91.33 | **0.9832** | 0.8444 | 0.9273 |
| v25 | ResNet-1D (SepConv, k=3) | 5 | 91.01 | 0.9784 | 0.8365 | 0.9238 |
| v27 | CNN-LSTM baseline | 5 | TBD | TBD | TBD | TBD |

### 3.2 Kết quả trên firmware ESP32-S3

| Ver | #Mẫu test | Acc% | F1-Fall | F1-Trans | Macro-F1 | Fall Recall | Inf. time |
|-----|-----------|------|---------|----------|----------|------------|-----------|
| v23* | 600 | 54.17 | 0.6620 | 0.4312 | 0.4025 | 95.00% | — |
| v24† | 1,200 | 90.33 | 0.9950 | 0.8374 | 0.9132 | **99.00%** | — |
| v25 | 1,200 | **90.33** | **0.9950** | **0.8374** | **0.9132** | **99.00%** | **70.23ms** |

> \* v23 firmware = kết quả **trước khi fix bug preprocessing** (INT8 input saturation), không phản ánh hiệu năng kiến trúc.  
> † v24 và v25 cho kết quả giống hệt — cùng file `inference_results.csv` (1,200 mẫu); bản chạy thực trên board là v25.

### 3.3 Mức độ tối ưu ESP-NN

| Ver | Kiến trúc | Dilation | SepConv | k=3 | ReLU6 | ESP-NN level |
|-----|-----------|----------|---------|-----|-------|-------------|
| v22 | TCN | ✅ (có) | ✗ | ✗ | ✗ | **Không tương thích** |
| v23 | ResNet Conv1D | ✗ | ✗ | ✗ (k=7) | ✅ | **Tối ưu một phần** |
| v25 | ResNet SepConv | ✗ | ✅ | ✅ | ✅ | **Tối ưu hoàn toàn** |
| v27 | CNN-LSTM | ✗ | ✗ | ✅ (conv) | ✗ | **CPU thuần (LSTM)** |

---

## 4. Kết luận

Quá trình phát triển thể hiện rõ 3 giai đoạn tối ưu phần cứng:

1. **v22 (TCN):** Hiệu năng tốt nhất trên PC nhưng kiến trúc dilated conv **không tương thích** với ESP-NN → không thể tận dụng SIMD trên ESP32-S3.

2. **v23 (stride-ResNet):** Loại bỏ dilation, thoát khỏi TCN paradigm. Conv1D tiêu chuẩn được ESP-NN hỗ trợ nhưng chưa đạt tối ưu: kernel=7 và thiếu SeparableConv → bỏ lỡ tăng tốc depthwise ×6.3.

3. **v25 (SepConv-ResNet):** Toàn bộ kiến trúc được thiết kế quanh tập lệnh ESP-NN — SeparableConv1D, kernel=3, filters bội số 16, ReLU6. Kết quả: inference **70ms**, tensor arena **28 KB**, Fall Recall **99%** trên firmware, gap desktop→firmware chỉ 0.68%.

v27 (CNN-LSTM) bổ sung góc nhìn baseline: kiến trúc cổ điển không MCU-aware, LSTM chạy hoàn toàn trên CPU generic kernel, inference dự kiến chậm hơn v25 khoảng 4–8× để đổi lấy khả năng so sánh cross-architecture trên cùng tập dữ liệu 5 nhãn.
