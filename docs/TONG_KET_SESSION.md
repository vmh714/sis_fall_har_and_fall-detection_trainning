# Tổng kết phiên làm việc — SisFall Fall/HAR tinyML (ESP32-S3)

> Mục tiêu phiên: gỡ nút thắt **Idle↔Trans nhầm lẫn** kéo accuracy xuống, và tối ưu mạng cho **deploy INT8 trên ESP32-S3 + ESP-NN**. Test đều là **subject-independent**, FP32, `fall_threshold=0.25`.

---

## 1. Hạ tầng & môi trường (đã dựng)

- Server GPU chung (RTX 2080 Ti), SSH, env conda **`har_fall`**, **Python 3.11**.
- Deps: `requirements_kfold.txt` (đầy đủ: TF 2.21, tf-keras, sklearn, seaborn, jupyter…).
- Dataset/cache: `/media/data3/users/hungvm/dataset/sisfall/`.
- Bài học GPU chung: bật `memory_growth`; **shut-down kernel sau train** (đừng ôm VRAM); model nhỏ OOM do GPU bận → `CUDA_VISIBLE_DEVICES='-1'` chạy CPU; batch nhỏ làm GPU đói (util 30–50%) → batch 256 + tăng LR.
- Rules cho agent: `AGENTS.md` + `CLAUDE.md` (import AGENTS.md).

---

## 2. Chẩn đoán gốc rễ Idle↔Trans (cốt lõi của phiên)

**Trans = chuyển tư thế tĩnh↔tĩnh (ngồi/đứng/nằm), năng lượng thấp.** Mọi kiến trúc (LSTM, TCN, ResNet, CNN) đều nhầm Idle↔Trans ở cùng mức → **bức tường nằm ở DỮ LIỆU/NHÃN, không phải model.**

Phân tích tín hiệu (đo trực tiếp trên dataset):
- **Trans sống trong GYRO**: xung ngắn ~0.5s, đỉnh ~55 dps; **accel gần như mù** (đỉnh chỉ 1.18g < đi bộ 2g).
- Nền idle gyro ~4–9 dps; transition peak 60–170 dps → **tách cực sạch bằng gyro**.
- Không gian đặc trưng (accel-RMS × gyro-RMS): Idle = thấp/thấp, **Trans = accel thấp + gyro cao** (góc riêng), Walk = accel cao + gyro vừa, Run/Fall = cao/cao.
  → Idle/Trans tách bằng **gyro**; Trans/Walk tách bằng **accel**. Cần CẢ hai.

---

## 3. Các phát hiện kỹ thuật quan trọng

### 3.1. Bug chuẩn hóa gyro (nghiêm trọng, ẩn trên FP32)
- Dữ liệu sau step1 ở **rad/s** (GYRO_SCALE có ×π/180), nhưng `apply_preprocessing` chia **/2000** — con số dành cho **°/s** → **gyro nhỏ đi ~57×**.
- **Trên INT8**: input quantize per-tensor, accel (~±1) áp đảo → scale ≈ 0.00784. Gyro /2000 ≈ 0.0005 < scale → **làm tròn về 0 → kênh gyro CHẾT trên MCU** (FP32 vẫn ≠0 nên giấu lỗi). Vì Trans sống nhờ gyro → mất tín hiệu trên chip.
- **Fix**: gyro = **dps** (`raw×4000/65536`, bỏ ×π/180), normalize **`clip(±500 dps)/500`** (rà soát dataset: per-axis p99.9≈349 dps, max ~366). **Khớp firmware** (firmware cũng dps/2000 ≡ rad/s /34.9; chuyển sang clip±500/500 cả hai). Gyro sống khỏe trên INT8.
- **Quy tắc bất biến**: chuẩn hóa lúc train **phải = lúc firmware** (train-space = deploy-space).

### 3.2. Windowing Trans theo SỰ KIỆN gyro (#3)
- Trigger = **gyro rolling-RMS > 20 dps** (nền 4 ↔ transition 21–52); **gom vùng liên tục thành 1 sự kiện** → 1 cửa sổ 2s căn vào đỉnh.
- Robust cho 1/2/3+ sub-peak (đếm: 53% sự kiện 1 peak, 27% cặp, 20% nhiều — phụ thuộc ADL) → **gom sự kiện, KHÔNG đếm peak**.
- Accel SVM **không dùng để cắt** (mù + nhiễu rìa file); chỉ làm feature cho model.
- Fall vẫn cắt theo accel (va chạm); D18/D19 → 1 Trans tại đỉnh gyro.

### 3.3. ESP-NN: thiết kế mạng bám tập lệnh tăng tốc
- **Tăng tốc**: pointwise 1×1 (**14×**), relu6 (**11×**), MaxPool/FullyConnected (**~8×**), depthwise 3×3 (**6×**), conv3 (**5.5×**), mean/GAP, add/mul.
- **KHÔNG tăng tốc** (chạy reference): **LSTM/GRU**, **dilated conv** (TCN cổ điển chậm vì cái này — v22 ~100ms), **sigmoid/tanh**, **REDUCE_MAX** (→ dùng `MaxPool(full)+Flatten` cho head max thay `GlobalMaxPooling`).
- → CNN thuần (separable + relu6 + maxpool + GAP + Dense) chạy nhanh nhất.

### 3.4. Head avg+max (tách Idle/Trans)
- GAP pha loãng xung gyro 0.5s của Trans → giống Idle. Thêm nhánh **max-over-time** (`MaxPool(full)+Flatten`, op MAX_POOL_2D) giữ đỉnh xung → tách. Đây là cú nâng Trans rõ nhất (v27→v28).

### 3.5. Những điều "tưởng vậy mà không phải"
- **Class weight = núm precision↔recall, KHÔNG tạo thông tin**: Trans w 0.55→1.0 chỉ đổi recall lấy precision (Trans F1 hòa) + làm hư Walk/Run (accuracy −0.5). → **giữ 0.55**.
- **Size INT8 do metadata, không do params**: v25 ResNet 19k params → **80KB** (75% overhead vì nhiều tensor nhỏ: separable + SE). "Ít tensor to" (LSTM) nhỏ flash hơn "nhiều tensor nhỏ". TFLite **không nén** (flatbuffer thô).
- **4 kênh không giảm size** (−0.6%) và **chỉ ~5% nhanh hơn** (số kênh input chỉ vào conv đầu ~15% MACs); **bất biến hướng lắp CHỈ một phần** (accel vẫn mang hướng trọng lực). → không đáng nếu đeo cố định.
- **GRU**: TFLM không có kernel fused → ra `WHILE` (bẫy như Keras-3-LSTM) → tránh.

---

## 4. Dòng model & kết quả (FP32, subject-independent)

| Model | Kiến trúc | Acc | Trans F1 | Fall recall | tflite |
|---|---|---|---|---|---|
| resize_64_32 | CNN-LSTM 2 lớp (64→32), windowing cũ | 0.9359 | 0.808 | 96.53% | 73.7 KB |
| resize_32 | CNN-LSTM 1 lớp (32), cũ | 0.9247 | 0.780 | 96.13% | 38 KB |
| v22_optimize | TCN ESP-NN (bỏ dilation) | 0.9147 | — | 97.33% | 81 KB |
| **v27** | CNN thuần ESP-NN, head **GAP** | 0.9103 | 0.684 | 97.47% | 24 KB |
| **v28** | v27 + head **avg+max** + Trans w0.55 | 0.9297 | 0.759 | 97.07% | 24.5 KB |
| **v29** | v28 + gyro rad/s /34.9 | 0.9299 | 0.741 | 97.73% | 25 KB |
| **v30** ⭐ | + windowing gyro-event + dps/500 + augment ×0.9/1.1 | **0.9343** | **0.799** | 97.60% | 25 KB |
| v30 (w1.0) | v30 nhưng Trans weight 1.0 | 0.9292 | 0.797 | 97.73% | 25 KB |
| **v31** | v30 nhưng **4 kênh** [3acc+gyro_mag] | 0.9237 | 0.788 | 97.47% | 24.9 KB |
| v30_lstm32 | pipeline v30 + **1 lớp LSTM(32)** | *(đang train)* | | | |

> Lưu ý: v30/v31 test trên windowing mới (Trans support 567 vs 410 ở v27–v29) → khác bộ test, không hoàn toàn apples-to-apples; nhưng vẫn cho thấy xu hướng.

**Quán quân hiện tại: v30 (6 kênh, Trans w=0.55)** — Acc 0.9343, Trans F1 0.799, Fall 97.60%, 25 KB, mọi op ESP-NN tăng tốc.

---

## 5. Quyết định đã chốt
- **6 kênh** (không 4) cho thiết bị đeo cố định: +1 accuracy, firmware đơn giản (clip+scale từng kênh, không tính magnitude), size như nhau.
- **Windowing Trans = sự kiện gyro RMS>20 dps**.
- **Normalize = dps `clip(±500)/500`** (đồng bộ firmware).
- **Trans class_weight = 0.55**.
- **v30** là baseline CNN-ESP-NN hiện tại.

---

## 6. Việc còn dở / bước tiếp
- [ ] **Eval INT8** (chạy `.tflite` trên X_test): so v28 (gyro chết) vs v30 (gyro sống) — đây mới là số THẬT trên chip; FP32 không thể hiện chênh lệch gyro-sống.
- [ ] **Báo cáo research taxonomy nhãn HAR/fall** (deep-research đang chạy nền): có nên giữ/tách/gộp lớp Trans, các hệ thống dùng tập nhãn nào.
- [ ] **v30_lstm32** vừa build — train + so với v30 (CNN) trên pipeline mới.
- [ ] (Tùy chọn) v32 rút window 200→100 hoặc kênh deep nhỏ hơn để tăng tốc thật (~20–40%, hơn hẳn bỏ kênh input).
- [ ] Cảnh báo deploy: **firmware phải clip±500/500 (dps)** khớp train; đổi scale = sửa cả firmware.

---

## 7. Folder chính
- `train_v27` … `train_v31`, `train_v30/{trans_weight_0_55, trans_weight_1}`, `train_v30_lstm32` — các thí nghiệm.
- `train_v30/SisFall_KFold_Experiments_v3.ipynb` — KFold cross-population (Elderly/Young/Both), đã đồng bộ pipeline v30 + 4 kênh.
- `export_tflite_with_ops.py` — convert INT8 + sinh `.cc/.h` nhúng sẵn ops/quant/resolver.
- `AGENTS.md` / `CLAUDE.md` — rules dự án.
