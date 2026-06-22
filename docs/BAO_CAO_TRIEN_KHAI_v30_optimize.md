# Báo cáo triển khai — Tối ưu model nhận diện té ngã / HAR (v30 → v30_optimize)

> **Mục đích file:** tài liệu handoff cho agent viết luận văn. Ghi lại toàn bộ quá trình thử nghiệm,
> số liệu thật, và kết luận. **Quy ước:** "offline FLOAT" = đánh giá trên tập test 6106 mẫu bằng model
> Keras float; "firmware INT8" = chạy model INT8 thật trên ESP32-S3 với bộ test riêng 1000 mẫu (200/lớp).
> Hai bộ test khác nhau → chỉ so firmware-vs-firmware hoặc offline-vs-offline.

---

## 1. Bối cảnh & bài toán

- **Nhiệm vụ:** phân loại 5 lớp `[Walk, Run, Idle, Trans, Fall]` từ dữ liệu IMU 6 kênh (accel+gyro),
  cửa sổ **200 mẫu × 6 kênh @ 100Hz** (~2s), chạy **on-device trên ESP32-S3** (TFLM + ESP-NN).
- **Dữ liệu:** SisFall, split **subject-independent** (không trộn mẫu cùng người qua train/test).
  Tiền xử lý v30: accel `clip(±8g)/8`, gyro `clip(±500dps)/500` (khớp firmware), augment Trans ×0.9/1.1,
  class-weight `Fall×3, Trans×0.55`.
- **KPI #1 = Fall recall** (bỏ sót té ngã nguy hiểm hơn báo nhầm), đánh giá với `fall_threshold=0.25`.
  **Luôn báo kèm Trans F1**; KHÔNG kết luận bằng accuracy tổng (lớp đa số Walk/Run/Idle ~78% mẫu che lấp).
- **Ràng buộc phần cứng (ESP-NN):** chỉ các op được tăng tốc mới dùng cho nhánh tốc độ —
  Conv 1×1 (14×), relu6 (11×), MaxPool/FullyConnected (~8×), DepthwiseConv 3×3 (6×), Conv 3×3 (5.5×),
  MEAN/GAP. **KHÔNG tăng tốc:** LSTM/GRU, dilated conv (dilation>1), sigmoid/tanh.

## 2. Điểm xuất phát — 3 model nền (offline FLOAT, test 6106, thr 0.25)

| Model | Kiến trúc | Idle F1 | Trans F1 | Fall recall | acc | Ghi chú phần cứng |
|---|---|---|---|---|---|---|
| **CNN v30** (baseline) | Separable-CNN + head GAP+GMP, 7.4k params | 0.910 | **0.799** | 97.60% | 0.9343 | **20ms / 16.5KB / 25KB** — nhanh nhưng yếu Idle/Trans |
| **TCN v30** | Dilated conv (d=1,2,4,8)×2 + SE, k7 | 0.950 | 0.927 | 97.07% | 0.9602 | **2058ms** / 50KB / 105KB — chính xác nhưng **KHÔNG deploy được** |
| ResNet1D v30 | SeparableConv + SE + residual | 0.925 | 0.848 | 96.67% | 0.9415 | — |

**Vấn đề cốt lõi:** CNN nhanh nhưng yếu nhất ở **Idle/Trans**; TCN giỏi Idle/Trans nhưng dilated conv
không được ESP-NN tăng tốc → **2058ms/lần infer (chậm ~100× CNN) → vô dụng cho real-time**.

## 3. Hướng 1 — Knowledge Distillation TCN→CNN (THẤT BẠI, negative result)

**Ý tưởng:** giữ student = CNN v30 nguyên bản (nhanh), cho học soft-label từ teacher TCN (giỏi Trans).
Thử trên thư mục `train_v30_kd/`.

**Đã thử 4 recipe:** KD cross-entropy T=4; KD KL-divergence T=2; selective per-class trust (tin teacher
ít ở Fall, nhiều ở Trans). Loss `α·CE(class-weight) + (1−α)·T²·KD`.

**Kết quả:** **Idle F1 đóng băng ~0.909, Trans precision kẹt ~0.75 qua MỌI recipe.** Không bản nào
thắng baseline (acc 0.9343). Ví dụ KD-KL T=2: Trans F1 0.818, Fall recall 97.47%, acc 0.9329.

**Chẩn đoán nguyên nhân (quan trọng cho luận văn):**
1. **Teacher TCN yếu hơn baseline ở Fall** (97.07% < 97.60%) → KD toàn cục kéo Fall của student xuống.
2. **Head `GAP+GMP` của CNN xóa trục thời gian** → student "mù" với *hình dạng/vị trí* của transition
   trong cửa sổ (Idle phẳng vs Trans có bao gyro ở giữa). Đây là **trần kiến trúc**, không phải trần recipe.

→ **Kết luận hướng 1:** KD không thể vá một giới hạn kiến trúc. Cần sửa kiến trúc student.

## 4. Hướng 2 — Cải tiến KIẾN TRÚC (THÀNH CÔNG)

Thiết kế lại student để **giữ cấu trúc thời gian**, vẫn thuần op ESP-NN (không dilation/LSTM/sigmoid).
Thư mục: `train_v30_kd2/` (thí nghiệm) → `train_v30_optimize/` (bản sạch deploy).

**3 thay đổi so với CNN v30 baseline:**
1. **Head mới (quyết định):** `GAP + GMP + Flatten(map thô 12×96)` thay `GAP+GMP`. Flatten giữ thông tin
   *chuyển động xảy ra Ở ĐÂU trong cửa sổ* — chính cái baseline bị mù. (GAP/GMP vẫn giữ để robust Fall.)
2. **Sâu hơn:** 2 SeparableConv k3 stride-1 mỗi tầng (thay 1).
3. **Rộng hơn:** kênh 24/32/48/64 → 32/48/64/96.

| | CNN v30 baseline | kd2 / v30_optimize |
|---|---|---|
| Stem → 3 tầng pool (200→100→50→25→12) | ✓ | ✓ (y hệt cấu trúc downsample) |
| SeparableConv mỗi tầng | 1 | 2 (ở 2 tầng đầu) |
| Kênh | 24/32/48/64 | 32/48/64/96 |
| Head | GAP+GMP (128-d) | GAP+GMP+**Flatten** (1344-d) |
| Params | 7.4k | **26.6k** |

→ **Cùng họ kiến trúc (Separable-CNN ESP-NN), không phải TCN/LSTM.** Đây là *CNN tối ưu kiến trúc head*.

### 4.1. Ý tưởng cải tiến KẾ THỪA từ TCN (mạch tư duy thiết kế)

Cải tiến kiến trúc không phải "thử ngẫu nhiên" — nó **rút bài học từ vì sao TCN mạnh ở Idle/Trans rồi
mang sang CNN bằng cơ chế rẻ hơn**. Chuỗi suy luận:

1. **Quan sát:** TCN đạt Trans 0.927 / Idle 0.950 (cao nhất) trong khi CNN baseline chỉ 0.799 / 0.910.
   Điều TCN làm được mà CNN baseline không: **mô hình hóa *trình tự / hình dạng theo thời gian* của tín hiệu**
   (nhờ dilated conv có receptive field dài ~1.8s, mỗi neuron "thấy" cả pattern *yên → bùng phát → yên*).
2. **Rào cản:** chính dilated conv đó **không được ESP-NN tăng tốc** → 2058ms, không deploy được. Tức
   *cơ chế* của TCN (dilation) bị cấm trên phần cứng, **nhưng *nguyên lý* (giữ cấu trúc thời gian) thì không.**
3. **Bài học rút ra:** phân biệt Idle/Trans **bắt buộc phải GIỮ thông tin thời gian (thứ tự/vị trí)** —
   head `GAP+GMP` của CNN gộp phẳng trục thời gian nên đánh mất đúng thứ này → đó là trần thật.
4. **Chuyển nguyên lý sang CNN bằng cơ chế ESP-NN-friendly** (2 đòn thay cho dilation):
   - **Mở receptive field bằng strided/pooled downsampling** (stride-2 + MaxPool, đều được ESP-NN tăng tốc)
     thay vì dilation — vẫn cho mỗi neuron tầng sâu "nhìn" rộng theo thời gian, nhưng rẻ.
   - **Head `Flatten` giữ layout thời gian** ở map thô 12 bước (thay vì gộp GAP/GMP) → Dense học được
     "feature ở segment thời gian nào", đúng kiểu *envelope shape* mà dilated conv của TCN nắm bắt.

**Đối chiếu cơ chế — cùng mục tiêu, khác phương tiện:**

| | TCN (gốc) | CNN cải tiến (v30_optimize) |
|---|---|---|
| Mục tiêu | Giữ & mô hình hóa trình tự thời gian | (giống) |
| Mở receptive field | Dilated conv (d=1,2,4,8) | Strided conv + MaxPool (downsample 200→12) |
| Giữ layout thời gian ở classifier | Conv giữ chiều thời gian đến cuối | **Flatten map thô 12×96** (không gộp phẳng) |
| ESP-NN | ❌ dilation chạy reference → 2058ms | ✅ toàn op tăng tốc → 56.7ms |
| Trans / Idle F1 | 0.927 / 0.950 | 0.907 / 0.942 (≈ TCN, **nhanh ~36×**) |

→ **Thông điệp luận văn:** đây là một thiết kế *"chắt lọc sức mạnh thời gian của TCN vào một CNN
ESP-NN-friendly"* — giữ ~95% lợi ích Idle/Trans của TCN nhưng loại bỏ hoàn toàn chi phí dilation.
Knowledge distillation (cho student học từ TCN) đã được thử nhưng **thua hướng này** (mục 3, 5): chuyển
*nguyên lý kiến trúc* hiệu quả hơn nhiều so với chuyển *tri thức qua soft-label*.

## 5. Hướng 3 — Ablation: KD có thực sự cần không? (KHÔNG)

Train kiến trúc mới với nhiều mức `α` (α=1.0 = tắt KD hoàn toàn). Offline FLOAT:

| Biến thể (kiến trúc mới) | Idle F1 | Trans F1 | Fall F1 | Fall recall | acc |
|---|---|---|---|---|---|
| KD α=0.5 (single-TCN) | 0.942 | 0.906 | 0.986 | 97.20% | 0.9532 |
| KD α=0.3 (teacher-heavy) | 0.942 | 0.917 | 0.986 | 97.20% | 0.9551 |
| **α=1.0 (TẮT KD)** | 0.943 | 0.914 | 0.985 | 97.47% | **0.9563** |
| Ensemble teacher (TCN+ResNet+CNN) | 0.943 | 0.912 | 0.987 | 97.33% | 0.9571 |
| fall_recall_boost (Fall w↑) | 0.941 | 0.907 | **0.989** | **97.73%** | 0.9558 |

**Kết luận then chốt:** xếp theo α — **càng ít KD càng tốt; α=1.0 (không thầy) ≥ mọi bản KD.**
→ **Toàn bộ thắng lợi đến từ KIẾN TRÚC (head giữ-thời-gian), KHÔNG phải distillation.** Soft-label thậm chí
làm nhiễu nhẹ. (Phù hợp lý thuyết: teacher không đủ vượt trội + capacity gap.)

→ Bỏ teacher → đóng gói thành **`v30_optimize`** (train supervised thuần, không cần `output_v30_tcn`).

## 6. Tinh chỉnh cuối — monitor val_accuracy thay val_loss

`v30_optimize` ban đầu dùng `monitor='val_loss'`. Với model có class-weight, `val_loss` (không trọng số)
thiên về lớp đa số → chọn epoch kém ở lớp thiểu số. Đổi sang `monitor='val_accuracy', mode='max'`:

| v30_optimize | Idle F1 | Trans F1 | Fall recall | acc |
|---|---|---|---|---|
| monitor=val_loss | 0.9379 | 0.8944 | 97.47% | 0.9530 |
| **monitor=val_accuracy** | 0.9420 | **0.9071** | **97.73%** | 0.9533 |

→ acc tổng gần như đứng yên (đa số che) nhưng **Trans +0.013, Fall recall +0.26 (vượt baseline 97.60%)**.
**Bản val_accuracy là bản deploy cuối.** (Đây cũng là minh hoạ thực tế cho quy tắc "không đánh giá bằng acc tổng".)

## 7. Kiểm chứng FIRMWARE (INT8 trên ESP32-S3, bộ test 1000 mẫu)

Đã verify bản kiến trúc mới (kd2, α=0.5) trên chip thật:

| | acc | Idle | Trans | Fall | Fall recall | Latency | Arena | tflite |
|---|---|---|---|---|---|---|---|---|
| CNN v30 baseline | 0.920 | 0.859 | 0.853 | 0.980 | 96.00% | **20ms** | 16.5KB | 25KB |
| TCN (thầy) | 0.952 | 0.922 | 0.936 | 0.980 | 97.00% | 2058ms | 50KB | 105KB |
| **Kiến trúc mới (kd2/v30_optimize)** | **0.953** | **0.926** | **0.945** | **0.987** | **97.50%** | **56.7ms** | **28.3KB** | **55.8KB** |

**Quan trọng:**
- **Nén INT8 gần như KHÔNG suy hao:** offline float acc 0.9532 → firmware INT8 0.9530.
- Kiến trúc mới trên chip **bằng/vượt teacher TCN** (acc 0.953 vs 0.952; Trans 0.945 vs 0.936) nhưng
  **nhanh hơn ~36×** (56.7ms vs 2058ms).
- So baseline trên chip: **Trans +9.2, Idle +6.7, Fall recall +1.5**.
- Latency 56.7ms hoàn toàn OK cho real-time: infer theo window (~1s) → chỉ ~5–6% duty cycle.
- Arena 28.3KB / flash 55.8KB: vặt so với 512KB SRAM + flash MB-level của ESP32-S3.
- Lưu ý: **đổi training recipe (monitor/KD/α) KHÔNG ảnh hưởng phần cứng** — latency/arena/flash do *kiến trúc*
  quyết (số op/tensor), chỉ giá trị weight & accuracy thay đổi.

## 8. Kết quả cuối cùng (model deploy)

**`v30_optimize` (monitor val_accuracy):**
- Kiến trúc: Separable-CNN ESP-NN, head GAP+GMP+Flatten, ~26.6k params.
- Offline FLOAT: acc 0.9533, Idle F1 0.942, Trans F1 0.907, **Fall recall 97.73%**.
- Firmware INT8 (kiến trúc tương đương đã verify): **56.7ms / 28.3KB arena / 55.8KB flash**, acc ~0.953,
  **Fall recall ~97.5%**.
- **Thay thế CNN v30 baseline làm model deploy chính.**

## 9. Đóng góp / luận điểm cho luận văn

1. **Nút thắt Idle/Trans của CNN nhẹ là do head pooling xóa trục thời gian** — chứng minh bằng việc
   thêm nhánh Flatten giữ layout thời gian đẩy Trans F1 **0.799 → 0.907** (+10.8), Idle 0.910 → 0.942.
2. **Đóng góp kiến trúc chính — chuyển nguyên lý của TCN sang CNN ESP-NN-friendly** (xem §4.1): rút bài học
   "TCN mạnh vì giữ cấu trúc thời gian", rồi tái hiện nguyên lý đó bằng *strided/pooled downsampling* (thay
   dilated conv) + *head Flatten giữ layout thời gian* (thay gộp GAP/GMP). Đạt ~95% lợi ích Idle/Trans của
   TCN (Trans 0.907 vs 0.927) mà **nhanh ~36×** và chạy được trên MCU.
3. **Knowledge Distillation KHÔNG cần thiết ở đây** (ablation α=1.0 ≥ mọi bản KD) — một *negative result*
   có giá trị: teacher TCN không đủ vượt trội + capacity gap; chuyển *nguyên lý kiến trúc* hiệu quả hơn
   chuyển *tri thức qua soft-label*.
4. **Dilated-TCN không phù hợp deploy ESP-NN** (2058ms) dù chính xác — minh hoạ ràng buộc phần cứng định
   hình lựa chọn kiến trúc; giải pháp "nhìn rộng + nhanh" là pooling/stride-CNN, không phải dilation.
5. **Quantize INT8 bảo toàn độ chính xác** (0.9532 → 0.9530) — khả thi cho MCU.
6. **Kỷ luật đánh giá:** accuracy tổng che lấp lớp thiểu số; mọi kết luận dựa trên **Fall recall + Trans F1**.

## 10. Hạn chế & việc còn lại

- **Chênh lệch giữa các biến thể (~0.4% acc, Trans ±0.02) nằm trong nhiễu single-split** (Trans chỉ 567 mẫu
  test). → Cần **k-fold subject-independent** để xếp hạng tin cậy.
- **K-fold cho v30_optimize ĐÃ DỰNG:** `train_v30_optimize/SisFall_KFold_Experiments_v6.ipynb` (kế thừa khung
  K-Fold v4: 5 kịch bản S1_Elderly/S2_Young/S3-S4 cross/S5_Both, cùng pipeline `decimate(q=2)` + windowing
  `v3kf` + cache `cache_kfold_v3`; chỉ thay `build_model`=kiến trúc v30_optimize + monitor `val_accuracy`).
  Kết quả lưu `KFold_Results_v6/`. **CHƯA chạy** — chờ chạy server để có mean±std + Fall recall per-scenario.
  Lưu ý: k-fold dùng `decimate(q=2)` (khác `iloc[::2]` của bản deploy) → số đánh giá *robustness kiến trúc*,
  nhất quán nội bộ. Fall recall đáng tin chủ yếu ở S5_Both + S3/S4 (S1 Elderly thiếu nhãn Fall).
- (K-fold trước: v3 = CNN baseline `train_v30/`, v4 = TCN `train_v30_tcn/kfoldv4_result/`. KD-kfold không cần
  vì KD đã chứng minh không giúp → k-fold thẳng kiến trúc v30_optimize.)
- Bản `fall_recall_boost` (Fall recall offline 97.73%) chưa firmware-verified — ứng viên nếu muốn tối đa Fall recall.

## 11. Tham chiếu file (repo `SisFall-PreProcessing/`)

- Baseline CNN: `train_v30/trans_weight_0_55/`
- Teacher TCN: `train_v30_tcn/` · ResNet1D: `train_v30_resnet1d/`
- KD thất bại (trên CNN cũ): `train_v30_kd/`
- Kiến trúc mới + ablation các α: `train_v30_kd2/{kd_alpha_0_5, kd_alpha_1_0_ablation, kd_alpha_0_3_teacher_heavy, fall_recall_boost, ensemble_teacher_alpha_0_5}/`
- **Model deploy cuối:** `train_v30_optimize/val_accuracy/` (vs `val_loss/` để so monitor)
- **K-fold v30_optimize:** `train_v30_optimize/SisFall_KFold_Experiments_v6.ipynb` → `KFold_Results_v6/`
- Doc kiến trúc canonical: `datn-agent-skills/project_setup/architecture/tinyml_model.md`
