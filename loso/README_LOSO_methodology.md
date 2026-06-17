# LOSO Personalization — Thiết kế thí nghiệm (Bài báo #1)

> Mục tiêu: chứng minh confusion **Trans/Idle** là **inter-subject** (biến thiên giữa người),
> và **on-device personalization** dưới ngân sách bộ nhớ ESP32-S3 thu hồi được phần lớn gap —
> mà **không** làm giảm Fall Recall.

---

## 0. Tiền đề & cảnh báo (đọc trước khi chạy)

- **Nguồn dữ liệu:** trỏ `DATA_DIR` về đúng nguồn windowed *5 lớp* đã tạo ra cache v25/v26.
  Thư mục `SisFall_dataset_Windowed` ở máy local chỉ có `D01–D19`/`F01–F15` → với
  `parse_filename_info` hiện tại chỉ sinh **Walk/Run/Fall** (Idle/Trans cần file mang
  `_Trans_/_StandSit_/_Lie_`). Thiếu Idle/Trans là hỏng toàn bộ giả thuyết.
- **Subject pool:** SisFall có **38 subject** — `SA01–SA23` (23 người trưởng thành) +
  `SE01–SE15` (15 người cao tuổi). Tên file: `D01_SA01_R01_W000.csv`
  = `activity_subject_trial_window` ⇒ `parts[1]`=subject, `parts[2]`=trial.
- **Tái dùng code sẵn có:** loader gọi thẳng `DataPreprocessor.parse_filename_info`
  và logic preprocessing fixed-scale ( `clip ±8g /8.0` cho accel, `/2000.0` cho gyro )
  trong [`ml_pipeline_v26.py`](../ml_pipeline_v26.py) → label & chuẩn hoá khớp tuyệt đối v25/v26.

---

## 1. Hai thí nghiệm

### Thí nghiệm A — Chứng minh biến thiên giữa người (động cơ của bài)
**LOSO thuần, không adapt.** Với mỗi subject S: train model trên 37 subject còn lại,
test trên S (zero adaptation). Báo cáo **F1-Trans / F1-Idle per-subject**.

- **Kết quả kỳ vọng:** phương sai per-subject *cao* (box-plot trải rộng). Đây đã là một
  finding đăng được: "model global không công bằng giữa người dùng".
- **Phân tầng:** so sánh nhóm `SA` (trẻ) vs `SE` (cao tuổi) — kỳ vọng `SE` phương sai lớn hơn
  (cử động đặc thù hơn). Đây là góc *y tế* mạnh cho venue EMBC/J-BHI.
- **Chống nhầm "phương sai = nhiễu":** kèm **oracle upper bound** — train *có* dữ liệu của S
  (subject-dependent) → khoảng cách LOSO↔oracle chính là *headroom* mà personalization nhắm tới.

### Thí nghiệm B — Personalization dưới ngân sách bộ nhớ (đóng góp chính)
Với mỗi subject S (dùng base model LOSO của A):
1. Lấy **K window/lớp** của S làm tập **calibration** (few-shot enrollment).
2. Fine-tune **một tập con tham số** trên calibration.
3. Đánh giá trên phần **còn lại** của S (disjoint theo trial).
4. Đo **ΔF1** và **chi phí bộ nhớ huấn luyện**.

Quét `K ∈ {5,10,20,40}` để vẽ đường cong "bao nhiêu dữ liệu enrollment là đủ".

---

## 2. Năm chốt chặn đúng đắn (đây là thứ khiến bài qua review)

1. **Tách calibration/test theo TRIAL, không theo window.** Window cùng một trial (`R01`)
   tương quan cao → chia random gây *leakage*, thổi phồng kết quả. Calibration lấy từ một
   tập trial; test lấy từ trial *khác* của cùng subject.
2. **KHÔNG đưa Fall vào calibration.** Thực tế người dùng không "ngã thử" khi enrollment.
   Chỉ personalize bằng lớp thu thập được theo yêu cầu (Walk/Idle/Trans/Run). Đây là điểm
   *hiện thực hoá* mạnh: cải thiện ranh giới Trans/Idle bằng ADL, **không đụng tới Fall**.
3. **Ràng buộc an toàn: Fall Recall không được giảm.** Báo cáo Fall Recall trước/sau adapt.
   Nếu personalize làm tụt Fall Recall → phương pháp vô dụng dù F1-Trans tăng. Đây là
   *constraint*, không phải metric phụ.
4. **Nhiều seed cho việc chọn calibration window.** Chọn K window nào có ảnh hưởng → chạy
   ≥5 seed, báo cáo mean ± std. Không báo cáo một lần chạy may mắn.
5. **Base preprocessing cố định, đóng băng.** Giữ nguyên fixed-scale (không z-score động) —
   khớp ràng buộc MCU. Không được lén normalize lại theo subject (sẽ là một "personalization"
   trá hình không deploy được).

---

## 3. Các chiến lược adapt (trục co-design — sân nhà embedded)

Xếp theo chi phí bộ nhớ huấn luyện tăng dần. Trên MCU, thứ đắt nhất khi backprop là **activation
memory** (phải lưu activation để tính gradient), không chỉ số tham số. Chiến lược càng gần đầu ra,
activation cần lưu càng ít → càng "deployable".

| Chiến lược | Tham số train | Activation backprop | Ghi chú |
|---|---|---|---|
| `last_dense` | chỉ `Dense(5)` cuối | ~0 (chỉ feature cuối) | rẻ nhất, kỳ vọng gain khiêm tốn |
| `norm_tuning` (BN γ/β) | affine của BatchNorm | trung bình | kiểu TinyTL, rất hợp INT8 |
| `se_only` | Dense trong SE block | thấp–trung bình | tinh chỉnh "kênh nào quan trọng" theo người |
| `last_block` | Block 5 + Dense | cao hơn | nhiều dung lượng hơn |
| `sparse_topk` | top-k% weight theo |grad| | tuỳ chọn vị trí | bản 256KB-paper; **đóng góp method** |
| `full` (upper bound) | toàn mạng | cao nhất | không deploy được, chỉ làm trần |

**Trục kết quả chính của bài:** vẽ **ΔF1-Trans (y)** vs **trainable-memory KB (x)** cho từng
chiến lược → **Pareto front**. Khoanh vùng các điểm *thực sự vừa* ngân sách ESP32-S3
(arena thực dùng của v25 ≈ 28 KB; RAM tự do ~256 KB) → "deployable region".

---

## 4. Metrics báo cáo

- Per-class F1 (đặc biệt **Trans, Idle**), Macro-F1, Accuracy.
- **Fall Recall** (ràng buộc an toàn) — trước & sau adapt.
- ΔF1 = sau − trước, mean ± std qua subject × seed.
- **Trainable params + bytes** mỗi chiến lược (proxy memory; kèm ước tính activation memory).
- (Mở rộng) latency/RAM một bước update đo/ước tính trên ESP32-S3.

---

## 5. Output kỳ vọng (hình cho paper)

1. **Box-plot** F1-Trans per-subject (Thí nghiệm A) — tách `SA` vs `SE`.
2. **Đường cong** ΔF1 theo K (lượng dữ liệu enrollment).
3. **Pareto** ΔF1 vs trainable-memory, đánh dấu "deployable region" (hình chủ lực).
4. **Bảng** Fall Recall trước/sau cho mọi chiến lược (chứng minh an toàn không vi phạm).

---

## 6. Ánh xạ sang luận điểm bài báo

| Claim | Bằng chứng |
|---|---|
| "Confusion Trans/Idle là inter-subject" | box-plot phương sai cao + gap LOSO↔oracle |
| "Personalize nhẹ thu hồi phần lớn gap" | đường cong ΔF1 theo K dương, đáng kể |
| "Khả thi trong ngân sách MCU" | Pareto: có điểm trong deployable region |
| "An toàn được giữ" | Fall Recall không giảm sau adapt |
| "Hợp cho người cao tuổi" | gain lớn hơn ở nhóm SE |

---

## 7. Cách chạy

```bash
# Pilot nhanh (vài subject, ít epoch) để smoke-test:
python loso/loso_pipeline.py --data-dir <NGUON_5_LOP> --subjects SA22 SA23 SE12 --epochs 20

# Full LOSO (38 subject) cho bản final:
python loso/loso_pipeline.py --data-dir <NGUON_5_LOP> --full --epochs 60 --seeds 5
```

Xem `loso/loso_pipeline.py` — skeleton runnable, đã tái dùng loader & preprocessing của bạn.
Các chỗ cần bạn quyết được đánh dấu `# TODO`.
