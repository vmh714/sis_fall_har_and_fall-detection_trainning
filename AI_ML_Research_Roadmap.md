# Lộ trình Học & Nghiên cứu AI/ML hướng tinyML / Edge AI

> Tài liệu đóng gói tri thức từ phiên thảo luận định hướng (2026-06-17).
> Bối cảnh người học: nền **embedded** vững, gap 1 năm trước khi học thạc sĩ, quỹ thời gian
> 20–30h/tuần chia cho *embedded + ngoại ngữ + ML/DL* (⇒ ~8–12h/tuần cho ML).
> Mục tiêu dài hạn: trở thành **R&D researcher** mảng tinyML / Edge AI.

---

## 0. TL;DR — Định vị chiến lược

- **Lợi thế độc quyền của bạn = deployment + hiểu chi phí phần cứng**, không phải lý thuyết ML thuần.
  Đừng cạnh tranh độ sâu toán với dân ML; chiếm ô *"học máy dưới ràng buộc phần cứng cực ngặt,
  đo đạc thật"* — ít người, cầu doanh nghiệp cao.
- **Deployment là bàn đạp, không phải đích.** Chỉ làm deployment = engineering. Phải gắn vào một
  *trục nghiên cứu dọc* để thành research.
- **Trục dọc đã chọn:** On-Device Learning / Continual Learning cho sensor time-series trên MCU
  (giao của "dư địa farm bài còn nhiều" + "giá trị doanh nghiệp cao" + "hợp gốc embedded").
- **Tận dụng tài sản sẵn có:** project SisFall (fall detection trên ESP32-S3) đã ở trình độ
  researcher — biến nó thành bài báo thay vì học lại từ đầu.

---

## 1. Nhận xét: Học ML "from scratch" với NumPy

**Nên làm — nhưng có liều lượng, không sa đà.**

- **Đáng làm** (với người gốc embedded còn đáng hơn người thường): tự code bằng NumPy
  *linear regression, logistic regression, một MLP + backprop viết tay, forward pass của 1 CNN*.
  Lý do: tinyML là môn học về **chi phí từng phép tính** (mỗi MAC, mỗi byte RAM). Tự viết backprop
  cho bạn trực giác về cost của từng layer — đúng thứ cần khi nhét model vào MCU 256KB.
- **Đừng sa đà:** đừng reimplement cả framework, đừng viết lại Transformer from scratch giai đoạn này.
  Mục tiêu là *trực giác*, không phải tái tạo PyTorch.
- **Bẫy cần tránh:** dân from-scratch hay "luôn muốn tự code mọi thứ" → chậm tiến độ. Khi đã hiểu
  cơ chế, dùng PyTorch/TF không mặc cảm. Research thật chạy trên framework.
- **Quy tắc vàng:** *from scratch để hiểu, framework để làm việc.*

---

## 2. Roadmap học ML/DL theo giai đoạn (~9–12 tháng, ~8–12h/tuần)

| GĐ | Nội dung | Thời lượng | Resource chính |
|----|----------|-----------|----------------|
| **0. Toán đủ dùng** | Linear algebra (nhân ma trận, eigen), Calculus (chain rule = tim của backprop), Probability (kỳ vọng, Bayes) | ~3–4 tuần, ôn dần | 3Blue1Brown *Essence of Linear Algebra* |
| **1. Python + NumPy fluency** | Tư duy *vectorization* (bỏ for-loop, nghĩ theo mảng) | ~3–4 tuần | — |
| **2. Classical ML + from scratch** ⭐ | train/val/test, overfitting, regularization, gradient descent, loss. **Tự code NumPy**: lin/log reg, MLP+backprop | ~6–8 tuần | Andrew Ng *ML Specialization*; làm lại bài tập bằng NumPy thuần |
| **3. Deep Learning + framework** | Chuyển sang **PyTorch**. CNN (cốt lõi edge vision), RNN cơ bản, chút Transformer | ~8–10 tuần | fast.ai hoặc Andrew Ng *DL Specialization* |
| **4. tinyML / Edge AI** ⭐ | Quantization (int8), pruning, KD, NAS cơ bản. **TFLite Micro, CMSIS-NN, Edge Impulse**. 1 project end-to-end deploy lên MCU + đo latency/RAM/năng lượng thật | ~8–12 tuần | Harvard/Google *TinyML* (edX, Vijay Janapa Reddi); MIT 6.5940 (Song Han) |

**Mẹo dùng thời gian:**
- Gộp **ngoại ngữ vào ML**: đọc paper, xem khóa học tiếng Anh → một mũi tên hai đích.
- **Embedded KHÔNG cạnh tranh thời gian với ML**: RTOS, DSP, fixed-point, tối ưu bộ nhớ
  tái dùng trực tiếp trong tinyML. Hai mảng bồi đắp lẫn nhau.
- **Project > certificate.** Một repo GitHub có tinyML chạy thật trên MCU > 5 khóa học.

---

## 3. Hướng nghiên cứu (xếp theo: dư địa farm bài × giá trị doanh nghiệp)

> Sự thật phải nói thẳng: **"dễ farm bài" và "novelty cao" thường ngược nhau.** Applied (HAR,
> fall detection) farm dễ nhưng incremental; systems/methods khó hơn nhưng định vị researcher thật.
> Chiến lược: dùng mảng dễ farm để có bài + dữ liệu + tự tin, rồi nâng dần lên mảng method.
> **Chọn một TRỤC DỌC, đừng chọn một điểm rời rạc.**

| Hướng | Farm bài | Giá trị DN | Hợp embedded |
|-------|---------|-----------|--------------|
| 1. **On-device / Continual learning trên MCU** | Rất cao | Cao (personalization, privacy, chống drift, no-cloud) | ⭐⭐⭐ |
| 2. **Efficient sensor/time-series models** (HAR, anomaly, predictive maintenance) | Rất cao | Rất cao (IoT công nghiệp, wearable y tế) | ⭐⭐ |
| 3. **Hardware-aware NAS / HW-model co-design** | Cao | Cao | ⭐⭐⭐ (áp đảo dân ML thuần) |
| 4. Sub-8-bit / mixed-precision quantization | Trung bình | Cao | ⭐⭐⭐ |
| 5. On-device SLM / foundation model ở edge | Rất cao (cực hot) | Rất cao nhưng cạnh tranh khốc liệt, cần GPU farm | ⭐ |
| 6. Spiking NN / neuromorphic / in-memory computing | Novelty cao, khó farm | Mới manh nha | ⭐⭐ |
| 7. Federated learning ở edge (TinyFL) | Cao | Cao (privacy, dữ liệu phân tán) | ⭐⭐ |

### Tổ hợp đề xuất (trục dọc 3 lớp leo thang)
- **🎯 Chính:** On-device/Continual Learning cho sensor time-series trên MCU (hướng 1+2 = sweet spot).
- **🪜 Phụ (leo lên method):** Hardware-aware NAS / quantization co-design (hướng 3) — gốc embedded thành vũ khí độc quyền.
- **🛠️ Nền tảng kèm theo:** TinyML systems/compiler (TVM-micro, code-gen, đo lường) — khiến *mọi* bài có phần "deploy & đo thật".

**Venue mục tiêu:** tinyML Summit · ML-for-Embedded (DATE, DAC) · SenSys/IPSN/IoTDI · MLSys ·
EMBC / IEEE J-BHI (nhánh wearable y tế).

---

## 4. Labs / nhóm cần theo dõi

- **Song Han — MIT HAN Lab** ⭐ trung tâm tuyệt đối. MCUNet, TinyTL, Once-for-All, On-Device Training.
  Khóa **MIT 6.5940** *TinyML and Efficient Deep Learning Computing* (slides + video free).
- **Luca Benini — ETH Zürich / Univ. Bologna** ⭐ cực hợp embedded. PULP, RISC-V, energy-efficient edge AI.
- **Marian Verhelst — KU Leuven** — accelerator ML tiết kiệm năng lượng, co-design.
- **Manuel Roveri / Massimo Pavan — Politecnico di Milano** — on-device learning & continual learning trên MCU (rất sát).
- **Nicholas Lane — Cambridge / Flower Labs** — federated + on-device (nếu rẽ nhánh TinyFL).
- **Vijay Janapa Reddi — Harvard** — benchmark MLPerf Tiny + khóa TinyML edX.
- **Pete Warden — Google** — TF Lite Micro, blog thực dụng cho người mới.

---

## 5. Lộ trình đọc paper (theo thứ tự, không đọc tuyến tính hết)

**Lớp 0 — Nền hiệu quả**
1. MobileNetV2 (Sandler 2018) — depthwise separable conv.
2. Deep Compression (Han 2016) — pruning + quantization + Huffman.
3. Quantization for Integer-Arithmetic-Only Inference (Jacob 2018) — nền int8.

**Lớp 1 — Hệ thống tinyML**
4. ⭐ MCUNet (Lin 2020, NeurIPS) — TinyNAS + TinyEngine co-design.
5. MCUNetV2 / patch-based inference (Lin 2021) — peak-memory mới là kẻ thù thật.
6. MLPerf Tiny Benchmark (Banbury 2021).

**Lớp 2 — Lõi On-Device Learning** ⭐ trái tim hướng chính
7. TinyTL (Cai 2020, NeurIPS) — freeze weight, train bias + lite-residual.
8. ⭐⭐ **On-Device Training Under 256KB Memory** (Lin 2022, NeurIPS) — quantization-aware scaling + sparse update. **Đọc kỹ nhất.**
9. TinyOL (Ren 2021) — online learning trên MCU, đơn giản để replicate đầu tiên.

**Lớp 3 — Nền Continual Learning (chống catastrophic forgetting)**
10. EWC — Overcoming Catastrophic Forgetting (Kirkpatrick 2017).
11. iCaRL (Rebuffi 2017) — class-incremental + replay.
12. Three Scenarios for Continual Learning (van de Ven 2019) — khung phân loại bài toán.

**Lớp 4 — Survey để tìm gap**
13. Survey On-Device Learning / TinyML gần đây (2023–2025, Scholar của Roveri / Song Han) — soi "open challenges".
14. Survey Deep Learning for Sensor-based HAR (Wang/Chen) — nối về dữ liệu IMU.

**Cách đọc hiệu quả:** đọc kỹ 3 bài ⭐ (MCUNet, On-Device 256KB, TinyTL). 11 bài còn lại chỉ đọc
abstract + intro + figures + **phần "Limitations/Future Work"** — đó là nơi tác giả tự khai gap = nguyên liệu đề tài.
*Lưu ý: cập nhật paper 2025–2026 mới nhất qua Google Scholar (cutoff kiến thức người tư vấn = 01/2026).*

---

## 6. Đề tài cụ thể từ project SisFall (ý a)

### Bối cảnh tài sản đã có (rất mạnh)
- Bài toán: fall detection IMU 6 kênh, 5 nhãn (Walk, Run, Idle, Trans, Fall), window 200 mẫu @100Hz.
- Tiến hóa: TCN (v16–v22) → ResNet-1D stride (v23) → **SeparableConv ResNet-1D (v25)**.
- **v25 trên ESP32-S3:** inference **70ms**, tensor arena **28.2KB**, Fall Recall **99%**,
  gap desktop→firmware chỉ **−0.68%**, model ~504KB, 19K params. ESP-NN/SIMD tối ưu hoàn toàn.
- Hạ tầng sẵn: KFold experiments (`train_v25_kq/SisFall_KFold_Experiments_v2.ipynb`),
  firmware inference (`sis_fall_firmware_inference/`, esp-tflite-micro), pipeline INT8 ổn định.

### Quan sát chìa khóa
Suốt v20→v26, **Trans bị nhầm Idle dai dẳng** (F1-Trans kẹt 0.83–0.85) dù đã thử dilated RF, SE block,
focal loss, augment. → Đã vắt kiệt hướng "sửa bằng kiến trúc/loss". Tín hiệu: **vấn đề ở phân bố
dữ liệu, không phải model** — nghi ngờ **biến thiên giữa người dùng** (tốc độ chuyển tư thế khác nhau
theo từng người ⇒ ranh giới Trans/Idle dịch theo subject). Một model global không chiều được tất cả.
→ Đây là cây cầu thẳng sang hướng **On-Device Personalization / Continual Learning**.

### 🎯 Bài báo #1 (chủ lực) — On-device personalization dưới ràng buộc bộ nhớ
**Câu hỏi:** Fine-tune trên thiết bị một tập tham số *thưa*, nằm trong ngân sách RAM ESP32-S3, có thu hồi
được phần độ chính xác mất do inter-subject variability (đặc biệt ranh giới Trans/Idle) mà không cần
cloud không? Pareto front giữa accuracy-gain và trainable-memory là gì?

**Giả thuyết:** confusion Trans/Idle phần lớn là inter-subject, không intrinsic. Vài chục window/người
+ cập nhật tham số thưa (kiểu On-Device 256KB / TinyTL bias-only) đóng được phần đo lường được của gap,
vẫn vừa 28KB arena.

**Thí nghiệm tối thiểu (chạy ngay trên Colab/GPU, chưa cần backprop trên MCU):**
1. **Đo để chứng minh giả thuyết:** Leave-One-Subject-Out (LOSO) trên v25 (đã có hạ tầng KFold).
   Báo cáo phương sai theo subject của F1-Trans. Phương sai lớn → giả thuyết đứng vững (đã là finding đăng được).
2. **Adapt:** mỗi subject giữ-lại, lấy K window/lớp làm calibration (K=5,10,20 → vẽ đường cong),
   fine-tune chỉ tập con tham số nhỏ, đo lại F1-Trans/Idle.
3. **So sánh chiến lược adapt theo chi phí bộ nhớ** (phần co-design):
   Full FT · Last-Dense only · Bias-only (TinyTL) · Sparse update (On-Device 256KB) · chỉ SE-block scale.
   Vẽ **accuracy-gain vs trainable-memory (KB)** → chỉ điểm nào *thực sự vừa* ngân sách ESP32-S3.
4. **Đóng đinh phần cứng:** với chiến lược thắng, ước tính/đo RAM + latency một bước update trên ESP32-S3.
   Cặp số "gain X% với +Y KB RAM" là đóng góp 95% paper fall-detection không có.

**Vì sao đăng được:** ghép (1) finding inter-subject + (2) Pareto co-design + (3) số đo MCU thật.
Đủ cho workshop tốt (tinyML/SenSys/IPSN) ngay; mở rộng thành full paper khi thêm on-device backprop thật.
**Future work của chính bài:** continual — chống catastrophic forgetting khi adapt liên tục (EWC/iCaRL).

### 🪜 Bài báo #2 (an toàn, gặt nhanh) — Ablation co-design ESP-NN-aware
**Đã có sẵn data** trong `firmware_deployment_espnn_analysis.md`: v22 (TCN dilated, tốt nhất PC nhưng
không chạy ESP-NN) → v23 (bỏ dilation) → v25 (SepConv k=3). Nghiên cứu định lượng **"cái giá của tính
tương thích phần cứng"**: hy sinh ~0.55% macro-F1 (v22→v25) để đổi inference >300ms → 70ms + bật SIMD.
Rất ít paper đo trade-off này trên *cùng bài toán, cùng board, số thật*. Viết được trong 2–3 tuần từ dữ liệu đã có.

### Thứ tự khuyến nghị
Viết **#2 trước** (gặt nhanh, lấy publication đầu + tự tin), **song song chạy LOSO cho #1**
(bài định vị researcher thật).

### 🚧 Trạng thái triển khai bài #1 (đang làm)
Pipeline thí nghiệm LOSO đã được dựng trong thư mục [`loso/`](loso/):
- `loso/README_LOSO_methodology.md` — thiết kế khoa học: 2 thí nghiệm (A: chứng minh
  inter-subject variance; B: personalization dưới ngân sách bộ nhớ), **5 chốt chặn đúng đắn**
  (tách calibration/test theo *trial* chống leakage; không đưa Fall vào calibration;
  Fall Recall không được giảm; nhiều seed; đóng băng preprocessing), và ánh xạ sang luận điểm paper.
- `loso/loso_pipeline.py` — script runnable, tái dùng `parse_filename_info` + preprocessing
  fixed-scale + kiến trúc ResNet-1D v26. 5 chiến lược adapt theo chi phí bộ nhớ tăng dần:
  `last_dense → norm_tuning (TinyTL) → se_only → last_block → full (upper bound)`.

**Lưu ý vận hành:**
- Máy local chỉ để *học*; **training chạy trên máy GPU**. Dataset windowed local đã lỗi thời
  (chỉ Walk/Run/Fall) → khi chạy LOSO phải trỏ `--data-dir` về **nguồn 5 lớp mới trên máy GPU**.
- **Bước tiếp theo:** chạy pilot (`--subjects SA22 SE12 --epochs 20`) để smoke-test → nếu giả thuyết
  inter-subject đứng vững thì (1) chạy full 38 subject, (2) thêm chiến lược `sparse_topk`
  (kiểu *On-Device 256KB*, viết bằng GradientTape) làm điểm novelty nâng workshop → full paper.

---

## 7. Nguyên tắc xuyên suốt
1. Lợi thế = deployment + đo phần cứng thật. Đừng đua lý thuyết.
2. Deployment là bàn đạp; phải gắn vào trục nghiên cứu dọc.
3. Một trục dọc (applied → method → systems), không phân mảnh.
4. Project chạy thật trên MCU > certificate.
5. Gộp ngoại ngữ + embedded vào ML để tiết kiệm quỹ giờ.
6. Săn phần "Limitations/Future Work" của paper = nguyên liệu đề tài.
