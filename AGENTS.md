# AGENTS.md — SisFall Fall Detection / HAR (tinyML cho ESP32-S3)

> Rules dùng chung cho mọi AI agent (Claude Code, Antigravity, …).
> Mục tiêu dự án: huấn luyện mạng nhận diện **té ngã + hoạt động** từ dữ liệu SisFall (accel + gyro),
> nén INT8 và **chạy trên vi điều khiển ESP32-S3** qua TensorFlow Lite Micro (TFLM) + ESP-NN.

## 1. Bức tranh tổng thể

Pipeline: `raw .txt (SisFall)` → **tiền xử lý** → **cắt cửa sổ (windowing)** → **cache .npy** → **train** → **export INT8 .tflite** → **`model_data.cc/.h`** → firmware ESP32-S3.

- **5 nhãn:** `['Walk', 'Run', 'Idle', 'Trans', 'Fall']` (Fall = index 4). Lie + StandSit gộp thành Idle.
- **Cửa sổ:** 200 mẫu × 6 kênh (ax,ay,az,gx,gy,gz), 100Hz (đã downsample từ 200Hz).
- **Chỉ số ưu tiên số 1: Fall recall** (bỏ sót té ngã tệ hơn báo động giả). Đánh giá dùng `fall_threshold = 0.25`.
- **Split subject-independent** (KHÔNG trộn mẫu cùng người qua train/test):
  - TRAIN: `SA01–18` + `SE01–08`  | VAL: `SA19–21` + `SE09–11`  | TEST: `SA22–23` + `SE12–15`
  - `SA*` = người trẻ, `SE*` = người già.

## 2. Cấu trúc repo

- `ml_pipeline.py` — module dùng chung: `DataPreprocessor` (load/cache/split/scale) + `OutputReporter` (history, confusion matrix, report). Notebook/script kế thừa 2 class này.
- `step1_extract_and_preprocess.py`, `step2_windowing.py`, `windowing_pipeline.py` — pipeline dạng script.
- `export_tflite_with_ops.py` — convert INT8 + sinh `model_data.cc/.h` **có nhúng sẵn ops + quant + gợi ý MicroMutableOpResolver** (để người viết firmware không phải đoán op).
- `train_vXX_kq/` — mỗi thí nghiệm 1 folder (`_kq` = kết quả). Chứa notebook/script + `report_*.txt` + `confusion_matrix_*.png` + `best_model_*.keras` + `model_*_int8.tflite` + `model_data_*.cc/.h`.
- `sis_fall_firmware_inference/` — project ESP-IDF (firmware ESP32-S3, TFLM + ESP-NN trong `managed_components/`).
- `SisFall_dataset/` — dữ liệu thô (read-only). `venv/`, `train_cache*/` — KHÔNG sửa, KHÔNG commit nội dung lớn.

### Dòng kiến trúc đã thử (để biết bối cảnh)
- `train_v1_*resize*` — CNN-LSTM thu nhỏ (LSTM 64→32, 32→16, 1 lớp).
- `train_v8`–`train_v22` — TCN (dilated conv). **v22 là TCN tốt nhất** nhưng ~100ms/infer.
- `train_v25` — ResNet1D (SeparableConv + SE block).
- `train_v27` — CNN thuần tối ưu ESP-NN. `train_v22_optimize` — TCN bỏ dilation cho ESP-NN.

## 3. Môi trường & cách chạy

- **Server** (GPU chung, RTX 2080 Ti): SSH, env conda **`har_fall`**, **Python 3.11**.
  - Workspace + cache + dataset ở: `/media/data3/users/hungvm/dataset/sisfall/`
- **Deps:** cài bằng `pip install -r requirements_kfold.txt` (đã đầy đủ: TF 2.21, tf-keras, sklearn, seaborn, jupyter…).
- **Train nền (giữ tiến trình khi rớt mạng):** `nohup python train_xx.py > train.log 2>&1 &` rồi `tail -f train.log`.
- **GPU dùng chung — bắt buộc nhớ:**
  - Bật memory growth để không ôm hết VRAM: `for g in tf.config.list_physical_devices('GPU'): tf.config.experimental.set_memory_growth(g, True)`.
  - **Train xong PHẢI Shut Down kernel** (đóng tab không tắt kernel → ôm GPU). Chỉ chạy 1 kernel GPU/lúc.
  - Model nhỏ (<50k params) bị OOM do GPU bận → ép CPU: `os.environ['CUDA_VISIBLE_DEVICES']='-1'` ở cell đầu (trước import TF) + Restart Kernel.
  - Batch nhỏ làm GPU đói (util 30–50%) với model tí hon → dùng `batch_size=256` (2080 Ti cân được), **tăng LR kèm theo** (≈ ×2–4).

## 4. ⚠️ Gotchas BẮT BUỘC tuân thủ

### 4.1. LSTM/GRU → cần Keras 2 để nhúng được vào TFLM
- Keras 3 (mặc định) convert LSTM ra op `WHILE` → **segfault khi int8 + TFLM không chạy**.
- Mọi notebook có LSTM **phải** đặt `os.environ['TF_USE_LEGACY_KERAS'] = '1'` **TRƯỚC khi import tensorflow** và cài `tf-keras` → LSTM fuse thành `UNIDIRECTIONAL_SEQUENCE_LSTM`.
- Khi convert: clone model sang `batch_shape=(1,200,6)` (shape tĩnh) cho chắc fuse; sau convert phải kiểm tra ops có `UNIDIRECTIONAL_SEQUENCE_LSTM` và KHÔNG có `WHILE`.
- **GRU:** TFLM KHÔNG có kernel fused → thường ra `WHILE` (giống bẫy trên). **Test export trước khi đầu tư.**

### 4.2. Thiết kế mạng phải bám tập lệnh ESP-NN tăng tốc (mục tiêu: latency thấp trên ESP32-S3)
ESP-NN **tăng tốc**: Conv 1×1 pointwise (**14×**), relu6 (**11×**), MaxPool/FullyConnected (**~8×**), Depthwise 3×3 (**6×**), Conv 3×3 (**5.5×**), mean/GAP, elementwise add/mul.
ESP-NN **KHÔNG tăng tốc** (→ chạy reference chậm): **LSTM/GRU**, **dilated conv (dilation>1)**, **sigmoid/tanh**, hard_swish.
- → Mạng tối ưu: **depthwise-separable Conv1D (k=3) + pointwise 1×1 + relu6 + MaxPool downsample + GAP + Dense**.
- → **Tránh** trong nhánh tối ưu tốc độ: LSTM, dilated conv (TCN cổ điển chậm vì cái này), SE block (dùng sigmoid + phình metadata).
- Conv1D → TFLite map sang Conv2D (H=1) nên vẫn được tăng tốc.

### 4.3. Dữ liệu & cache
- **Windowing rất chậm** (sinh hàng chục nghìn file CSV nhỏ; trên disk mạng càng chậm). **Dùng lại cache `.npy`, KHÔNG cắt window lại** nếu cache đã có.
- Cache `.npy` **chỉ phụ thuộc windowing, không phụ thuộc kiến trúc** → nhiều thí nghiệm cùng windowing **dùng chung 1 cache** (vd `cache_resize_64_32` cho dòng 5-label resize/v27).
- **KHÔNG dùng chung thư mục windowed giữa các logic windowing khác nhau** (vd v25/KFold dùng decimate + D09/D10/D14 → đặt namespace riêng `*_v25`/`*_v27`), nếu không file cũ sót lại sẽ bị load nhầm.
- Scaling (trong `apply_preprocessing`): accel `clip(-8,8)/8`, gyro `/2000`. Tiền xử lý: accel `×32/8192`, gyro `×(4000/65536)×(π/180)`, downsample 200→100Hz.

### 4.4. Kích thước model INT8
- Với mạng **nhiều tensor nhỏ** (SeparableConv + SE như v25), **metadata phình** → file INT8 lớn hơn cả float weight (v25: 19k params nhưng 80KB, ~75% là overhead). Param count KHÔNG dự đoán được tflite size.
- Muốn nhỏ flash: **ít tensor to** thắng **nhiều tensor nhỏ**. TFLite KHÔNG nén (flatbuffer thô) — đừng đổ lỗi "thuật toán nén".

## 5. Quy ước khi tạo thí nghiệm mới

1. Tạo folder `train_vXX[_mô_tả]/`. Clone notebook/script gần nhất.
2. Chỉ đổi: `build_model` (kiến trúc), `self.version`, `out_dir`. **Giữ nguyên `cache_dir` để dùng lại cache.**
3. Nếu chỉ đổi kiến trúc (không đổi windowing) → **bỏ/skip cell tiền xử lý + windowing**, để `load_or_create_dataset()` nạp thẳng từ cache.
4. Export dùng kiểu **nhúng ops** (`export_tflite_with_ops.py` hoặc cell tương đương) — sinh `model_data_<ver>.cc/.h` có sẵn danh sách ops + quant + gợi ý resolver.
5. Ghi kết quả vào bảng so sánh: **params · tflite size (KB) · accuracy (float & INT8) · Fall recall · Trans F1 · inference ms (ESP32-S3)**.

## 6. Đừng (Don'ts)

- ❌ Đừng commit `venv/`, `SisFall_dataset/`, file `.npy` cache lớn, hay `build/` firmware.
- ❌ Đừng sửa dữ liệu thô `SisFall_dataset/` hay xoá cache đang dùng chung.
- ❌ Đừng thêm LSTM/GRU/dilated-conv/sigmoid vào nhánh "tối ưu ESP-NN" mà không cảnh báo về latency.
- ❌ Đừng train rồi để kernel GPU sống (ôm VRAM của người khác trên server chung).
- ❌ Đừng đánh giá chỉ bằng accuracy tổng — **luôn báo cáo Fall recall + Trans F1** (accuracy bị lớp đa số che lấp).
- ❌ Đừng kết luận size/accuracy từ model float — số **thật là từ `.tflite` INT8** chạy trên chip.

## 7. Giao tiếp
- Người dùng trao đổi bằng **tiếng Việt**. Trả lời tiếng Việt; giữ thuật ngữ kỹ thuật + lệnh + path bằng tiếng Anh.
