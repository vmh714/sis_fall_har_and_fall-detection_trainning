# 🌳 Phân tích Cây tiến hóa (Evolution Tree) Model SisFall (v2 -> v24)

Dựa trên việc phân tích chi tiết tài liệu `instruction.md` của các phiên bản, đây là **Sơ đồ tiến hóa và kế thừa** của quá trình nghiên cứu và tối ưu hóa kiến trúc model nhận dạng hành động (HAR) và té ngã (Fall Detection). 

Các số hiệu version (`v2`, `v10`...) không hoàn toàn mang tính tuần tự tuyến tính mà có những thời điểm phân chia thành các "nhánh" thử nghiệm độc lập (6-channel, 7-channel...) trước khi hội tụ lại ở các phiên bản hoàn chỉnh nhất.

---

## 🟢 1. Kỷ nguyên Khởi đầu: Hybrid CNN-LSTM (v2 → v4)
**Mục tiêu:** Xây dựng Baseline (4 Classes) với phương pháp Deep Learning truyền thống.

- **`v2`, `v3`, `v4`:** Sử dụng kiến trúc lai (Hybrid).
  - Dùng **CNN** (Conv1D) để trích xuất đặc trưng không gian cục bộ.
  - Dùng **LSTM** để học tính chuỗi thời gian.
- **Đóng góp quan trọng:** Định hình chiến lược Data Split **LSO (Leave-Subject-Out)** và cửa sổ cố định **200 timesteps**, thứ được giữ nguyên cho tới tận v24.

---

## 🟡 2. Kỷ nguyên Phân nhánh "Chuyên gia": Trunk-and-Branches (v5 → v7)
**Mục tiêu:** Tách biệt đặc trưng nhịp điệu (HAR) và đặc trưng đột biến (Fall Detection). Không kế thừa kiến trúc từ v4 mà thiết kế lại theo hướng Two-Stream.

- **`v5`, `v6`:** Loại bỏ cấu trúc tuần tự đơn thuần, chuyển sang **Late-branching**:
  - **Nhánh Fall Expert:** Dùng `GlobalMaxPooling1D` chuyên bắt tín hiệu đột biến.
  - **Nhánh HAR Expert:** Dùng `LSTM` để bắt nhịp điệu.
- **`v7`:** *Kế thừa v6*, nhưng thực hiện Feature Engineering dữ dội: bổ sung tính toán **Pitch, Roll, và SVM** (Signal Vector Magnitude), nâng input từ 6 kênh lên **9 kênh**.

---

## 🔵 3. Kỷ nguyên Chuyển giao thiết bị nhúng: TCN-MCU Baseline (v8 → v11)
**Mục tiêu:** Nhắm đến vi điều khiển (MCU), cần TFLite INT8 Quantization. LSTM quá nặng nên bị loại bỏ hoàn toàn.

- **`v8`, `v9`, `v10`, `v11`:** Chuyển sang mạng **TCN (Temporal Convolutional Network)**.
  - Sử dụng **Dilated Convolutions** `[1, 2, 4, 8]` để mở rộng trường nhìn (receptive field) mà không tăng tham số.
  - Áp dụng **Causal Padding** để mô hình hóa chuỗi thời gian.
  - Lớp gộp sử dụng `GlobalAveragePooling1D`.

---

## 🟠 4. Kỷ nguyên Tối ưu hóa TFLite & Thử nghiệm song song (v12 → v14)
**Mục tiêu:** Tối ưu triệt để cho toán tử của MCU (tránh lệnh `Pad`) và thử nghiệm thêm bớt các kênh đặc trưng.

- **`v12`, `v14` (Trục chính):** *Kế thừa v11*. Cải tiến cốt lõi là thay `Causal Padding` bằng `padding='valid'` kết hợp `Cropping1D` ở các nhánh Residual, loại bỏ hoàn toàn toán tử `Pad` gây chậm trên TFLite.
- **Các nhánh thử nghiệm song song (Rẽ nhánh):**
  - **Nhánh `6ch` (`v12_6ch`, `v12_6ch_opt`, `v14_6ch`):** Giữ nguyên 6 trục, thêm `GaussianNoise(0.05)` để tăng tính bền vững.
  - **Nhánh `7ch` (`v12_7ch`, `v14_7ch`):** Tích hợp thêm kênh **SVM** vào dữ liệu thô (thành 7 kênh input) thay vì tính toán 9 kênh phức tạp như v7.
  - **Nhánh `v13`:** *Đột biến từ v12.* Thử nghiệm thay đổi Average Pooling thành `GlobalMaxPooling1D` và dùng thêm `Label Smoothing`. Tuy nhiên bản `v14` đã *rollback (loại bỏ)* Label Smoothing để giữ mô hình nhẹ nhất.

---

## 🟣 5. Kỷ nguyên Mở rộng & Tái cấu trúc Classes (v15 → v21)
**Mục tiêu:** Mở rộng khả năng nhận dạng (nhiều hơn 4 lớp) và khắc phục nhược điểm nhận dạng các trạng thái tĩnh/động.

- **Giai đoạn Mở rộng 6 Lớp (`v16`, `v17`):** Tách nhãn tĩnh thành 2 lớp riêng biệt: **StandSit** và **Lie** (tổng cộng 6 classes).
- **Giai đoạn Tối ưu 5 Lớp (`v18` → `v21`):** *Nhận ra 6 lớp quá dư thừa hoặc khó hội tụ*, tiến hành gộp lại thành 5 lớp: Gộp StandSit + Lie về chung nhãn **Idle**.
- **Cải tiến kiến trúc cốt lõi (`v20`, `v21`):** *Học hỏi từ kỷ nguyên v5 (Two-Stream).* 
  - Đưa cơ chế bắt đặc trưng kép vào mạng bằng cách kết nối (`Concatenate`) cả **GAP (`GlobalAveragePooling1D`)** và **GMP (`GlobalMaxPooling1D`)**.
  - `v21`: Tăng Kernel size lên 5.

---

## 🚀 6. Kỷ nguyên Hiện đại: Attention & Strided Residual (v22 → v24)
**Mục tiêu:** Đạt trạng thái tinh gọn, thông minh và được "đo ni đóng giày" cho phần cứng. Lột xác hoàn toàn cấu trúc TCN truyền thống.

- **`v22` (Bổ sung Attention):** *Kế thừa v21*. Bổ sung **SE Block (Squeeze-and-Excitation)** để mạng tự học sự chú ý (Attention) trên các kênh không gian. Lấy lại công nghệ `Label Smoothing (0.1)`.
- **`v23` (Đại tu TCN):** *Kế thừa v22*. Quyết định cực lớn: **Loại bỏ hoàn toàn Dilation** (thứ vốn là bản sắc của TCN). Chuyển sang nén không gian bằng **`strides=2`** với **Kernel size lớn (7)**. Mô hình bây giờ mang hình hài của *Strided Residual Conv1D* hơn là TCN cổ điển.
- **`v24` (Hoàn thiện - State of the Art hiện tại):** *Kế thừa v23*. 
  - Gắn cứng bộ quy tắc tiền xử lý sát với vật lý thực tế ngay trong thân script: **Cắt (Clip) tín hiệu gia tốc ở [-8.0, 8.0]g** để lọc nhiễu cực đoan, chia Scale chuẩn mực (Accel / 8.0, Gyro / 2000.0).
  - Áp dụng hàm kích hoạt `relu6` chuyên dụng cho vi điều khiển.
  - Thêm `L2 Regularization` (1e-4) chống overfitting.

---
### 📌 Tóm tắt Dòng chảy chính (Mainline Lineage):
**Baseline (v4)** ➔ **Late-Branching (v6)** ➔ **TCN Baseline (v11)** ➔ **TFLite Opt (v12)** ➔ **5-Class GAP+GMP (v20)** ➔ **SE Block (v22)** ➔ **Strided Conv1D (v23)** ➔ **Pre-Processed & Relu6 (v24)**.
