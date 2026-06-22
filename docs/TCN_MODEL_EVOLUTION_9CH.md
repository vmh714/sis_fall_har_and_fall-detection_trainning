# Lịch Sử Huấn Luyện & Tối Ưu Hóa Mô Hình TCN (Dataset 9 Kênh Chuẩn - Không La Bàn)

Tài liệu này tổng hợp quá trình tinh chỉnh (tuning) kiến trúc Temporal Convolutional Network (TCN) cho bài toán nhận diện hành vi (HAR) và phát hiện té ngã (Fall Detection) dựa trên dataset **9 kênh chuẩn** (gồm: 6 trục IMU thô, 2 góc Pitch/Roll, và biên độ SVM).

Mục tiêu tối thượng là đạt Recall > 95% cho lớp Fall (tiêu chuẩn an toàn y tế) trong khi giữ được sự mượt mà và chính xác giữa các lớp hành vi liên tục (Walk, Run, Static/ADL).

---

## 1. Bảng So Sánh Các Phiên Bản Huấn Luyện (v11 - v15)

| Chỉ số / Lớp | v11 | v12 (9ch) | v12 (6ch) | v12 (7ch) | v13 | v14 (9ch) | v14 (6ch) | v14 (7ch) | v15 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy Tổng** | 91.40% | 89.87% | 91.86% | 91.30% | 85.49% | **92.16%** | 89.44% | 90.20% | 91.54% |
| **Walk Recall** | 83.60% | **85.42%** | 84.88% | 82.00% | 79.30% | 85.11% | 85.51% | 83.51% | *85.68%* |
| **Walk Precision**| **90.11%** | 82.63% | 90.08% | 88.99% | 89.00% | 88.97% | 85.62% | 90.62% | 87.94% |
| **Static Recall** | 93.29% | 88.22% | 92.60% | 93.27% | 84.35% | **93.74%** | 88.12% | 90.53% | 92.84% |
| **Fall Recall** | 98.13% | 98.13% | 97.73% | **98.93%** | 94.00% | 97.73% | 96.93% | 96.93% | 96.80% |
| **Fall Precision**| 88.04% | 91.20% | 90.38% | 91.49% | 61.73% | **94.83%** | 80.60% | 78.51% | 91.21% |

---

## 2. Chi Tiết Các Thử Nghiệm & Bài Học Rút Ra

### 🧱 Phiên bản v11: Bản khởi điểm
- **Kết quả:** Đạt Accuracy khá tốt (91.40%) và Fall Recall vượt chuẩn (98.13%).
- **Nhận xét:** Một sự khởi đầu vững chắc với bộ dữ liệu 9 kênh mới, cho thấy dữ liệu có chất lượng phân cụm rất tốt. Tuy nhiên Precision của Fall hơi thấp (88.04%) đồng nghĩa với việc vẫn còn tỷ lệ báo động giả (False Positives).

### 📉 Phiên bản v12: Baseline (9ch, 6ch, 7ch)
- **Thay đổi:** So sánh giữa bản 9 kênh (Baseline), 6 kênh (chỉ IMU gốc) và 7 kênh (IMU + SVM).
- **Đánh giá 9ch:** Việc nới lỏng cơ chế Dropout/Class weights khiến mô hình bị xáo trộn, Accuracy tổng giảm xuống 89.87%, đặc biệt là Static Recall tụt mạnh xuống 88.22%.
- **Đánh giá 6ch/7ch:** Cả 6ch và 7ch lại cho kết quả rất tốt (Accuracy trên 91%, Fall Recall cực cao - đặc biệt v12 7ch đạt Recall tới 98.93%). Điều này chứng tỏ đôi khi việc giảm bớt các đặc trưng tính toán thủ công (pitch, roll) lại giúp mô hình bớt nhiễu nếu các siêu tham số (hyperparameters) không được ép chuẩn.

### ❌ Phiên bản v13: Thất bại với Max Pooling & Label Smoothing
- **Thay đổi:** Đổi Pooling thành `GlobalMaxPooling1D` và giữ nguyên Label Smoothing.
- **Kết quả:** Tương tự như các thử nghiệm trong lịch sử, kỹ thuật này hoàn toàn phá hỏng mô hình trên dữ liệu mới. Accuracy rớt thảm hại xuống 85.49%, Fall Recall không đạt chuẩn (94.00%) và kinh hoàng nhất là Fall Precision bốc hơi chỉ còn 61.73%.
- **Bài học (tái khẳng định):** Việc sử dụng MaxPooling trên chuỗi thời gian làm mất đi bức tranh liên tục của toàn bộ sự kiện tiêu hao năng lượng, là kỹ thuật cực kỳ tối kỵ với bài toán HAR và Fall Detection.

### 🏆 Phiên bản v14: Sự trỗi dậy của Class Weights (NHÀ VÔ ĐỊCH)
- **Thay đổi:** Trở lại với `GlobalAveragePooling1D`, sử dụng Class Weights phạt nặng (Walk x1.3, Fall x3.0). Thử nghiệm trên 3 biến thể đầu vào: 9ch (đầy đủ), 6ch (chỉ IMU) và 7ch (IMU + SVM).
- **Đánh giá 9 kênh (Nhà vô địch):** Trái ngược với kết quả của bộ dữ liệu cũ (nơi v14 bị học lệch), trên **dataset 9 kênh chuẩn này**, cấu hình v14 đã mang lại điểm cân bằng hoàn hảo nhất. Accuracy đạt đỉnh **92.16%**, Fall Recall giữ vững mức an toàn cực cao (97.73%), đồng thời Fall Precision xuất sắc vươn lên **94.83%** triệt tiêu gần như hoàn toàn sự nhầm lẫn (False Positive).
- **Đánh giá 6 kênh & 7 kênh:** Rất bất ngờ! Khi loại bỏ góc Pitch/Roll (tức dùng 6ch và 7ch), hiệu năng của v14 **giảm sút nghiêm trọng**. Accuracy giảm xuống quanh mốc 89-90%. Đặc biệt, Precision của lớp Fall tụt dốc thê thảm từ 94.83% xuống chỉ còn **~78-80%** (báo động giả tăng vọt).
- **Bài học cốt lõi:** Khi bị ép học bằng Class Weights cực đoan để bắt lớp Fall, mô hình *bắt buộc phải bấu víu vào các đặc trưng định hướng không gian (Pitch/Roll)* để phân biệt rạch ròi giữa hành động đứng/đi với việc ngã nằm dưới đất. Nếu tước đi Pitch/Roll, mô hình bị "mù phương hướng" và hoảng loạn, dẫn đến việc nhìn đâu cũng thấy ngã. Vì thế **v14 (9 kênh)** mới là chìa khóa cuối cùng.

### ⚠️ Phiên bản v15: Thừa thãi tầng Dense phi tuyến
- **Thay đổi:** Bổ sung lớp giải mã `Dense(64, activation='relu')` trước lớp Softmax đầu ra.
- **Đánh giá:** Mô hình vẫn duy trì phong độ tốt (Accuracy 91.54%, Fall Recall 96.80%) nhưng kém v14 về mọi mặt. Sự phức tạp hóa kiến trúc bằng lớp phi tuyến (Dense) không mang lại lợi ích bứt phá nào, ngược lại còn làm giảm khả năng nhận diện té ngã nhạy bén và đẩy Precision của lớp Fall tụt xuống 91.21%.

---

## 3. 🚨 Case Study: "Lệch pha tiền xử lý dữ liệu dẫn đến tràn số khi lượng tử hóa INT8"

Để đưa vào cuốn thuyết minh khóa luận hoặc trình bày trước hội đồng bảo vệ, đây là cách đóng gói lỗi kinh điển này một cách học thuật và chuyên nghiệp theo cấu trúc "Vấn đề - Nguyên nhân - Giải pháp".

### 3.1. Tên lỗi học thuật (Technical Term)
**"Lệch pha tiền xử lý dữ liệu dẫn đến tràn số khi lượng tử hóa INT8"** *(Preprocessing Mismatch causing INT8 Quantization Saturation)*

### 3.2. Biểu hiện (Symptom)
* **Trên PC (Mô phỏng Float32):** Mô hình đạt độ chính xác cực cao (Accuracy > 91%), nhận diện xuất sắc cả 5 lớp hành vi.
* **Trên Firmware ESP32-S3 (Bản cũ - Trước khi xử lý):** Độ chính xác sụt giảm nghiêm trọng. Hệ thống phân loại mất kiểm soát, mô hình dường như chỉ đang "đoán mò".
  - **Độ chính xác tổng (Accuracy):** Rớt thảm hại chỉ còn **39.83%**.
  - **Hành vi đi bộ (Walk):** Bị xóa sổ hoàn toàn (Precision 15.46%, Recall 15.00%).
  - **Nhận diện té ngã (Fall Recall):** Chỉ đạt **67.00%**, tỷ lệ sai số trầm trọng gây nguy hiểm tính mạng nếu triển khai thực tế.

### 3.3. Nguyên nhân gốc rễ (Root Cause)
Lỗi xảy ra do sự bất đồng bộ trong Data Pipeline giữa môi trường huấn luyện (Python) và môi trường thực thi (C/C++ trên RTOS):
* **Bản chất lượng tử hóa (Quantization):** Khi chuyển đổi mô hình từ Float32 sang INT8 (Post-Training Quantization), TFLite tính toán các tham số `scale` và `zero_point` dựa trên phân phối của dữ liệu huấn luyện. Dữ liệu này đã được chuẩn hóa về một khoảng biên độ rất hẹp (kẹp trong khoảng `±8.0` rồi chia tỷ lệ để xoay quanh mốc `[-1, 1]`). Lớp dữ liệu đầu vào (Input Tensor) bắt buộc phải dùng cơ chế **Per-Tensor Quantization** (dùng chung 1 hệ số Scale cực nhỏ `~0.0078` cho TOÀN BỘ 6 trục).
* **Thiếu hụt tiền xử lý trên Firmware:** Khi đưa vào chạy thực tế, vi điều khiển nhận luồng dữ liệu thô (raw data) có biên độ khổng lồ. Gia tốc kế (ADXL345) có dải đo `±16g`, con quay hồi chuyển (ITG3200) có dải đo tới `±2000°/s`.
* **Hiện tượng Tràn số (Saturation):** Khi đưa các giá trị thô khổng lồ này vào công thức ép kiểu: `round(raw_value / scale) + zero_point`, kết quả trả ra vượt quá xa giới hạn của kiểu dữ liệu 8-bit có dấu. Toàn bộ mảng đặc trưng bị kẹp cứng (clamped) ở hai giá trị biên là `-128` hoặc `127`. Tín hiệu động lực học của cảm biến bị "phẳng hóa" hoàn toàn, khiến mô hình bị mù thông tin đặc trưng (feature loss).

### 3.4. Giải pháp khắc phục (Resolution)
Tái tạo chính xác chu trình chuẩn hóa dữ liệu (Global Min-Max Scaling) ngay bên trong Firmware vi điều khiển trước khi đưa vào Node đầu vào của TFLite Micro. Cụ thể:
1. **Phân tách luồng xử lý:** Tách rõ ràng các trục của Gia tốc kế (Accel) và Con quay hồi chuyển (Gyro) trong mảng dữ liệu C++.
2. **Kẹp giới hạn phần cứng (Clipping):** Kẹp giới hạn cho Accel ở mức `±8.0g` để mô phỏng sự quá tải phần cứng khi va chạm mạnh.
3. **Phép chia tỷ lệ:** Thực hiện chia tỷ lệ (chia 8.0 cho Accel và chia 2000.0 cho Gyro) trên kiểu dữ liệu số thực (Float) để gộp cả 6 trục về chung một hệ quy chiếu biên độ `[-1.0 : 1.0]` trước khi lượng tử hóa INT8.

### 3.5. Kết Cục Của Case Study: Sự Vươn Mình Ngoạn Mục
Sau khi cấy giải pháp tiền xử lý này trực tiếp vào `tflite_wrapper.cpp`, mô hình bản **v24** trên Firmware đã ghi nhận sự lột xác thần kỳ:
- **Độ chính xác tổng (Accuracy):** Nhảy vọt lên **91.17%**, đồng nhất hoàn hảo với kết quả Float32 ban đầu trên máy tính.
- **Hành vi đi bộ (Walk):** Phục hồi ngoạn mục với F1-Score **90.72%**.
- **💥 Lớp Té Ngã (Fall):** Đạt ngưỡng **TỐI ĐA THẦN THÁNH - Precision: 100% | Recall: 100%**. 
=> *Tuyệt đối không bỏ sót một cú ngã nào, và hoàn toàn triệt tiêu báo động giả (Zero False Positives).*

**Bài học đắt giá:** Lịch sử tiến hóa của case study này là minh chứng hoàn hảo cho triết lý: **Trong TinyML, việc thấu hiểu ranh giới vật lý của cảm biến (Sensor Physical Limits) và đồng bộ Data Pipeline quan trọng ngang ngửa với việc thiết kế kiến trúc Deep Learning.**

---

## 5. Chiến Lược Triển Khai Thực Tế (MCU / C++)

Dựa trên chuỗi thực nghiệm toàn diện với tập dữ liệu chuẩn 9 kênh (Không la bàn) và khắc phục thành công yếu điểm Quantization, chúng ta thống nhất hệ thống phần mềm nhúng để cài đặt lên thiết bị đeo (Wearable) như sau:

> [!IMPORTANT]  
> **Chốt sổ Kiến trúc & Trọng số:** Sử dụng cấu hình và tệp trọng số `.keras` / `.tflite` của phiên bản **v24**. Đảm bảo thuật toán chia tỷ lệ biên độ (Max Range Scaling) đã được hard-code vào hàm inference C++.

> [!TIP]  
> **Kỹ Thuật Hậu Xử Lý (Post-Processing) Bắt Buộc:**
> 
> 1. **Luồng Khẩn Cấp (Fall):** Theo dõi xác suất từ đầu ra mô hình liên tục. Ngay khi `Prob_Fall >= 0.25`, BỎ QUA MỌI CHỜ ĐỢI và kích hoạt ngay chuỗi cảnh báo rung (Pre-alarm SOS). Cơ chế này kết hợp với Precision cực cao sẽ tạo ra một hệ thống SOS vừa an toàn vừa cực kỳ đáng tin cậy.
> 
> 2. **Luồng Hành Vi Liên Tục (Majority Vote):** Dù model xuất sắc, thực tế cảm biến rung lắc vẫn có thể gây chớp nháy tín hiệu. Phải triển khai Bộ đệm xoay vòng (Circular Buffer) từ 3-5 cửa sổ (tương đương 3-5 giây) để bầu chọn số đông (Majority Vote) cho các lớp Walk, Run và Static. Điều này đảm bảo UI/UX mượt mà trên App điện thoại.
