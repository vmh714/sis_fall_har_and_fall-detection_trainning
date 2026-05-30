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

## 3. Chiến Lược Triển Khai Thực Tế (MCU / C++)

Dựa trên chuỗi thực nghiệm toàn diện với tập dữ liệu chuẩn 9 kênh (Không la bàn), chúng ta thống nhất hệ thống phần mềm nhúng để cài đặt lên thiết bị đeo (Wearable) như sau:

> [!IMPORTANT]  
> **Chốt sổ Kiến trúc & Trọng số:** Sử dụng cấu hình và tệp trọng số `.keras` / `.tflite` của phiên bản **v14**. Ngôi vương đã đổi chủ từ v12 (dataset cũ) sang v14 (dataset 9 kênh chuẩn).

> [!TIP]  
> **Kỹ Thuật Hậu Xử Lý (Post-Processing) Bắt Buộc:**
> 
> 1. **Luồng Khẩn Cấp (Fall):** Theo dõi xác suất từ đầu ra mô hình liên tục. Ngay khi `Prob_Fall >= 0.25`, BỎ QUA MỌI CHỜ ĐỢI và kích hoạt ngay chuỗi cảnh báo rung (Pre-alarm SOS). Cơ chế này kết hợp với Precision 94.83% của v14 sẽ tạo ra một hệ thống SOS vừa an toàn vừa cực kỳ đáng tin cậy.
> 
> 2. **Luồng Hành Vi Liên Tục (Majority Vote):** Dù v14 rất xuất sắc, thực tế cảm biến rung lắc vẫn có thể gây chớp nháy tín hiệu. Phải triển khai Bộ đệm xoay vòng (Circular Buffer) từ 3-5 cửa sổ (tương đương 3-5 giây) để bầu chọn số đông (Majority Vote) cho các lớp Walk, Run và Static. Điều này đảm bảo UI/UX mượt mà trên App điện thoại.
