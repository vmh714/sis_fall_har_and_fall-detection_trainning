# Lịch Sử Huấn Luyện & Tối Ưu Hóa Mô Hình TCN (Fall Detection & HAR)

Tài liệu này tổng hợp quá trình tinh chỉnh (tuning) kiến trúc Temporal Convolutional Network (TCN) cho bài toán nhận diện hành vi (HAR) và phát hiện té ngã (Fall Detection) dựa trên dữ liệu cảm biến IMU (Gia tốc & Góc quay). 

Mục tiêu tối thượng là đạt Recall > 95% cho lớp Fall (tiêu chuẩn an toàn y tế) trong khi giữ được sự mượt mà và chính xác giữa các lớp hành vi liên tục (Walk, Run, Static/ADL).

---

## 1. Bảng So Sánh Các Phiên Bản Huấn Luyện (v11 - v15)

| Chỉ số / Lớp | v11 (Bản Khởi Điểm) | v12 (Baseline Tối Ưu) | v13 (Max Pooling) | v14 (Ép Class Weight) | v15 (Dense Phi Tuyến) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy Tổng** | 85.37% | **92.24%** | 91.55% | 88.87% | 87.35% |
| **Walk Recall** | **90.38%** | 83.47% | 82.27% | 86.88% | 85.82% |
| **Walk Precision**| 69.83% | **90.44%** | 90.58% | 78.40% | 78.99% |
| **Static Recall** | 76.42% | **94.19%** | 94.51% | 85.84% | 83.06% |
| **Fall Recall** | 97.73% | **97.73%** | 94.27% | 97.33% | **98.00%** |
| **Fall Precision**| 92.55% | **95.44%** | 94.90% | 94.56% | 81.13% |

---

## 2. Chi Tiết Các Thử Nghiệm & Bài Học Rút Ra

### 🧱 Phiên bản v11: Bản khởi điểm (Bài học về Over-penalizing)
- **Tình trạng:** Cấu hình Dropout cao và hệ số phạt (Class Weight) cho lớp Fall/Walk được thiết lập rất lớn.
- **Kết quả:** Walk Recall đạt đỉnh (90.38%) và Fall Recall vượt chuẩn an toàn (97.73%). Tuy nhiên, hệ số phạt quá cao khiến mô hình rơi vào trạng thái "sợ hãi" và dự đoán nhầm hàng loạt mẫu Static thành Walk hoặc Fall. Static Recall chỉ đạt 76.42% và Walk Precision chạm đáy 69.83%. Accuracy tổng thể thấp nhất (85.37%).
- **Bài học:** Ép trọng số phạt quá nặng ngay từ đầu làm hỏng khả năng phân loại tổng quát.

### 🏆 Phiên bản v12: Sự đơn giản tạo nên nhà vô địch (Bước ngoặt từ v11)
- **Thay đổi:** Giảm tỉ lệ Dropout và hạ bớt hệ số phạt khi đoán sai Fall so với `v11`. Kiến trúc TCN thuần túy + `GlobalAveragePooling1D`. Không dùng Label Smoothing.
- **Cấu hình:** Ngưỡng Fall cố định ở mức `0.25` để ưu tiên phát hiện ngã.
- **Đánh giá:** Đạt được điểm cân bằng hoàn hảo nhất. Fall Recall vượt chuẩn an toàn (97.73%) đồng thời Fall Precision cực kỳ xuất sắc (95.44%). Độ nhiễu loạn giữa Đi bộ và Đứng yên rất thấp. Đây là phiên bản TỐT NHẤT để triển khai thực tế.

### ❌ Phiên bản v13: Thất bại với Max Pooling & Label Smoothing
- **Thay đổi:** Đổi Pooling thành `GlobalMaxPooling1D` và thêm Label Smoothing.
- **Bài học:** Té ngã là một sự kiện tiêu hao năng lượng phân bổ theo chuỗi thời gian, không phải là một đỉnh xung đơn lẻ. Việc dùng `MaxPooling` bỏ qua đi bức tranh toàn cảnh của cửa sổ 2 giây. Hơn nữa, Label Smoothing làm dẹt xác suất, gây xung đột mạnh với ngưỡng cảnh báo cứng `0.25` của Fall. Kết quả Fall Recall tụt dưới 95%.

### ❌ Phiên bản v14: Tác dụng phụ của Class Weights cực đoan
- **Thay đổi:** Khôi phục Average Pooling. Thêm hệ số phạt cực mạnh: `Walk x1.3` và `Fall x3.0`.
- **Bài học:** Việc ép mô hình phải "sợ" lỗi của lớp Walk khiến nó phát sinh hội chứng "bắt nhầm hơn bỏ sót". Hàng trăm cửa sổ Static/ADL bị nhận diện sai thành Walk, kéo sập Precision của Walk xuống 78.40% và giảm mạnh Recall của Static. Hiện tượng này (Training loss có trọng số vs Val loss không trọng số) gây ra sự phân kỳ sớm, khiến mô hình dừng học ở ngay Epoch 10.

### ❌ Phiên bản v15: Sự hoang tưởng của tầng Dense phi tuyến
- **Thay đổi:** Bổ sung lớp giải mã phi tuyến `Dense(64, activation='relu')` trước khi ra Softmax, hạ Walk weight xuống 1.05.
- **Bài học:** Với năng lực bẻ cong ranh giới quyết định mạnh mẽ hơn cộng với sức ép phải đạt ngưỡng Fall 0.25, mô hình đã mở rộng "vùng nhận diện Fall" quá lớn. Fall Recall chạm đỉnh vinh quang 98.00%, nhưng đổi lại là ~170 ca báo động giả (False Positives), khiến Precision rớt thê thảm xuống 81.13%. Trên thực tế, mức báo động giả này sẽ phá hủy hoàn toàn trải nghiệm UX của người dùng đeo thiết bị.

---

## 3. Chiến Lược Triển Khai Thực Tế (MCU / C++)

Dựa trên thực nghiệm, chúng ta thống nhất cấu trúc hệ thống phần mềm nhúng để cài đặt lên thiết bị đeo (Wearable / ESP32) như sau:

> [!IMPORTANT]  
> **Chốt sổ Kiến trúc & Trọng số:** Sử dụng cấu hình và tệp trọng số `.keras` / `.tflite` của **v12**.

> [!TIP]  
> **Kỹ Thuật Hậu Xử Lý (Post-Processing) Bắt Buộc:**
> 
> 1. **Luồng Khẩn Cấp (Fall):** Theo dõi xác suất từ đầu ra mô hình. Ngay khi `Prob_Fall >= 0.25`, BỎ QUA MỌI CHỜ ĐỢI và kích hoạt ngay cảnh báo rung (Pre-alarm SOS). Điều này giúp cứu sống bệnh nhân chỉ trong tích tắc.
> 
> 2. **Luồng Hành Vi Liên Tục (Majority Vote):** Dù `v12` là nhà vô địch, vẫn sẽ có những nhịp đi bộ bị nhiễu cảm biến tạm thời. Phải triển khai Bộ đệm xoay vòng (Circular Buffer) 3-5 cửa sổ (tương đương 3-5 giây) để bầu chọn số đông cho các lớp Walk, Run và Static. Nhờ Precision > 90% của `v12`, cơ chế Majority Vote sẽ hoạt động cực kỳ mượt mà, triệt tiêu hoàn toàn sự chớp nháy của giao diện UI giữa trạng thái đi và đứng.
