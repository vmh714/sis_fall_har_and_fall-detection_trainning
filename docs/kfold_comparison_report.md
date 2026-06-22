# Báo cáo Phân tích Kết quả K-Fold: v25 (ResNet1D) vs v30 (CNN) vs v30_tcn (TCN)

Báo cáo này trình bày tổng quan cấu trúc của 3 mô hình và so sánh kết quả đánh giá (evaluation metrics) từ các thử nghiệm K-Fold trên tập dữ liệu SisFall. Dựa trên yêu cầu của dự án, **Fall Recall** (tránh bỏ sót té ngã) và **Trans F1** (nhận diện tốt nhãn chuyển tiếp) là hai chỉ số ưu tiên hàng đầu.

---

## Phần 1: Tổng quan các mô hình

### 1. Mô hình v30 (CNN thuần tối ưu ESP-NN)
- **Kiến trúc chính:** Sử dụng cấu trúc mạng Convolutional Neural Network 1D (CNN thuần) với các lớp `Conv1D` và `SeparableConv1D`. Các lớp kích hoạt là `relu6` kết hợp với `BatchNormalization`. Đặc biệt sử dụng cả `GlobalAveragePooling1D` và `MaxPooling1D` ở đoạn cuối để không làm loãng đỉnh gia tốc của các cử động đột ngột (như té ngã).
- **Kích thước mô hình (Keras):** ~174 KB.
- **Đặc điểm & Ứng dụng:** Đây là mô hình được thiết kế hoàn toàn dựa trên các toán tử được tối ưu hóa cứng bởi vi xử lý **ESP-NN** trên chip ESP32-S3 (như SeparableConv, ReLU6, MaxPool). Mô hình có khả năng chạy suy luận (inference) cực kỳ nhanh (latency thấp) và tiêu tốn ít bộ nhớ flash nhất.

### 2. Mô hình v30_tcn (TCN - Temporal Convolutional Network)
- **Kiến trúc chính:** Dựa trên các lớp `Conv1D` với hệ số giãn nở (Dilation Rate: 1, 2, 4, 8) nhằm gia tăng kích thước cửa sổ nhìn (Receptive Field) để bắt được ngữ cảnh chuỗi thời gian dài. Ngoài ra, mô hình còn sử dụng cấu trúc Residual (Skip-connections) kết hợp khối SE (Squeeze-and-Excitation) dùng hàm `sigmoid`.
- **Kích thước mô hình (Keras):** ~975 KB.
- **Đặc điểm & Ứng dụng:** TCN có khả năng ghi nhớ ngữ cảnh chuỗi thời gian vượt trội. Nhờ đó, hiệu suất phân loại cực kỳ xuất sắc. Tuy nhiên, nhược điểm chí mạng là **dilated convolution và sigmoid KHÔNG ĐƯỢC ESP-NN hỗ trợ tăng tốc**. Mạch ESP32-S3 sẽ phải dùng reference kernel để chạy, dẫn đến độ trễ suy luận rất cao (latency ~100ms hoặc hơn).

### 3. Mô hình v25 (ResNet1D)
- **Kiến trúc chính:** Là mô hình mạng dư (Residual Network) 1D. Cấu trúc cốt lõi sử dụng các block `SeparableConv1D` kết hợp kết nối dư (skip-connections) và có tích hợp khối SE (Squeeze-and-Excitation).
- **Kích thước mô hình (Keras):** ~502 KB.
- **Đặc điểm & Ứng dụng:** Là sự kết hợp cân bằng giữa khả năng trích xuất đặc trưng sâu (ResNet) và việc giảm thiểu số lượng tham số (SeparableConv). Khối SE có hàm `sigmoid` có thể làm tăng nhẹ thời gian suy luận so với v30 (CNN) nhưng hiệu năng nhận diện nhãn khó (như Trans) được cải thiện.

---

## Phần 2: So sánh Kết quả Đánh giá K-Fold

*Các chỉ số dưới đây được lấy trung bình từ các Folds.*

### 2.1. So sánh các chỉ số ưu tiên (Fall Recall & Trans F1)
*Định dạng: `[Fall Recall] / [Trans F1]`*

| Kịch bản đánh giá | v30 (CNN) | v30_tcn (TCN) | v25 (ResNet1D) | Mô hình tốt nhất |
| :--- | :---: | :---: | :---: | :---: |
| **S1_Elderly** <br>*(Train & Test trên người già)* | 0.0000 / 0.6610 | 0.0000 / **0.8305** | 0.0000 / 0.7349 | **v30_tcn** |
| **S2_Young** <br>*(Train & Test trên người trẻ)* | **0.9957** / 0.8192 | 0.9949 / **0.9348** | 0.9937 / 0.8840 | **v30_tcn** |
| **S3_TrainSE_TestSA** <br>*(Train trên Già, Test trên Trẻ)* | 0.8586 / 0.6495 | **0.9586** / **0.8847** | 0.8811 / 0.7379 | **v30_tcn** |
| **S4_TrainSA_TestSE** <br>*(Train trên Trẻ, Test trên Già)* | 0.9867 / 0.7457 | **0.9893** / **0.9096** | 0.9680 / 0.8362 | **v30_tcn** |
| **S5_Both** <br>*(Trộn chung Già & Trẻ - K-Fold)* | **0.9950** / 0.8185 | 0.9922 / **0.9371** | 0.9930 / 0.8900 | **v30_tcn** |

> **Ghi chú:** Ở kịch bản S1_Elderly, Fall Recall bằng 0 do số lượng mẫu té ngã của người già trong tập test thực tế rất nhỏ hoặc bị thiếu, dẫn đến mô hình không bắt được. Ở hầu hết các kịch bản khác, Recall đều > 95%.

### 2.2. So sánh Tổng quan (Accuracy & Macro F1)
*Định dạng: `[Accuracy] / [Macro F1]`*

| Kịch bản đánh giá | v30 (CNN) | v30_tcn (TCN) | v25 (ResNet1D) |
| :--- | :---: | :---: | :---: |
| **S1_Elderly** | 0.8746 / 0.6856 | **0.9056** / **0.7263** | 0.8454 / 0.6845 |
| **S2_Young** | 0.9515 / 0.9352 | **0.9718** / **0.9677** | 0.9409 / 0.9462 |
| **S3_TrainSE_TestSA** | 0.9060 / 0.8800 | **0.9524** / **0.9451** | 0.8833 / 0.8846 |
| **S4_TrainSA_TestSE** | 0.9061 / 0.8831 | **0.9379** / **0.9345** | 0.9035 / 0.9114 |
| **S5_Both** | 0.9445 / 0.9320 | **0.9657** / **0.9648** | 0.9378 / 0.9462 |

---

## Phần 3: Kết luận & Đánh giá chuyên sâu (4 Ý Chính)

1. **Hiệu suất tổng thể (Accuracy & Macro F1):**
   Mô hình **v30_tcn (TCN)** là mô hình có độ chính xác và Macro F1 tốt nhất trên hầu hết các kịch bản đánh giá. Nó đặc biệt vượt trội và bỏ xa hai mô hình còn lại ở các kịch bản dự đoán chéo khó nhằn như S3 (Train Già, Test Trẻ).
2. **Fall Recall (Chỉ số ưu tiên số 1):**
   Trên tập S5 (trộn chung dữ liệu), cả 3 mô hình đều đạt Fall Recall rất cao (>99.2%). Tuy nhiên, điểm tạo nên sự khác biệt là ở kịch bản S3 (Train trên Người già, Test trên Người trẻ). **v30_tcn** vẫn giữ được mức Recall ấn tượng là **95.86%**, trong khi v30 (CNN) bị sụt giảm xuống còn 85.86% và v25 (ResNet) chỉ đạt 88.11%. TCN có sức chống chịu (robustness) rất tốt trước sự thay đổi miền dữ liệu.
3. **Phân loại nhãn dễ nhầm lẫn - Trans (Chuyển tiếp):**
   **v30_tcn** xử lý cực kỳ xuất sắc việc phân loại nhãn Trans. Cụ thể trên tập S5, Trans F1 của TCN đạt **93.71%**, đánh bại hoàn toàn v25 (89.00%) và v30 (81.85%). Khả năng nắm bắt chuỗi thời gian (temporal context) qua Dilated Convolutions đã phát huy tác dụng tối đa.
4. **Lưu ý triển khai (Hardware / ESP-NN Constraints):**
   Mặc dù TCN có độ chính xác tuyệt vời, cấu trúc của nó (Dilated Conv, Sigmoid) **KHÔNG ĐƯỢC bộ vi xử lý ESP-NN hỗ trợ tăng tốc**. Điều này có nghĩa là khi nén INT8 và chạy trên chip ESP32-S3, độ trễ suy luận (inference latency) sẽ rất chậm (~100ms hoặc hơn). Ngược lại, **v30 (CNN thuần)** hoặc **v25 (ResNet1D)** sử dụng kiến trúc cực kỳ "thân thiện" với vi mạch (pointwise, depthwise conv, relu6), mang lại độ trễ siêu thấp (~10ms). Sự lựa chọn cuối cùng sẽ phải là sự cân bằng (trade-off) giữa độ chính xác và tốc độ.
