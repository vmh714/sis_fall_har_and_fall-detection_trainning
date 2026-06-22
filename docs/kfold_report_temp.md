# Báo cáo Kết quả K-Fold Cross Validation: v25 (ResNet1D) vs v30 (CNN) vs v30\_tcn (TCN)

## 1. Tổng quan các mô hình

### 1.1. Mô hình v25 (ResNet1D)
- **Kiến trúc chính:** Là mạng Residual Network 1D. Cấu trúc cốt lõi sử dụng các block `SeparableConv1D` kết hợp kết nối dư (skip-connections) và có tích hợp khối SE (Squeeze-and-Excitation).
- **Kích thước mô hình (Keras):** ~502 KB.
- **Đặc điểm & Ứng dụng:** Là sự cân bằng giữa khả năng trích xuất đặc trưng sâu (ResNet) và việc giảm thiểu tham số (SeparableConv). Khối SE có hàm `sigmoid` có thể làm tăng nhẹ thời gian suy luận so với v30, nhưng cải thiện đáng kể khả năng nhận diện các nhãn khó (như Trans).

### 1.2. Mô hình v30 (CNN thuần tối ưu ESP-NN)
- **Kiến trúc chính:** Sử dụng cấu trúc mạng Convolutional Neural Network 1D (CNN thuần) với các lớp `Conv1D` và `SeparableConv1D`. Các lớp kích hoạt là `relu6` kết hợp với `BatchNormalization`. Đặc biệt sử dụng cả `GlobalAveragePooling1D` và `MaxPooling1D` ở đoạn cuối để không làm loãng đỉnh gia tốc của các cử động đột ngột (như té ngã).
- **Kích thước mô hình (Keras):** ~174 KB.
- **Đặc điểm & Ứng dụng:** Được thiết kế hoàn toàn dựa trên các toán tử tối ưu hóa cứng bởi vi xử lý **ESP-NN** trên chip ESP32-S3. Mô hình có khả năng chạy suy luận nhanh (latency cực thấp) và tiêu tốn ít bộ nhớ flash.

### 1.3. Mô hình v30\_tcn (TCN - Temporal Convolutional Network)
- **Kiến trúc chính:** Dựa trên các lớp `Conv1D` với hệ số giãn nở (Dilation Rate: 1, 2, 4, 8) nhằm gia tăng kích thước cửa sổ nhìn (Receptive Field) để bắt được ngữ cảnh chuỗi thời gian dài. Ngoài ra, mô hình còn sử dụng cấu trúc Residual (Skip-connections) kết hợp khối SE (Squeeze-and-Excitation) dùng hàm `sigmoid`.
- **Kích thước mô hình (Keras):** ~975 KB.
- **Đặc điểm & Ứng dụng:** TCN ghi nhớ ngữ cảnh chuỗi thời gian vượt trội, đem lại hiệu suất phân loại xuất sắc. Tuy nhiên, **dilated convolution và sigmoid không được ESP-NN hỗ trợ tăng tốc**. Mạch ESP32-S3 sẽ phải dùng reference kernel để chạy, dẫn đến độ trễ suy luận rất cao (~100ms hoặc hơn).

## 2. Kết quả Đánh giá K-Fold

*Ghi chú: Các chỉ số được lấy trung bình từ kết quả các Fold trong từng kịch bản thử nghiệm.*

### 2.1. So sánh các chỉ số ưu tiên (Fall Recall & Trans F1)

*Định dạng: Fall Recall (%) / Trans F1 (%)*

| Kịch bản đánh giá | v25 (ResNet1D) | v30 (CNN) | v30\_tcn (TCN) |
| :--- | :---: | :---: | :---: |
| **S1\_Elderly** (Train & Test trên người già) | 0.00 / 73.49 | 0.00 / 66.10 | 0.00 / **83.05** |
| **S2\_Young** (Train & Test trên người trẻ) | 99.37 / 88.40 | **99.57** / 81.92 | 99.49 / **93.48** |
| **S3\_TrainSE\_TestSA** (Train Già, Test Trẻ) | 88.11 / 73.79 | 85.86 / 64.95 | **95.86** / **88.47** |
| **S4\_TrainSA\_TestSE** (Train Trẻ, Test Già) | 96.80 / 83.62 | 98.67 / 74.57 | **98.93** / **90.96** |
| **S5\_Both** (Trộn chung Già & Trẻ - K-Fold) | 99.30 / 89.00 | **99.50** / 81.85 | 99.22 / **93.71** |

*(Fall Recall ở S1 bằng 0 do số lượng mẫu té ngã của người già trong tập test thực tế rất nhỏ hoặc bị thiếu).*

### 2.2. So sánh Tổng quan (Accuracy & Macro F1)

*Định dạng: Accuracy (%) / Macro F1 (%)*

| Kịch bản đánh giá | v25 (ResNet1D) | v30 (CNN) | v30\_tcn (TCN) |
| :--- | :---: | :---: | :---: |
| **S1\_Elderly** | 84.54 / 68.45 | 87.46 / 68.56 | **90.56** / **72.63** |
| **S2\_Young** | 94.09 / 94.62 | 95.15 / 93.52 | **97.18** / **96.77** |
| **S3\_TrainSE\_TestSA** | 88.33 / 88.46 | 90.60 / 88.00 | **95.24** / **94.51** |
| **S4\_TrainSA\_TestSE** | 90.35 / 91.14 | 90.61 / 88.31 | **93.79** / **93.45** |
| **S5\_Both** | 93.78 / 94.62 | 94.45 / 93.20 | **96.57** / **96.48** |

## 3. Đánh giá chuyên sâu và Nhận xét

1. **Hiệu suất tổng thể (Accuracy & Macro F1):**
   Mô hình **v30\_tcn (TCN)** đạt độ chính xác và Macro F1 tốt nhất trên hầu hết các kịch bản đánh giá. Nó đặc biệt vượt trội ở các kịch bản dự đoán chéo khó nhằn như S3 (Train trên người già, Test trên người trẻ), cho thấy khả năng tổng quát hóa xuất sắc.
   
2. **Fall Recall (Chỉ số ưu tiên số 1):**
   Trên tập S5 (trộn chung dữ liệu), cả 3 mô hình đều đạt Fall Recall rất cao (>99.2%). Điểm tạo nên sự khác biệt lớn nhất là ở kịch bản S3. **v30\_tcn** giữ được mức Recall ấn tượng là **95.86%**, trong khi v25 (ResNet) đạt 88.11% và v30 (CNN) sụt giảm mạnh xuống 85.86%. TCN thể hiện sức chống chịu (robustness) rất tốt trước sự thay đổi miền dữ liệu.
   
3. **Phân loại nhãn dễ nhầm lẫn - Trans (Chuyển tiếp):**
   **v30\_tcn** xử lý cực kỳ xuất sắc việc phân loại nhãn Trans. Cụ thể trên tập S5, Trans F1 của TCN đạt **93.71%**, đánh bại hoàn toàn v25 (89.00%) và v30 (81.85%). Khả năng nắm bắt chuỗi thời gian qua Dilated Convolutions giúp TCN giảm thiểu nhầm lẫn giữa các hoạt động phức tạp.
   
4. **Lưu ý triển khai thực tế (Hardware / ESP-NN Constraints):**
   Mặc dù TCN có độ chính xác tuyệt vời, cấu trúc của nó (Dilated Conv, Sigmoid) **không được bộ vi xử lý ESP-NN hỗ trợ tăng tốc**. Do đó, khi nén INT8 và chạy trên chip ESP32-S3, thời gian suy luận (inference latency) sẽ rất chậm (~100ms hoặc hơn). Ngược lại, **v25 (ResNet1D)** hoặc **v30 (CNN thuần)** sử dụng cấu trúc cực kỳ thân thiện với vi mạch (pointwise, depthwise conv, relu6), mang lại độ trễ siêu thấp (~10ms). Việc chọn mô hình cuối cùng sẽ là bài toán đánh đổi (trade-off) giữa độ chính xác và tốc độ.

## Chi tiết kết quả 17 lần huấn luyện: v25 (ResNet1D)

### Kịch bản: S1_Elderly

| Lần chạy | Chỉ số | Walk | Run | Idle | Trans | Fall |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | **Precision (Tự tin)** | 0.8922 | 0.9916 | 0.8322 | 0.6495 | 0.0000 |
| | **Recall (Độ phủ)** | 0.8408 | 0.6268 | 0.9048 | 0.7462 | 0.0000 |
| | **F1-Score** | 0.8657 | 0.7681 | 0.8670 | 0.6945 | 0.0000 |
| **Fold 2** | **Precision (Tự tin)** | 0.9209 | 0.9271 | 0.8225 | 0.6351 | 0.0000 |
| | **Recall (Độ phủ)** | 0.9402 | 0.9882 | 0.9437 | 0.7255 | 0.0000 |
| | **F1-Score** | 0.9304 | 0.9567 | 0.8789 | 0.6773 | 0.0000 |
| **Fold 3** | **Precision (Tự tin)** | 0.9449 | 0.8240 | 0.7589 | 0.8420 | 0.0000 |
| | **Recall (Độ phủ)** | 0.8092 | 0.9458 | 0.9537 | 0.5758 | 0.0000 |
| | **F1-Score** | 0.8718 | 0.8807 | 0.8452 | 0.6839 | 0.0000 |
| **Fold 4** | **Precision (Tự tin)** | 0.9185 | 0.9949 | 0.8052 | 0.9075 | 0.0000 |
| | **Recall (Độ phủ)** | 0.8983 | 0.9933 | 0.9254 | 0.7212 | 0.0000 |
| | **F1-Score** | 0.9083 | 0.9941 | 0.8611 | 0.8037 | 0.0000 |
| **Fold 5** | **Precision (Tự tin)** | 0.9605 | 1.0000 | 0.8128 | 0.9291 | 0.0000 |
| | **Recall (Độ phủ)** | 0.9182 | 0.9814 | 0.9585 | 0.7258 | 0.0000 |
| | **F1-Score** | 0.9389 | 0.9906 | 0.8797 | 0.8150 | 0.0000 |

### Kịch bản: S2_Young

| Lần chạy | Chỉ số | Walk | Run | Idle | Trans | Fall |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | **Precision (Tự tin)** | 0.9933 | 0.9980 | 0.8670 | 0.9581 | 0.9915 |
| | **Recall (Độ phủ)** | 0.9377 | 1.0000 | 0.9734 | 0.8456 | 0.9957 |
| | **F1-Score** | 0.9647 | 0.9990 | 0.9171 | 0.8983 | 0.9936 |
| **Fold 2** | **Precision (Tự tin)** | 0.9699 | 0.9782 | 0.8921 | 0.9554 | 0.9995 |
| | **Recall (Độ phủ)** | 0.9522 | 0.9949 | 0.9619 | 0.8712 | 0.9904 |
| | **F1-Score** | 0.9610 | 0.9865 | 0.9257 | 0.9114 | 0.9949 |
| **Fold 3** | **Precision (Tự tin)** | 0.9642 | 0.9990 | 0.8525 | 0.9695 | 0.9973 |
| | **Recall (Độ phủ)** | 0.9660 | 0.9980 | 0.9702 | 0.7866 | 0.9968 |
| | **F1-Score** | 0.9651 | 0.9985 | 0.9075 | 0.8685 | 0.9971 |
| **Fold 4** | **Precision (Tự tin)** | 0.9772 | 0.9762 | 0.8558 | 0.9698 | 0.9933 |
| | **Recall (Độ phủ)** | 0.9111 | 0.9874 | 0.9725 | 0.8585 | 0.9853 |
| | **F1-Score** | 0.9430 | 0.9818 | 0.9104 | 0.9108 | 0.9893 |
| **Fold 5** | **Precision (Tự tin)** | 0.9828 | 1.0000 | 0.8302 | 0.8804 | 0.9828 |
| | **Recall (Độ phủ)** | 0.8560 | 0.9987 | 0.9672 | 0.7875 | 1.0000 |
| | **F1-Score** | 0.9151 | 0.9994 | 0.8935 | 0.8313 | 0.9914 |

### Kịch bản: S3_TrainSE_TestSA

| Lần chạy | Chỉ số | Walk | Run | Idle | Trans | Fall |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **1 Lần duy nhất** | **Precision (Tự tin)** | 0.9544 | 0.8718 | 0.8054 | 0.8368 | 0.9916 |
| | **Recall (Độ phủ)** | 0.9251 | 0.9980 | 0.9745 | 0.6599 | 0.8811 |
| | **F1-Score** | 0.9395 | 0.9307 | 0.8819 | 0.7379 | 0.9331 |

### Kịch bản: S4_TrainSA_TestSE

| Lần chạy | Chỉ số | Walk | Run | Idle | Trans | Fall |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **1 Lần duy nhất** | **Precision (Tự tin)** | 0.9248 | 0.9996 | 0.8392 | 0.9293 | 0.8963 |
| | **Recall (Độ phủ)** | 0.9363 | 0.9404 | 0.9487 | 0.7600 | 0.9680 |
| | **F1-Score** | 0.9305 | 0.9691 | 0.8906 | 0.8362 | 0.9308 |

### Kịch bản: S5_Both

| Lần chạy | Chỉ số | Walk | Run | Idle | Trans | Fall |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | **Precision (Tự tin)** | 0.9436 | 0.9988 | 0.8764 | 0.9372 | 0.9984 |
| | **Recall (Độ phủ)** | 0.9478 | 0.9587 | 0.9475 | 0.8524 | 0.9973 |
| | **F1-Score** | 0.9457 | 0.9783 | 0.9106 | 0.8928 | 0.9979 |
| **Fold 2** | **Precision (Tự tin)** | 0.9610 | 0.9874 | 0.8776 | 0.9594 | 0.9982 |
| | **Recall (Độ phủ)** | 0.9493 | 0.9930 | 0.9637 | 0.8447 | 0.9889 |
| | **F1-Score** | 0.9551 | 0.9902 | 0.9186 | 0.8984 | 0.9935 |
| **Fold 3** | **Precision (Tự tin)** | 0.9663 | 0.9978 | 0.8860 | 0.9434 | 0.9979 |
| | **Recall (Độ phủ)** | 0.9591 | 0.9821 | 0.9585 | 0.8485 | 0.9968 |
| | **F1-Score** | 0.9627 | 0.9899 | 0.9208 | 0.8934 | 0.9973 |
| **Fold 4** | **Precision (Tự tin)** | 0.9708 | 0.9787 | 0.8604 | 0.9604 | 0.9847 |
| | **Recall (Độ phủ)** | 0.9101 | 0.9949 | 0.9649 | 0.8482 | 0.9840 |
| | **F1-Score** | 0.9395 | 0.9868 | 0.9097 | 0.9008 | 0.9843 |
| **Fold 5** | **Precision (Tự tin)** | 0.9757 | 0.9993 | 0.8440 | 0.9271 | 0.9796 |
| | **Recall (Độ phủ)** | 0.9025 | 0.9949 | 0.9636 | 0.8102 | 0.9980 |
| | **F1-Score** | 0.9377 | 0.9971 | 0.8998 | 0.8647 | 0.9887 |


## Chi tiết kết quả 17 lần huấn luyện: v30 (CNN thuần)

### Kịch bản: S1_Elderly

| Lần chạy | Chỉ số | Walk | Run | Idle | Trans | Fall |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | **Precision (Tự tin)** | 0.8967 | 0.9985 | 0.8854 | 0.7186 | 0.0000 |
| | **Recall (Độ phủ)** | 0.9442 | 0.8778 | 0.8854 | 0.7571 | 0.0000 |
| | **F1-Score** | 0.9198 | 0.9343 | 0.8854 | 0.7374 | 0.0000 |
| **Fold 2** | **Precision (Tự tin)** | 0.9450 | 0.9733 | 0.8324 | 0.3609 | 0.0000 |
| | **Recall (Độ phủ)** | 0.9549 | 0.9865 | 0.9167 | 0.7053 | 0.0000 |
| | **F1-Score** | 0.9499 | 0.9799 | 0.8725 | 0.4774 | 0.0000 |
| **Fold 3** | **Precision (Tự tin)** | 0.9441 | 0.9974 | 0.8555 | 0.6381 | 0.0000 |
| | **Recall (Độ phủ)** | 0.9036 | 0.9335 | 0.8876 | 0.7128 | 0.0000 |
| | **F1-Score** | 0.9234 | 0.9644 | 0.8713 | 0.6734 | 0.0000 |
| **Fold 4** | **Precision (Tự tin)** | 0.9352 | 0.9916 | 0.8315 | 0.6694 | 0.0000 |
| | **Recall (Độ phủ)** | 0.8931 | 0.9899 | 0.8511 | 0.6953 | 0.0000 |
| | **F1-Score** | 0.9137 | 0.9907 | 0.8412 | 0.6821 | 0.0000 |
| **Fold 5** | **Precision (Tự tin)** | 0.9760 | 0.9898 | 0.8368 | 0.7244 | 0.0000 |
| | **Recall (Độ phủ)** | 0.8962 | 0.9848 | 0.9020 | 0.7449 | 0.0000 |
| | **F1-Score** | 0.9344 | 0.9873 | 0.8682 | 0.7345 | 0.0000 |

### Kịch bản: S2_Young

| Lần chạy | Chỉ số | Walk | Run | Idle | Trans | Fall |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | **Precision (Tự tin)** | 0.9875 | 0.9960 | 0.9038 | 0.8209 | 0.9989 |
| | **Recall (Độ phủ)** | 0.9465 | 0.9960 | 0.9344 | 0.8404 | 0.9995 |
| | **F1-Score** | 0.9666 | 0.9960 | 0.9188 | 0.8305 | 0.9992 |
| **Fold 2** | **Precision (Tự tin)** | 0.9658 | 0.9879 | 0.9268 | 0.8330 | 0.9952 |
| | **Recall (Độ phủ)** | 0.9579 | 0.9919 | 0.9331 | 0.8412 | 0.9915 |
| | **F1-Score** | 0.9618 | 0.9899 | 0.9299 | 0.8371 | 0.9933 |
| **Fold 3** | **Precision (Tự tin)** | 0.9778 | 0.9989 | 0.9142 | 0.8616 | 0.9745 |
| | **Recall (Độ phủ)** | 0.9679 | 0.9545 | 0.9474 | 0.7935 | 0.9995 |
| | **F1-Score** | 0.9728 | 0.9762 | 0.9305 | 0.8262 | 0.9868 |
| **Fold 4** | **Precision (Tự tin)** | 0.9803 | 0.9774 | 0.8872 | 0.8315 | 0.9913 |
| | **Recall (Độ phủ)** | 0.9024 | 0.9861 | 0.9477 | 0.8632 | 0.9880 |
| | **F1-Score** | 0.9398 | 0.9817 | 0.9165 | 0.8471 | 0.9896 |
| **Fold 5** | **Precision (Tự tin)** | 0.9805 | 1.0000 | 0.8794 | 0.7177 | 0.9953 |
| | **Recall (Độ phủ)** | 0.9095 | 0.9975 | 0.9119 | 0.7961 | 1.0000 |
| | **F1-Score** | 0.9437 | 0.9987 | 0.8953 | 0.7549 | 0.9977 |

### Kịch bản: S3_TrainSE_TestSA

| Lần chạy | Chỉ số | Walk | Run | Idle | Trans | Fall |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **1 Lần duy nhất** | **Precision (Tự tin)** | 0.9431 | 0.9585 | 0.8937 | 0.5713 | 0.9974 |
| | **Recall (Độ phủ)** | 0.9434 | 0.9943 | 0.9230 | 0.7526 | 0.8586 |
| | **F1-Score** | 0.9433 | 0.9761 | 0.9081 | 0.6495 | 0.9228 |

### Kịch bản: S4_TrainSA_TestSE

| Lần chạy | Chỉ số | Walk | Run | Idle | Trans | Fall |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **1 Lần duy nhất** | **Precision (Tự tin)** | 0.9381 | 0.9993 | 0.8811 | 0.7228 | 0.8061 |
| | **Recall (Độ phủ)** | 0.9387 | 0.9145 | 0.8974 | 0.7702 | 0.9867 |
| | **F1-Score** | 0.9384 | 0.9550 | 0.8892 | 0.7457 | 0.8873 |

### Kịch bản: S5_Both

| Lần chạy | Chỉ số | Walk | Run | Idle | Trans | Fall |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | **Precision (Tự tin)** | 0.9483 | 0.9988 | 0.8990 | 0.8240 | 0.9968 |
| | **Recall (Độ phủ)** | 0.9571 | 0.9455 | 0.9350 | 0.7975 | 0.9995 |
| | **F1-Score** | 0.9527 | 0.9714 | 0.9166 | 0.8105 | 0.9981 |
| **Fold 2** | **Precision (Tự tin)** | 0.9696 | 0.9899 | 0.9077 | 0.8418 | 0.9920 |
| | **Recall (Độ phủ)** | 0.9536 | 0.9893 | 0.9434 | 0.7899 | 0.9902 |
| | **F1-Score** | 0.9615 | 0.9896 | 0.9252 | 0.8151 | 0.9911 |
| **Fold 3** | **Precision (Tự tin)** | 0.9714 | 0.9977 | 0.9122 | 0.8248 | 0.9675 |
| | **Recall (Độ phủ)** | 0.9627 | 0.9398 | 0.9411 | 0.7914 | 0.9995 |
| | **F1-Score** | 0.9670 | 0.9679 | 0.9264 | 0.8078 | 0.9832 |
| **Fold 4** | **Precision (Tự tin)** | 0.9619 | 0.9836 | 0.8768 | 0.8797 | 0.9886 |
| | **Recall (Độ phủ)** | 0.9196 | 0.9942 | 0.9313 | 0.8292 | 0.9867 |
| | **F1-Score** | 0.9403 | 0.9889 | 0.9033 | 0.8537 | 0.9877 |
| **Fold 5** | **Precision (Tự tin)** | 0.9792 | 0.9993 | 0.8780 | 0.8130 | 0.9770 |
| | **Recall (Độ phủ)** | 0.9115 | 0.9921 | 0.9425 | 0.7982 | 0.9993 |
| | **F1-Score** | 0.9441 | 0.9956 | 0.9091 | 0.8055 | 0.9881 |


## Chi tiết kết quả 17 lần huấn luyện: v30_tcn (TCN)

### Kịch bản: S1_Elderly

| Lần chạy | Chỉ số | Walk | Run | Idle | Trans | Fall |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | **Precision (Tự tin)** | 0.9187 | 0.9985 | 0.9057 | 0.8949 | 0.0000 |
| | **Recall (Độ phủ)** | 0.9451 | 0.9084 | 0.9505 | 0.8821 | 0.0000 |
| | **F1-Score** | 0.9317 | 0.9513 | 0.9276 | 0.8885 | 0.0000 |
| **Fold 2** | **Precision (Tự tin)** | 0.9566 | 0.9543 | 0.7741 | 0.5304 | 0.0000 |
| | **Recall (Độ phủ)** | 0.9486 | 0.9882 | 0.9464 | 0.8561 | 0.0000 |
| | **F1-Score** | 0.9526 | 0.9710 | 0.8516 | 0.6550 | 0.0000 |
| **Fold 3** | **Precision (Tự tin)** | 0.9613 | 0.9796 | 0.8777 | 0.8155 | 0.0000 |
| | **Recall (Độ phủ)** | 0.8847 | 0.9458 | 0.9415 | 0.8936 | 0.0000 |
| | **F1-Score** | 0.9214 | 0.9624 | 0.9085 | 0.8528 | 0.0000 |
| **Fold 4** | **Precision (Tự tin)** | 0.9438 | 0.9673 | 0.8748 | 0.8413 | 0.0000 |
| | **Recall (Độ phủ)** | 0.8795 | 0.9966 | 0.8933 | 0.9099 | 0.0000 |
| | **F1-Score** | 0.9105 | 0.9818 | 0.8840 | 0.8742 | 0.0000 |
| **Fold 5** | **Precision (Tự tin)** | 0.9930 | 0.9983 | 0.8692 | 0.8730 | 0.0000 |
| | **Recall (Độ phủ)** | 0.8899 | 0.9949 | 0.9686 | 0.8907 | 0.0000 |
| | **F1-Score** | 0.9386 | 0.9966 | 0.9162 | 0.8818 | 0.0000 |

### Kịch bản: S2_Young

| Lần chạy | Chỉ số | Walk | Run | Idle | Trans | Fall |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | **Precision (Tự tin)** | 0.9934 | 0.9960 | 0.9302 | 0.9252 | 0.9989 |
| | **Recall (Độ phủ)** | 0.9403 | 0.9980 | 0.9728 | 0.9457 | 0.9995 |
| | **F1-Score** | 0.9661 | 0.9970 | 0.9511 | 0.9353 | 0.9992 |
| **Fold 2** | **Precision (Tự tin)** | 0.9730 | 0.9900 | 0.9471 | 0.9160 | 0.9973 |
| | **Recall (Độ phủ)** | 0.9522 | 0.9960 | 0.9631 | 0.9412 | 0.9904 |
| | **F1-Score** | 0.9625 | 0.9930 | 0.9551 | 0.9284 | 0.9938 |
| **Fold 3** | **Precision (Tự tin)** | 0.9859 | 1.0000 | 0.9464 | 0.9490 | 0.9989 |
| | **Recall (Độ phủ)** | 0.9679 | 0.9970 | 0.9747 | 0.9354 | 0.9957 |
| | **F1-Score** | 0.9768 | 0.9985 | 0.9604 | 0.9421 | 0.9973 |
| **Fold 4** | **Precision (Tự tin)** | 0.9915 | 0.9789 | 0.9065 | 0.9468 | 0.9973 |
| | **Recall (Độ phủ)** | 0.9158 | 0.9975 | 0.9743 | 0.9574 | 0.9887 |
| | **F1-Score** | 0.9521 | 0.9881 | 0.9392 | 0.9521 | 0.9930 |
| **Fold 5** | **Precision (Tự tin)** | 0.9916 | 1.0000 | 0.9089 | 0.9173 | 0.9960 |
| | **Recall (Độ phủ)** | 0.9237 | 0.9987 | 0.9753 | 0.9150 | 1.0000 |
| | **F1-Score** | 0.9564 | 0.9994 | 0.9409 | 0.9162 | 0.9980 |

### Kịch bản: S3_TrainSE_TestSA

| Lần chạy | Chỉ số | Walk | Run | Idle | Trans | Fall |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **1 Lần duy nhất** | **Precision (Tự tin)** | 0.9608 | 0.9768 | 0.8980 | 0.8924 | 0.9982 |
| | **Recall (Độ phủ)** | 0.9393 | 0.9974 | 0.9562 | 0.8771 | 0.9586 |
| | **F1-Score** | 0.9499 | 0.9870 | 0.9262 | 0.8847 | 0.9780 |

### Kịch bản: S4_TrainSA_TestSE

| Lần chạy | Chỉ số | Walk | Run | Idle | Trans | Fall |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **1 Lần duy nhất** | **Precision (Tự tin)** | 0.9455 | 0.9917 | 0.9112 | 0.9016 | 0.8791 |
| | **Recall (Độ phủ)** | 0.9290 | 0.9401 | 0.9488 | 0.9179 | 0.9893 |
| | **F1-Score** | 0.9371 | 0.9652 | 0.9296 | 0.9096 | 0.9310 |

### Kịch bản: S5_Both

| Lần chạy | Chỉ số | Walk | Run | Idle | Trans | Fall |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | **Precision (Tự tin)** | 0.9689 | 0.9952 | 0.9264 | 0.9417 | 0.9989 |
| | **Recall (Độ phủ)** | 0.9452 | 0.9575 | 0.9785 | 0.9482 | 0.9979 |
| | **F1-Score** | 0.9569 | 0.9760 | 0.9517 | 0.9450 | 0.9984 |
| **Fold 2** | **Precision (Tự tin)** | 0.9791 | 0.9924 | 0.9411 | 0.9443 | 0.9978 |
| | **Recall (Độ phủ)** | 0.9560 | 0.9930 | 0.9750 | 0.9384 | 0.9871 |
| | **F1-Score** | 0.9674 | 0.9927 | 0.9577 | 0.9413 | 0.9924 |
| **Fold 3** | **Precision (Tự tin)** | 0.9804 | 0.9985 | 0.9465 | 0.9251 | 0.9973 |
| | **Recall (Độ phủ)** | 0.9642 | 0.9821 | 0.9678 | 0.9416 | 0.9952 |
| | **F1-Score** | 0.9723 | 0.9902 | 0.9571 | 0.9333 | 0.9963 |
| **Fold 4** | **Precision (Tự tin)** | 0.9752 | 0.9780 | 0.9096 | 0.9311 | 0.9973 |
| | **Recall (Độ phủ)** | 0.9187 | 0.9971 | 0.9562 | 0.9558 | 0.9847 |
| | **F1-Score** | 0.9461 | 0.9875 | 0.9323 | 0.9433 | 0.9909 |
| **Fold 5** | **Precision (Tự tin)** | 0.9854 | 0.9942 | 0.8984 | 0.9226 | 0.9900 |
| | **Recall (Độ phủ)** | 0.9110 | 0.9964 | 0.9710 | 0.9226 | 0.9960 |
| | **F1-Score** | 0.9468 | 0.9953 | 0.9333 | 0.9226 | 0.9930 |

