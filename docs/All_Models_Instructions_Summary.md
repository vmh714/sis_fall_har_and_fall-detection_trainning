# TỔNG HỢP TÀI LIỆU CÁC PHIÊN BẢN MODEL

*Tài liệu này được tự động tổng hợp từ các file `instruction.md` trong từng thư mục huấn luyện.*

---

# 🏷️ Phiên bản: `train_v2_kq`

## 1. Tổng quan Model
- **Mục đích:** Nhận diện hoạt động con người (Human Activity Recognition - HAR) dựa trên dữ liệu cảm biến IMU.
- **Quy mô đầu ra:** Phân loại thành **4 lớp (classes)** với hàm kích hoạt `softmax` ở lớp cuối.

## 2. Chiến lược Data & Gắn nhãn
- **Phương pháp gán nhãn:** Dựa trên quy ước đặt tên file `Dxx_Sxx_Rxx_Wxxx` (trong đó `Dxx` là mã định danh nhãn).
- **Phân chia dữ liệu:** Sử dụng kỹ thuật **LSO (Leave-Subject-Out)**:
    - `TRAIN_SUBJECTS`: Dữ liệu dùng để huấn luyện.
    - `VAL_SUBJECTS`: Dữ liệu dùng để tinh chỉnh và kiểm tra validation.
    - `TEST_SUBJECTS`: Dữ liệu độc lập dùng để đánh giá mô hình cuối cùng.
- **Cấu trúc dữ liệu:** Mỗi mẫu dữ liệu là một cửa sổ (window) có kích thước cố định là **200 bước thời gian (timesteps)** với 6 đặc trưng (3 trục Accel + 3 trục Gyro).

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc dữ liệu đầu vào:** Dữ liệu được đọc từ các file CSV, mỗi file đại diện cho một window 200 mẫu.
- **Tối ưu hiệu năng:** 
    - Sử dụng `ThreadPoolExecutor` (8 workers) để tăng tốc độ tải dữ liệu I/O từ đĩa.
    - Cơ chế **Caching**: Dữ liệu đã xử lý được lưu dưới dạng file `.npy` để tăng tốc độ khởi tạo cho các lần chạy sau.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Hybrid CNN-LSTM.
    - **CNN (Spatial Extraction):** Gồm 2 lớp `Conv1D` (64 filters, kernel 3) kết hợp với `MaxPooling1D` để trích xuất đặc trưng không gian cục bộ.
    - **LSTM (Temporal Learning):** Gồm 2 lớp `LSTM` (128 và 64 units) để học mối quan hệ tuần tự/thời gian.
- **Kỹ thuật tối ưu:**
    - **Regularization:** Sử dụng `Dropout` (tỉ lệ 0.4 sau CNN và 0.5 sau LSTM) để chống overfitting.
    - **Xử lý mất cân bằng:** Sử dụng `class_weight='balanced'` để tính toán trọng số cho từng lớp, giúp mô hình tập trung vào các class thiểu số.
    - **Optimizer:** `Adam` với learning rate `1e-3`.
    - **Loss Function:** `sparse_categorical_crossentropy`.

---

# 🏷️ Phiên bản: `train_v3_kq`

## 1. Tổng quan Model
- **Mục đích:** Nhận dạng hành động người dùng (HAR - Human Activity Recognition) dựa trên dữ liệu cảm biến IMU (Gia tốc và Góc quay).
- **Quy mô đầu ra:** Phân loại 4 lớp (4 classes) hành động.

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc dữ liệu:** Dữ liệu được chia theo cửa sổ (window) cố định với 200 mẫu/window.
- **Phương pháp phân chia:** Sử dụng chiến lược **Leave-Subject-Out (LSO)**, dữ liệu được tách biệt hoàn toàn giữa các tập Train, Val và Test dựa trên ID đối tượng (`subject_id`) để đảm bảo tính tổng quát hóa cho mô hình.
- **Gắn nhãn:** Sử dụng file định dạng `D01_SA01_R01_W000`, trong đó `D01` tương ứng với mã nhãn (Label code) được ánh xạ qua `LABEL_MAP`.

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc input:** Dữ liệu bao gồm 6 cột (ax, ay, az, gx, gy, gz).
- **Định dạng dữ liệu:** Loại bỏ các window không đủ 200 mẫu để duy trì tính nhất quán đầu vào.
- **Tối ưu hóa I/O:** Sử dụng `ThreadPoolExecutor` để đọc dữ liệu song song và hệ thống `cache` (file `.npy`) để tăng tốc độ tải dữ liệu cho các lần huấn luyện tiếp theo.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Kiến trúc lai (Hybrid) kết hợp:
    - **CNN (Conv1D):** Hai lớp trích xuất đặc trưng cục bộ (64 filters, kernel size 3) kết hợp với `MaxPooling1D` và `Dropout(0.3)`.
    - **LSTM:** Hai lớp RNN (128 và 64 units) để học phụ thuộc tuần tự theo thời gian, kết hợp `Dropout(0.4)` để chống overfitting.
    - **Classifier:** Lớp Dense(32) dẫn tới đầu ra Softmax(4).
- **Kỹ thuật tối ưu:**
    - **Optimizer:** Adam (Learning rate = 5e-4).
    - **Class Weights:** Áp dụng `compute_class_weight` (chế độ 'balanced') để xử lý tình trạng mất cân bằng dữ liệu giữa các lớp nhãn.
    - **Loss function:** `sparse_categorical_crossentropy`.

---

# 🏷️ Phiên bản: `train_v4_kq`

## 1. Tổng quan Model
- **Mục đích:** Nhận diện hoạt động con người (Human Activity Recognition - HAR) dựa trên dữ liệu cảm biến quán tính (IMU).
- **Đầu ra:** Mô hình phân loại 4 lớp (classes) hoạt động khác nhau.

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc:** Sử dụng chiến lược chia tập dữ liệu **LSO (Leave-Subject-Out)** dựa trên `subject_id` để đảm bảo mô hình có khả năng tổng quát hóa trên đối tượng mới.
- **Dữ liệu đầu vào:** Mỗi mẫu (window) là một file CSV có kích thước cố định là 200 bước thời gian (time-steps) với 6 đặc trưng (ax, ay, az, gx, gy, gz).
- **Gắn nhãn:** Dựa trên tiền tố `label_code` trong tên tệp tin (ví dụ: `D01_SA01_R01_W000` được ánh xạ thông qua `LABEL_MAP`).

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc dữ liệu:** Chuẩn hóa đầu vào về dạng `(200, 6)`.
- **Cơ chế tải:** Sử dụng `ThreadPoolExecutor` để đọc dữ liệu song song nhằm tối ưu hóa I/O, kết hợp lưu trữ dạng Cache (`.npy`) để tăng tốc độ khởi tạo.
- **Lưu ý:** Code hiện tại giả định dữ liệu đã được chuẩn hóa (scale) hoặc ở định dạng thô phù hợp với kiến trúc; các tham số clip (8g) và scale cụ thể sẽ được áp dụng tại bước chuẩn bị file CSV đầu vào (theo logic ngoài phạm vi hàm `prepare_dataset`).

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Kiến trúc lai **CNN-LSTM** (Sequential model).
    - **Feature Extraction:** 2 lớp `Conv1D` (kernel size 3) kết hợp `BatchNormalization` và `MaxPooling1D`.
    - **Sequence Modeling:** 2 lớp `LSTM` (64 và 32 units) để trích xuất phụ thuộc thời gian.
    - **Regularization:** Sử dụng `L2 Regularization` (0.001) cho tất cả các lớp và `Dropout` (0.3 - 0.4) để tránh quá mức (overfitting).
- **Kỹ thuật tối ưu:**
    - **Class Weights:** Sử dụng `compute_class_weight='balanced'` để xử lý tình trạng mất cân bằng dữ liệu giữa các lớp.
    - **Loss Function:** `sparse_categorical_crossentropy`.
    - **Optimizer:** `Adam` (learning rate 1e-3).

---

# 🏷️ Phiên bản: `train_v5_kq`

## 1. Tổng quan Model
- **Mục đích:** Mô hình lai (hybrid) tích hợp phát hiện ngã (Fall Detection) và nhận diện hoạt động thường ngày (HAR).
- **Quy mô:** Đầu ra gồm 4 lớp (4 classes) sử dụng hàm kích hoạt Softmax.

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc:** Sử dụng dữ liệu dạng cửa sổ (windowed data) với độ dài cố định 200 mẫu/file.
- **Phương pháp phân chia:** Sử dụng chiến lược **LSO (Leave-Subject-Out)**: chia tập Train/Val/Test dựa trên ID đối tượng (`subject_id`) để đảm bảo tính tổng quát hóa cho người mới.
- **Nhãn:** Được trích xuất từ tiền tố của tên file (`label_code`) thông qua từ điển `LABEL_MAP`.

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc dữ liệu:** Input bao gồm 6 kênh (3 trục gia tốc - Accel, 3 trục góc quay - Gyro).
- **Kiểm soát chất lượng:** Loại bỏ các file không đạt chuẩn kích thước (không đủ 200 mẫu).
- **Tăng tốc xử lý:** Sử dụng `ThreadPoolExecutor` để đọc dữ liệu I/O song song và cơ chế lưu trữ Cache (`.npy`) để tối ưu thời gian tải dữ liệu cho các lần chạy sau.

## 4. Kiến trúc mô hình (Architecture)
- **Kiến trúc:** **Two-Stream (Phân nhánh)**:
    - **Nhánh Fall Expert:** Sử dụng `Conv1D` kết hợp `GlobalMaxPooling1D` để trích xuất các đặc trưng đột biến mạnh (cú ngã).
    - **Nhánh HAR Expert:** Sử dụng `MaxPooling1D` kết hợp hai lớp `LSTM` để học các phụ thuộc theo thời gian (chuỗi nhịp điệu) của hoạt động.
    - **Hợp nhất:** Kết hợp đặc trưng từ hai nhánh qua lớp `Concatenate`.
- **Kỹ thuật tối ưu:**
    - **Regularization:** Sử dụng `L2 Regularization` (0.001) cho hầu hết các lớp Dense và Conv để tránh Overfitting.
    - **Class Weight:** Sử dụng kỹ thuật `compute_class_weight` với tham số `balanced` để giải quyết vấn đề mất cân bằng dữ liệu giữa các lớp.
    - **Normalization:** Áp dụng `BatchNormalization` để ổn định quá trình hội tụ.
    - **Optimizer:** Adam (learning_rate=1e-3) và Loss function là `sparse_categorical_crossentropy`.

---

# 🏷️ Phiên bản: `train_v6_kq`

## 1. Tổng quan Model
- **Mục đích:** Mô hình được thiết kế để phân loại hoạt động con người (HAR) và phát hiện ngã (Fall Detection) dựa trên dữ liệu cảm biến IMU (Gia tốc và Góc quay).
- **Quy mô đầu ra:** Phân loại thành **4 lớp (4 classes)**, được xác định thông qua hàm kích hoạt Softmax tại lớp cuối cùng.

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc dữ liệu:** Dữ liệu đầu vào là các cửa sổ thời gian (windows) với kích thước cố định **200 mẫu (samples)** cho mỗi file CSV.
- **Gắn nhãn:** Sử dụng hàm `parse_filename_info` để giải mã nhãn từ tên file (định dạng `D01_SA01_R01_W000`).
- **Phân chia dữ liệu:** Thực hiện theo phương pháp **LSO (Leave-Subject-Out)**, chia tập Train/Val/Test dựa trên định danh đối tượng (`subject_id`) để đảm bảo tính tổng quát hóa.

## 3. Tiền xử lý (Preprocessing)
- **Định dạng đầu vào:** Dữ liệu 6 trục (ax, ay, az, gx, gy, gz).
- **Tối ưu hóa:** Sử dụng `ThreadPoolExecutor` để đọc song song dữ liệu từ ổ đĩa và áp dụng cơ chế lưu **Cache (.npy)** giúp rút ngắn thời gian chuẩn bị dữ liệu cho các lần chạy tiếp theo.
- **Xử lý mất cân bằng:** Sử dụng kỹ thuật `compute_class_weight` với tham số `class_weight='balanced'` để tính toán trọng số cho các lớp, giúp mô hình tập trung hơn vào các lớp thiểu số.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Sử dụng kiến trúc **Trunk-and-Branches (Late-Branching)**:
    - **Trunk (Thân cây):** Gồm 2 lớp `Conv1D` kết hợp `BatchNormalization` và `MaxPooling1D` để trích xuất đặc trưng và lọc nhiễu chung.
    - **Fall Expert (Nhánh 1):** Sử dụng `GlobalMaxPooling1D` để nhận diện các thay đổi đột ngột (đặc trưng của việc ngã).
    - **HAR Expert (Nhánh 2):** Sử dụng kiến trúc 2 tầng `LSTM` để phân tích tính chuỗi thời gian của các hoạt động hàng ngày.
- **Hợp nhất (Merge):** Kết hợp kết quả từ hai nhánh qua lớp `Concatenate` trước khi đưa vào lớp phân loại cuối cùng.
- **Kỹ thuật tối ưu:**
    - **Regularization:** Sử dụng `L2 Regularization` (0.001) trên các lớp Dense/Conv1D và `Dropout` (0.3 - 0.4) để tránh quá mức (overfitting).
    - **Optimizer:** `Adam` với learning rate `1e-3`.
    - **Loss Function:** `sparse_categorical_crossentropy`.

---

# 🏷️ Phiên bản: `train_v7_kq`

## 1. Tổng quan Model
- **Mục đích:** Phát hiện té ngã (Fall detection) và phân loại hoạt động thể chất (HAR).
- **Quy mô:** Mô hình đầu ra gồm 4 lớp (4 classes), sử dụng hàm kích hoạt Softmax để phân loại đa lớp.

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc nhãn:** Dữ liệu được gán nhãn dựa trên tiền tố của tên file (định dạng `D01_SA01_...`).
- **Phân chia:** Sử dụng chiến lược **LSO (Leave-Subject-Out)** để chia tập dữ liệu thành Train, Validation và Test dựa trên ID đối tượng (`subject_id`), đảm bảo mô hình không học vẹt đặc trưng cá nhân.

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc dữ liệu đầu vào:** Mỗi cửa sổ (window) có độ dài cố định là 200 mẫu.
- **Tính năng mở rộng:** Từ 6 cột dữ liệu thô ban đầu (ax, ay, az, gx, gy, gz), mô hình tính toán thêm 3 đặc trưng vật lý quan trọng:
    - **Pitch & Roll:** Tính từ dữ liệu gia tốc (góc nghiêng).
    - **SVM (Signal Vector Magnitude):** Tính từ gia tốc để đo cường độ vận động.
- **Kích thước vector:** Tổng cộng 9 đặc trưng (9 features) cho mỗi điểm thời gian.
- **Lưu trữ:** Hỗ trợ lưu trữ/tải cache dạng `.npy` để tăng tốc độ huấn luyện.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Kiến trúc **Trunk-and-Branches (Late-Branching)**:
    - **Thân cây (Trunk):** Sử dụng 2 lớp `Conv1D` kết hợp `BatchNormalization` và `MaxPooling1D` để trích xuất đặc trưng không gian và lọc nhiễu.
    - **Nhánh 1 (Fall Expert):** Sử dụng `GlobalMaxPooling1D` để nhận diện các biến đổi đột ngột (đặc trưng của té ngã).
    - **Nhánh 2 (HAR Expert):** Sử dụng 2 lớp `LSTM` để nắm bắt tính chuỗi thời gian và nhịp điệu của các hoạt động.
- **Hợp nhất:** Kết hợp kết quả từ hai nhánh qua `Concatenate` trước khi đưa vào lớp `Dense` phân loại cuối cùng.
- **Kỹ thuật tối ưu:**
    - **Class Weights:** Sử dụng `compute_class_weight` (balanced) để xử lý mất cân bằng dữ liệu giữa các lớp.
    - **Regularization:** Áp dụng `L2 Regularization` (0.001) trên các lớp Dense/Conv1D và `Dropout` (0.3 - 0.4) để tránh Overfitting.
    - **Trình tối ưu:** Adam (learning rate 1e-3).

---

# 🏷️ Phiên bản: `train_v8_kq_legacy`

## 1. Tổng quan Model
- **Mục đích:** Nhận diện hoạt động dựa trên dữ liệu cảm biến IMU (thời gian thực).
- **Quy mô:** Thiết kế mô hình với 4 lớp đầu ra (`n_classes=4`), tập trung vào sự tối ưu hóa cho các thiết bị nhúng (MCU) thông qua định dạng TFLite Micro và chuẩn bị cho Full INT8 Quantization.

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc dữ liệu:** Dữ liệu đầu vào là các "window" cố định có độ dài 200 mẫu (samples/window).
- **Phân bổ:** Dữ liệu được chia theo `subject_id` (người thực hiện) vào 3 tập riêng biệt: `Train`, `Val`, và `Test` để đảm bảo tính tổng quát hóa.
- **Xử lý nhãn:** Sử dụng `LABEL_MAP` để ánh xạ từ mã nhãn trong tên file (định dạng `LabelCode_SubjectID`) sang giá trị số nguyên.

## 3. Tiền xử lý (Preprocessing)
- **Cắt gọt (Clipped):** Dữ liệu được lọc theo độ dài cố định 200 mẫu/file. Các file không đạt chuẩn bị loại bỏ (`len(df) != 200`).
- **Scale:** Dữ liệu được tải dưới dạng mảng `float32`. Mặc dù code không hiện rõ thông số scale cụ thể, mô hình thực hiện `BatchNormalization` ngay tại lớp đầu vào để chuẩn hóa phân phối dữ liệu (trung bình và phương sai) trước khi đưa vào các khối tích chập.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Temporal Convolutional Network (TCN) bao gồm 2 stacks, mỗi stack chứa 4 khối `Conv1D` với các tốc độ giãn nở (dilation rates) khác nhau [1, 2, 4, 8].
- **Cấu trúc chi tiết:**
    - Sử dụng `Padding='causal'` đảm bảo tính nhân quả cho dữ liệu chuỗi thời gian.
    - Kết nối tắt (Residual connections) được áp dụng để tránh suy giảm gradient.
    - Sử dụng `GlobalAveragePooling1D` để nén đặc trưng trước lớp phân loại cuối cùng.
- **Kỹ thuật tối ưu:**
    - **Cân bằng lớp:** Sử dụng `class_weight='balanced'` để xử lý tình trạng mất cân bằng dữ liệu đầu vào.
    - **Điều chuẩn:** Áp dụng `Dropout(0.2)` trong các block TCN và `Dropout(0.3)` sau lớp pooling.
    - **Định dạng triển khai:** Mô hình được tối ưu hóa cho **TFLite INT8 Quantization**, giúp giảm kích thước và tăng tốc độ suy luận trên vi điều khiển.

---

# 🏷️ Phiên bản: `train_v8_kq`

## 1. Tổng quan Model
- **Mục đích:** Nhận diện hoạt động dựa trên dữ liệu cảm biến IMU (cảm biến gia tốc và con quay hồi chuyển) trên thiết bị nhúng.
- **Quy mô:** Thiết kế cho bài toán phân loại đa lớp (mặc định `n_classes=4`), tối ưu hóa để chuyển đổi sang **TFLite Micro** với định dạng **Full INT8 Quantization**.

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc dữ liệu:** Dữ liệu được chia theo cửa sổ (windowed) với độ dài cố định là **200 mẫu (samples)** mỗi file.
- **Gán nhãn:** Nhãn được trích xuất trực tiếp từ tên file thông qua tiền tố (`label_code`) và được ánh xạ qua từ điển `LABEL_MAP`.
- **Phân chia tập:** Dữ liệu được chia tập Train/Val/Test dựa trên định danh người dùng (`subject_id`) để đảm bảo tính độc lập giữa các tập dữ liệu.
- **Xử lý mất cân bằng:** Sử dụng kỹ thuật `class_weight='balanced'` để tính toán trọng số, đặc biệt ưu tiên/phạt lỗi cho các lớp hiếm (như Fall - té ngã).

## 3. Tiền xử lý (Preprocessing)
- **Định dạng dữ liệu:** Input được chuẩn hóa về dạng `(200, features)`.
- **Đặc điểm kỹ thuật:**
    - Sử dụng `BatchNormalization` ngay tại lớp đầu vào để ổn định phân phối dữ liệu (giảm ảnh hưởng của việc lệch thang đo giữa các cảm biến).
    - Dữ liệu được lưu trữ dạng cache `.npy` để tối ưu hóa quy trình training và đảm bảo tính nhất quán của dữ liệu đầu vào giữa các phiên chạy.

## 4. Kiến trúc mô hình (Architecture)
- **Kiến trúc chính:** **Temporal Convolutional Network (TCN)** với 2 stacks, mỗi stack gồm 4 khối (tổng cộng 8 lớp tích chập).
- **Các thành phần kỹ thuật:**
    - **Dilated Conv1D:** Sử dụng `dilation_rate` lần lượt [1, 2, 4, 8] để mở rộng trường nhìn (receptive field) mà không làm tăng tham số đáng kể.
    - **Kết nối tắt (Residual Connection):** Sử dụng các nhánh cộng (Add) để tránh vấn đề triệt tiêu đạo hàm trong mạng sâu.
    - **Regularization:** Kết hợp `Dropout (0.2 - 0.3)` và `BatchNormalization` sau mỗi khối để chống quá khớp (overfitting).
    - **Pooling:** Sử dụng `GlobalAveragePooling1D` để nén đặc trưng trước khi đưa vào lớp phân loại `Dense`.
    - **Tối ưu hóa:** Sử dụng bộ tối ưu `Adam` với learning rate `1e-3` và hàm mất mát `sparse_categorical_crossentropy`.
    - **Triển khai:** Hỗ trợ xuất mô hình sang TFLite với cơ chế `Full INT8 Quantization` để tối ưu cho chip MCU.

---

# 🏷️ Phiên bản: `train_v9_kq`

## 1. Tổng quan Model
- **Mục đích:** Phân loại hành vi (Human Activity Recognition - HAR) từ dữ liệu IMU.
- **Quy mô:** Mô hình được thiết kế với đầu ra gồm **4 lớp (classes)**, tối ưu hóa để triển khai trên các thiết bị nhúng (MCU) thông qua TFLite Micro.

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc dữ liệu:** Dữ liệu đầu vào dạng cửa sổ (windowed) với độ dài cố định **200 mẫu/window**.
- **Phân chia tập dữ liệu:** Dựa trên `subject_id` (người thực hiện) để tách biệt hoàn toàn tập Train, Validation và Test, tránh tình trạng rò rỉ dữ liệu (data leakage).
- **Gắn nhãn:** Dựa trên tiền tố của tên file (được định nghĩa trong `LABEL_MAP`).
- **Cân bằng lớp:** Sử dụng kỹ thuật `class_weight='balanced'` để xử lý dữ liệu mất cân bằng (có đề cập đến việc áp dụng trọng số tăng cường cho lớp 'Fall').

## 3. Tiền xử lý (Preprocessing)
- **Chuẩn hóa:** Dữ liệu sau khi nạp được xử lý qua `BatchNormalization` ở lớp đầu tiên của mô hình.
- **Tối ưu Pipeline:**
    - Sử dụng `ThreadPoolExecutor` (8 workers) để nạp dữ liệu song song.
    - Cơ chế lưu Cache dữ liệu (`.npy`) giúp giảm thiểu thời gian đọc file trong các lần chạy sau.
- **Lưu ý kỹ thuật:** Mặc dù code trích xuất không liệt kê cụ thể hệ số nhân cho Accel/Gyro, nhưng logic `prepare_dataset` đảm bảo tính đồng nhất (200 mẫu) trước khi đưa vào mô hình.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** **Temporal Convolutional Network (TCN)** với cấu trúc 2 stacks, mỗi stack gồm 4 block sử dụng `dilation_rate` [1, 2, 4, 8] để mở rộng trường nhìn (receptive field) theo thời gian.
- **Đặc điểm thiết kế:**
    - Sử dụng **Residual Connections** (Add) để hỗ trợ huấn luyện mạng sâu.
    - Sử dụng **Causal Padding** để đảm bảo tính nhân quả (phù hợp với dữ liệu chuỗi thời gian).
    - **Pooling:** `GlobalAveragePooling1D` để nén đặc trưng trước lớp phân loại.
    - **Dropout:** Tỷ lệ 0.2 trong các block và 0.3 ở lớp Fully Connected để giảm Overfitting.
- **Kỹ thuật tối ưu:**
    - Tối ưu hóa cho TFLite (Full INT8 Quantization).
    - Optimizer: Adam (Learning rate 1e-3).
    - Loss function: `sparse_categorical_crossentropy`.

---

# 🏷️ Phiên bản: `train_v10_kq`

## 1. Tổng quan Model
- **Mục đích:** Phân loại hành động người dùng dựa trên dữ liệu cảm biến IMU (cảm biến chuyển động).
- **Quy mô:** Hỗ trợ 4 lớp đầu ra (n_classes=4).

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc:** Sử dụng các file CSV được cắt thành các đoạn (window) có độ dài cố định là 200 mẫu (samples).
- **Phân chia dữ liệu:** Dữ liệu được chia theo `subject_id` (ID người thực hiện) thành tập Huấn luyện (Train), Kiểm chứng (Val) và Kiểm thử (Test) để đảm bảo tính độc lập giữa các tập.
- **Ánh xạ:** Sử dụng từ điển `LABEL_MAP` (dựa trên tiền tố của tên file) để quy đổi nhãn văn bản sang nhãn số nguyên.

## 3. Tiền xử lý (Preprocessing)
- **Cắt gọt:** Dữ liệu đầu vào bắt buộc phải có chiều dài chính xác là 200 samples/window. Các dữ liệu không thỏa mãn điều kiện này sẽ bị loại bỏ trong quá trình nạp.
- **Định dạng:** Dữ liệu sau khi xử lý được cache dưới định dạng `.npy` để tăng tốc độ nạp cho các lần huấn luyện sau.
- **Chuẩn hóa:** Áp dụng `BatchNormalization` ngay lớp đầu vào của mô hình để ổn định phân phối dữ liệu (thay vì tiền xử lý thủ công trên file gốc).

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Temporal Convolutional Network (TCN) kết hợp kỹ thuật `Causal Conv1D` để xử lý dữ liệu chuỗi thời gian.
- **Cấu trúc lớp:** 
    - 2 chồng (stacks) convolution với các hệ số giãn nở (dilation rates: 1, 2, 4, 8) nhằm tăng trường nhìn (receptive field).
    - Sử dụng `Residual connection` (kết nối tắt) để tránh triệt tiêu gradient.
    - `GlobalAveragePooling1D` được dùng để nén đặc trưng trước khi đi vào lớp `Dense` đầu ra.
- **Kỹ thuật tối ưu:**
    - **Class weights:** Sử dụng `balanced` class weights để xử lý vấn đề mất cân bằng dữ liệu giữa các lớp.
    - **Regularization:** Sử dụng `Dropout` (0.2 - 0.3) và `BatchNormalization` xuyên suốt mô hình.
    - **Hậu xử lý:** Hỗ trợ chuyển đổi sang định dạng `TFLite (int8)` để tối ưu hóa việc triển khai trên vi điều khiển (MCU).

---

# 🏷️ Phiên bản: `train_v11_kq`

## 1. Tổng quan Model
- **Mục đích**: Phân loại các hoạt động dựa trên dữ liệu cảm biến IMU (Inertial Measurement Unit) theo chuỗi thời gian.
- **Quy mô**: Mô hình được thiết kế để phân loại 4 lớp (default `n_classes=4`).

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc dữ liệu**: Dữ liệu đầu vào được chia thành các cửa sổ (windows) cố định với độ dài 200 mẫu (samples).
- **Phân chia tập dữ liệu**: Dữ liệu được gán nhãn dựa trên tiền tố của tên file (`label_code`) và được phân tách theo `subject_id` thành 3 tập: Train, Validation và Test để đảm bảo tính độc lập giữa các đối tượng.
- **Cân bằng dữ liệu**: Sử dụng kỹ thuật tính toán `class_weight='balanced'` để xử lý tình trạng mất cân bằng giữa các lớp trong quá trình huấn luyện.

## 3. Tiền xử lý (Preprocessing)
- **Độ dài cửa sổ**: Cố định 200 điểm dữ liệu cho mỗi sample đầu vào.
- **Chuẩn hóa**: Dữ liệu được nạp trực tiếp qua các file đã được windowing, sau đó được mô hình hóa thành dạng `float32`.
- **Thông số kỹ thuật**: Mặc dù mã nguồn không trực tiếp hiển thị hệ số scale cụ thể, mô hình sử dụng `BatchNormalization` ngay từ lớp đầu vào để ổn định phân phối dữ liệu (giảm ảnh hưởng của việc chênh lệch giá trị cảm biến).

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng**: Kiến trúc **TCN (Temporal Convolutional Network)** tối ưu cho dữ liệu chuỗi thời gian.
- **Thành phần chính**:
    - **Initial Layer**: Một lớp `Conv1D` (kernel=1) để chuẩn hóa số kênh lên 32.
    - **TCN Blocks**: 2 khối (stacks), mỗi khối chứa 4 lớp tích chập với các tốc độ giãn nở (dilation rates) lần lượt là [1, 2, 4, 8] giúp mô hình học các phụ thuộc xa.
    - **Cơ chế Shortcut**: Sử dụng phép cộng `Add()` (Residual connection) để duy trì luồng gradient và tránh suy giảm tín hiệu qua các lớp sâu.
    - **Pooling**: Sử dụng `GlobalAveragePooling1D` để nén thông tin chuỗi trước khi đưa vào lớp phân loại cuối cùng.
- **Kỹ thuật tối ưu**:
    - **Regularization**: Sử dụng `Dropout` (0.2 trong các block TCN và 0.3 trước lớp Dense) để tránh Overfitting.
    - **Optimizer**: Adam (learning rate = 1e-3).
    - **Loss Function**: `sparse_categorical_crossentropy`.

---

# 🏷️ Phiên bản: `train_v12_kq_7ch`

## 1. Tổng quan Model
- **Mục đích**: Nhận diện hoạt động (Activity Recognition) dựa trên dữ liệu cảm biến IMU.
- **Quy mô**: Phân loại 4 lớp (n_classes=4).

## 2. Chiến lược Data & Gắn nhãn
- **Dữ liệu đầu vào**: Dữ liệu IMU gồm 6 trục gốc (3 Accel, 3 Gyro).
- **Kỹ thuật bổ sung**: Tích hợp thêm kênh **SVM (Signal Vector Magnitude)** từ 6 trục gốc, tạo thành tập dữ liệu đầu vào 7 trục.
- **Cân bằng dữ liệu**: Sử dụng `class_weight='balanced'` trong quá trình huấn luyện để xử lý vấn đề mất cân bằng giữa các lớp.

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc**: Input được chuẩn hóa thông qua `BatchNormalization` ngay tại lớp đầu vào.
- **Nhiễu**: Áp dụng `GaussianNoise(0.01)` để tăng tính ổn định cho mô hình, đặc biệt bảo vệ các đặc trưng của kênh SVM vốn rất nhạy cảm.
- **Lưu ý**: Dữ liệu được nạp từ các file `.npy` đã được lưu cache trước đó.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng**: Temporal Convolutional Network (TCN) với cơ chế Residual Connection.
- **Cấu hình lớp Conv1D**:
    - **Stacking**: 2 stack, mỗi stack bao gồm các lớp với `dilation_rate` lần lượt là [1, 2, 4, 8].
    - **Filter**: 64 filters cho lớp đầu tiên, 32 filters cho các lớp tiếp theo.
    - **Kernel size**: 3, padding 'valid'.
- **Kỹ thuật tối ưu**:
    - **Residual Connection**: Sử dụng `Cropping1D` để căn chỉnh kích thước tensor trước khi cộng (Add) vào nhánh residual.
    - **Regularization**: Sử dụng `Dropout` (0.2 trong các block và 0.3 trước lớp đầu ra) để chống overfitting.
    - **Pooling**: `GlobalAveragePooling1D` để nén dữ liệu trước khi đưa vào lớp Dense cuối cùng.
    - **Optimizer**: Adam với learning rate = 1e-3, hàm mất mát `sparse_categorical_crossentropy`.

---

# 🏷️ Phiên bản: `train_v12_kq`

## 1. Tổng quan Model
- **Mục đích**: Nhận diện hoạt động (Human Activity Recognition) dựa trên dữ liệu cảm biến IMU (Accel/Gyro).
- **Quy mô**: Phân loại theo 4 lớp đầu ra (`n_classes=4`).

## 2. Chiến lược Data & Gắn nhãn
- **Gắn nhãn**: Dựa trên tiền tố của tên tệp tin (`label_code`) thông qua dictionary `LABEL_MAP`.
- **Phân tách**: Dữ liệu được chia tập huấn luyện (train), kiểm thử (val) và đánh giá (test) dựa trên `subject_id` (ID người tham gia), đảm bảo không bị rò rỉ dữ liệu giữa các tập.
- **Cấu trúc**: Mỗi mẫu dữ liệu được yêu cầu cố định ở độ dài 200 time-steps.

## 3. Tiền xử lý (Preprocessing)
- **Cửa sổ thời gian**: Cố định `window size = 200`.
- **Cân bằng dữ liệu**: Sử dụng kỹ thuật `class_weight='balanced'` để tính toán trọng số cho từng lớp, giúp xử lý vấn đề mất cân bằng dữ liệu trong quá trình huấn luyện.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng**: Temporal Convolutional Network (TCN) tùy chỉnh cho MCU.
- **Thành phần chính**:
    - **Stacking**: 2 khối (stacks), mỗi khối gồm 4 lớp `Conv1D` với các `dilation_rate` lần lượt là [1, 2, 4, 8].
    - **Cơ chế**: Sử dụng `padding='valid'` kết hợp `Cropping1D` để căn chỉnh kích thước `residual` thay vì dùng `causal padding`, tối ưu hóa cho việc chuyển đổi sang định dạng TFLite (tránh toán tử `Pad`/`SpaceToBatchND`).
    - **Regularization**: Sử dụng `BatchNormalization` sau mỗi lớp `Conv1D` và `Dropout` (0.2 trong block, 0.3 ở lớp Global Pooling) để giảm overfitting.
    - **Pooling**: `GlobalAveragePooling1D` để nén đặc trưng trước lớp phân loại.
- **Tối ưu**: Sử dụng `Adam` optimizer với learning rate `1e-3` và hàm mất mát `sparse_categorical_crossentropy`.

---

# 🏷️ Phiên bản: `train_v12_kq_6ch`

## 1. Tổng quan Model
- **Mục đích:** Phân loại hành động dựa trên dữ liệu cảm biến IMU (Accel & Gyro).
- **Quy mô:** Thiết kế cho bài toán đa lớp (mặc định `n_classes=4`), phù hợp với các thiết bị biên (MCU).

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc dữ liệu:** Dữ liệu đầu vào được cắt thành các cửa sổ (window) cố định với độ dài **200 mẫu (samples)**.
- **Phân chia tập dữ liệu:** Sử dụng chiến lược tách theo người thực hiện (`subject_id`) thành 3 tập riêng biệt: `Train`, `Val` và `Test` để đảm bảo tính tổng quát hóa.
- **Gán nhãn:** Dựa trên tiền tố của tên file (`label_code` trong `LABEL_MAP`).

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc dữ liệu:** Chỉ giữ lại 6 trục đầu tiên (tương ứng với 3 trục Accelerometer và 3 trục Gyroscope).
- **Bộ nhớ đệm:** Sử dụng cơ chế `cache` (.npy files) để tăng tốc độ tải dữ liệu trong các lần huấn luyện sau.
- **Chuẩn hóa:** Sử dụng `BatchNormalization` ngay tại lớp đầu vào của mô hình để ổn định phân phối dữ liệu đầu vào.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Temporal Convolutional Network (TCN) với 2 lớp xếp chồng (stacks), mỗi lớp sử dụng các dilation rate [1, 2, 4, 8] để mở rộng trường tiếp nhận (receptive field).
- **Cấu trúc bổ trợ:**
    - Sử dụng **Residual Connection** (Add layer) để giảm thiểu hiện tượng mất mát gradient.
    - **Cropping1D:** Xử lý sự chênh lệch kích thước do sử dụng `padding='valid'` trong các lớp Convolution.
- **Kỹ thuật tối ưu:**
    - **Class Weights:** Sử dụng `class_weight='balanced'` để xử lý vấn đề mất cân bằng dữ liệu giữa các lớp.
    - **Regularization:** Kết hợp `Dropout` (0.2 - 0.3) và `GlobalAveragePooling1D` để chống overfitting.
    - **Trình tối ưu:** Adam optimizer (learning rate 1e-3) với hàm mất mát `sparse_categorical_crossentropy`.

---

# 🏷️ Phiên bản: `train_v12_kq_6ch_opt`

## 1. Tổng quan Model
- **Mục đích:** Phân loại hành vi (Activity Recognition) dựa trên dữ liệu cảm biến quán tính.
- **Quy mô:** Mô hình phân loại 4 lớp (n_classes = 4) đầu ra thông qua hàm kích hoạt Softmax.

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc:** Sử dụng bộ `LABEL_MAP` (được trích xuất từ tiền tố tên file, ví dụ: `label_code_subject_id`).
- **Gắn nhãn:** Dữ liệu đầu vào yêu cầu độ dài cố định là 200 điểm dữ liệu (time steps).
- **Cân bằng dữ liệu:** Sử dụng kỹ thuật `class_weight='balanced'` để tự động tính toán trọng số cho các lớp, đặc biệt ưu tiên xử lý dữ liệu mất cân bằng cho class "Fall" (ngã).

## 3. Tiền xử lý (Preprocessing)
- **Cắt gọt dữ liệu:** Chỉ giữ lại 6 trục cảm biến (3 trục Accelerometer + 3 trục Gyroscope).
- **Định dạng:** Dữ liệu nạp từ cache dưới dạng `.npy` với kích thước đầu vào `(samples, 200, 6)`.
- **Lưu ý:** Quy trình yêu cầu tiền xử lý đồng nhất cho tập Train, Val và Test trước khi đưa vào mô hình.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Temporal Convolutional Network (TCN) với 2 stacks, mỗi stack gồm 4 lớp Conv1D sử dụng dilation rate lần lượt là `[1, 2, 4, 8]`.
- **Kỹ thuật tối ưu:**
    - **Data Augmentation:** Thêm `GaussianNoise(0.05)` ở tầng đầu vào để tăng tính ổn định.
    - **Residual Connections:** Sử dụng kết nối tắt (shortcut) có điều chỉnh shape qua lớp `Cropping1D` và `Conv1D(1)` để duy trì luồng dữ liệu.
    - **Cải tiến Convolution:** Lớp Conv1D đầu tiên (stack 0, d=1) tăng lên 64 filters, các lớp còn lại sử dụng 32 filters.
    - **Regularization:** Sử dụng `BatchNormalization` sau mỗi lớp Conv và `Dropout(0.2)` trong block, `Dropout(0.3)` sau lớp `GlobalAveragePooling1D`.
    - **Loss Function:** `sparse_categorical_crossentropy` với Optimizer Adam (learning rate = 1e-3).

---

# 🏷️ Phiên bản: `train_v13_kq`

## 1. Tổng quan Model
- **Mục đích:** Phân loại hành động (Activity Recognition) dựa trên dữ liệu cảm biến IMU.
- **Quy mô:** Thiết kế cho bài toán phân loại đa lớp (default: 4 lớp đầu ra).

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc:** Sử dụng các cửa sổ trượt (windowing) cố định với độ dài 200 mẫu (samples) mỗi tệp.
- **Phân chia:** Dữ liệu được chia theo `subject_id` (Train/Val/Test) để đảm bảo tính độc lập giữa các đối tượng thực hiện thử nghiệm.
- **Gắn nhãn:** Sử dụng ánh xạ `LABEL_MAP` (được đọc trực tiếp từ tiền tố của tên tệp tin).

## 3. Tiền xử lý (Preprocessing)
- **Chuẩn hóa:** Dữ liệu đầu vào được chuẩn hóa qua lớp `BatchNormalization` ngay tại lớp đầu tiên của mô hình.
- **Định dạng dữ liệu:** Chuyển đổi nhãn sang định dạng One-hot Encoding (`to_categorical`) phục vụ cho quá trình huấn luyện với `CategoricalCrossentropy`.
- **Cân bằng dữ liệu:** Tính toán trọng số lớp (`class_weight='balanced'`) để xử lý vấn đề mất cân bằng dữ liệu trong tập huấn luyện.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Sử dụng kiến trúc **TCN (Temporal Convolutional Network)** với 2 stacks, mỗi stack bao gồm các lớp `Conv1D` có độ trễ (dilation rate) tăng dần [1, 2, 4, 8].
- **Cơ chế đặc biệt:** 
    - Áp dụng **Residual Connections** để duy trì luồng thông tin qua các lớp convolution.
    - Sử dụng **Cropping1D** để khớp kích thước trong các khối residual.
    - **GlobalMaxPooling1D:** Được lựa chọn thay thế cho Average Pooling để ưu tiên giữ lại các đỉnh cường độ tín hiệu (đặc biệt hữu ích cho các hành động như đi bộ).
- **Kỹ thuật tối ưu:**
    - **Loss Function:** Sử dụng `CategoricalCrossentropy` kết hợp với **Label Smoothing (0.1)** để giảm overfitting.
    - **Regularization:** Sử dụng `Dropout` (0.2 trong các khối TCN và 0.3 ở lớp dense cuối) để tăng khả năng tổng quát hóa.
    - **Optimizer:** Adam (learning rate = 1e-3).

---

# 🏷️ Phiên bản: `train_v14_kq`

Dưới đây là tài liệu hướng dẫn kỹ thuật cho phiên bản model dựa trên mã nguồn bạn đã cung cấp.


## 1. Tổng quan Model
- **Mục đích:** Phân loại hành động dựa trên dữ liệu chuỗi thời gian từ cảm biến IMU (thường là Accel/Gyro).
- **Quy mô:** Đầu ra gồm 4 lớp (n_classes = 4).

## 2. Chiến lược Data & Gắn nhãn
- **Cơ chế:** Dữ liệu được gán nhãn tự động thông qua tiền tố (prefix) của tên tệp tin (ví dụ: `[label_code]_[subject_id].csv`).
- **Phân chia tập dữ liệu:** Dữ liệu được phân chia theo `subject_id` (người thực hiện) vào các tập Train, Validation và Test để đảm bảo tính độc lập của dữ liệu.
- **Xử lý mất cân bằng:** Sử dụng kỹ thuật `class_weight='balanced'` trong quá trình huấn luyện để tính toán trọng số lớp, đảm bảo mô hình không bị thiên lệch bởi các class có số lượng mẫu ít.

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc dữ liệu:** Mỗi mẫu đầu vào là một cửa sổ thời gian (window) cố định với độ dài chính xác là 200 đơn vị (timestep).
- **Định dạng:** Dữ liệu được lưu trữ và tải thông qua tệp `.npy` để tăng tốc độ huấn luyện (cache system).
- **Lưu ý:** Mã nguồn hiện tại tập trung vào việc chuẩn hóa cấu trúc dữ liệu theo `subject_id`, đảm bảo tính nhất quán của dữ liệu đầu vào trước khi đưa vào mạng.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Temporal Convolutional Network (TCN) với 2 lớp stack (mỗi lớp gồm các khối giãn nở - dilation rates 1, 2, 4, 8).
- **Thành phần chính:**
    - `BatchNormalization`: Ổn định quá trình học.
    - `Conv1D`: Trích xuất đặc trưng không gian-thời gian.
    - `Residual Connection (Add)`: Kết nối tắt để tránh mất mát thông tin (kèm theo `Cropping1D` để đồng bộ chiều dữ liệu sau tích chập).
    - `GlobalAveragePooling1D`: Giảm chiều dữ liệu trước lớp phân loại.
    - `Dropout` (0.2 - 0.3): Chống quá khớp (overfitting).
- **Tối ưu hóa:**
    - **Loss:** `sparse_categorical_crossentropy`.
    - **Optimizer:** Adam (Learning rate = 1e-3).
    - **Cấu hình:** Đã loại bỏ *Label Smoothing* và quay về cấu trúc mặc định để tối ưu hóa hiệu suất trên MCU.

---

# 🏷️ Phiên bản: `train_v14_kq_6ch`

## 1. Tổng quan Model
- **Mục đích:** Nhận diện hoạt động dựa trên dữ liệu cảm biến IMU (Accel + Gyro).
- **Quy mô:** Thiết kế cho các thiết bị nhúng (MCU), phân loại đầu ra với `n_classes=4`.

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc dữ liệu:** Dữ liệu được chia theo cửa sổ (window) cố định với độ dài 200 điểm mẫu/file.
- **Phân chia tập dữ liệu:** Sử dụng kỹ thuật phân tách theo đối tượng thực hiện (`subject_id`) để đảm bảo tính độc lập giữa tập Train, Val và Test (tránh rò rỉ dữ liệu).
- **Gắn nhãn:** Dựa trên mã nhãn (`label_code`) được trích xuất trực tiếp từ tên tệp tin (sử dụng `LABEL_MAP`).

## 3. Tiền xử lý (Preprocessing)
- **Cắt gọt (Clipping):** Dữ liệu đầu vào được chọn lọc bằng cách giữ lại 6 trục cảm biến chính (`X[:, :, :6]`), tương ứng với 3 trục Gia tốc (Accel) và 3 trục Con quay (Gyro).
- **Định dạng đầu vào:** Dữ liệu được chuẩn hóa thông qua `BatchNormalization` ngay tại lớp đầu tiên của mô hình.
- **Cache:** Sử dụng cơ chế lưu trữ cache (`.npy`) để tối ưu hóa quá trình tải dữ liệu lặp lại.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Temporal Convolutional Network (TCN) với các đặc điểm:
    - **Cấu trúc:** 2 khối (stack), mỗi khối gồm 4 lớp `Conv1D` với các tốc độ giãn nở (dilation rates) lần lượt là [1, 2, 4, 8].
    - **Cơ chế:** Sử dụng kết nối tắt (Residual Connection) với `Add` layer sau mỗi khối `Conv1D` để tránh triệt tiêu gradient.
    - **Pooling:** Sử dụng `GlobalAveragePooling1D` để giảm chiều dữ liệu trước khi đưa vào lớp phân loại cuối.
- **Kỹ thuật tối ưu:**
    - **Regularization:** Sử dụng `Dropout` (0.2 trong block, 0.3 trước đầu ra) và `BatchNormalization` để ổn định huấn luyện.
    - **Cân bằng lớp:** Sử dụng `class_weight='balanced'` để xử lý vấn đề mất cân bằng dữ liệu trong quá trình huấn luyện.
    - **Optimizer:** `Adam` với learning rate 1e-3, hàm mất mát `sparse_categorical_crossentropy`.

---

# 🏷️ Phiên bản: `train_v14_kq_7ch`

## 1. Tổng quan Model
- **Mục đích**: Nhận diện hoạt động (Human Activity Recognition) dựa trên dữ liệu cảm biến IMU (Accel & Gyro).
- **Quy mô đầu ra**: Mô hình phân loại 4 lớp (n_classes=4).

## 2. Chiến lược Data & Gắn nhãn
- **Cấu trúc dữ liệu**: Mỗi sample là một cửa sổ (window) có độ dài cố định 200 đơn vị thời gian.
- **Phân chia tập dữ liệu**: Dữ liệu được chia theo `subject_id` (Train/Val/Test) để đảm bảo tính độc lập giữa các đối tượng người dùng.
- **Gắn nhãn**: Sử dụng `LABEL_MAP` để ánh xạ mã nhãn từ tên file (dạng `labelCode_subjectId.csv`).

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc trục**: Dữ liệu đầu vào ban đầu gồm Accel và Gyro, sau đó được bổ sung thêm kênh SVM (Signal Vector Magnitude) để tăng cường đặc trưng (tổng cộng 7 trục dữ liệu).
- **Định dạng**: Dữ liệu được kiểm tra độ dài nghiêm ngặt (chỉ chấp nhận 200 timestep).
- **Tối ưu hóa**: Dữ liệu được lưu trữ dạng Cache (`.npy`) để tăng tốc độ tải trong các lần huấn luyện tiếp theo.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng**: Kiến trúc **TCN (Temporal Convolutional Network)** tối ưu cho thiết bị nhúng (MCU).
- **Thành phần**:
    - Sử dụng các lớp `Conv1D` với cơ chế **Dilation** (tăng dần từ 1, 2, 4, 8) qua 2 stacks.
    - **Cấu trúc Residual**: Kết hợp các nhánh nối tắt (skip connections) với `Cropping1D` để đồng bộ kích thước tensor.
    - **Regularization**: Ứng dụng `BatchNormalization` và `Dropout` (tỉ lệ 0.2 trong các block và 0.3 trước lớp Output).
    - **Pooling**: `GlobalAveragePooling1D` được sử dụng để giảm số lượng tham số trước khi đưa qua lớp Dense cuối cùng.
- **Kỹ thuật tối ưu**:
    - **Loss Function**: `sparse_categorical_crossentropy`.
    - **Class Weighting**: Sử dụng chiến lược `class_weight='balanced'` để xử lý vấn đề mất cân bằng dữ liệu giữa các lớp.
    - **Optimizer**: Adam với learning rate 1e-3.

---

# 🏷️ Phiên bản: `train_v15_kq`

## 1. Tổng quan Model
- **Mục đích:** Phân loại hoạt động dựa trên dữ liệu cảm biến IMU (Cảm biến quán tính).
- **Quy mô:** Thiết kế cho hệ thống nhúng (MCU), mô hình đầu ra phân loại 4 lớp (classes).

## 2. Chiến lược Data & Gắn nhãn
- **Cơ chế:** Gán nhãn dựa trên tiền tố của tên tệp (filename).
- **Phân chia tập dữ liệu:** Dữ liệu được chia theo `subject_id` (Train/Val/Test) để đảm bảo tính độc lập giữa người thực hiện.
- **Độ dài cửa sổ:** Cố định 200 mẫu (samples) cho mỗi tệp đầu vào.
- **Cân bằng dữ liệu:** Sử dụng kỹ thuật `class_weight='balanced'` để xử lý sự mất cân bằng giữa các lớp trong quá trình huấn luyện.

## 3. Tiền xử lý (Preprocessing)
- **Định dạng dữ liệu:** Cắt gọt dữ liệu thành các window có độ dài 200.
- **Chuẩn hóa:** Dữ liệu được đưa vào mô hình dưới dạng `float32`. (Lưu ý: Logic scale cho Gyro/Accel được thực hiện trước đó trong quá trình windowing, dữ liệu đầu vào hiện tại đã qua xử lý sẵn).

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Temporal Convolutional Network (TCN) với các lớp `Conv1D` tích hợp cơ chế Dilated Convolutions (dilation rates: 1, 2, 4, 8) và kết nối tắt (Residual connections).
- **Thành phần chính:**
    - **Backbone:** 2 tầng TCN, mỗi tầng sử dụng `BatchNormalization` và `Dropout (0.2)` để chống overfitting.
    - **Pooling:** Sử dụng `GlobalAveragePooling1D` để trích xuất đặc trưng từ chuỗi thời gian.
    - **Head (v15):** Thêm lớp `Dense(64)` kèm `ReLU` và `BatchNormalization` để tăng cường khả năng phân biệt giữa các lớp có đặc điểm tương đồng (ví dụ: Walk vs Static).
- **Kỹ thuật tối ưu:**
    - **Optimizer:** Adam (learning rate = 1e-3).
    - **Loss:** `sparse_categorical_crossentropy`.
    - **Regularization:** `Dropout (0.3)` tại lớp Dense cuối, không sử dụng Label Smoothing trong phiên bản này.

---

# 🏷️ Phiên bản: `train_v16_kq`

## 1. Tổng quan Model
- **Mục đích**: Phân loại hoạt động người dùng (Human Activity Recognition - HAR) dựa trên dữ liệu cảm biến quán tính (IMU).
- **Quy mô**: Mô hình đầu ra gồm **6 lớp (classes)**.

## 2. Chiến lược Data & Gắn nhãn
Dữ liệu được gán nhãn dựa trên thông tin tên file (`filename_info`):
- **Fall**: Các file có tiền tố bắt đầu bằng 'F'.
- **Run**: Các file có tiền tố 'D03', 'D04'.
- **Walk**: Các file có tiền tố 'D01', 'D02', 'D05', 'D06'.
- **Trans**: Các file chứa nhãn '_Trans_'.
- **StandSit**: Các file chứa nhãn '_StandSit_'.
- **Lie**: Các file chứa nhãn '_Lie_'.
*Lưu ý: Dữ liệu được cắt cố định thành các đoạn (windows) có chiều dài 200 mẫu (samples).*

## 3. Tiền xử lý (Preprocessing)
- **Định dạng input**: Dữ liệu bao gồm 6 features (ax, ay, az, gx, gy, gz).
- **Cấu trúc**: Mỗi instance có kích thước (200, 6).
- **Chuẩn hóa**: Sử dụng `BatchNormalization` ngay tại lớp đầu vào để ổn định phân phối dữ liệu (gián tiếp xử lý các giá trị scale của Accel/Gyro).

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng**: **Temporal Convolutional Network (TCN)** kết hợp với khối residual.
- **Chi tiết cấu trúc**:
    - Sử dụng 2 stack, mỗi stack bao gồm các lớp `Conv1D` với dilation rate là [1, 2, 4, 8] để mở rộng trường nhìn (receptive field).
    - Áp dụng `BatchNormalization` và `Dropout (0.2)` sau mỗi lớp tích chập.
    - Sử dụng kết nối tắt (**Residual connections**) với `Cropping1D` để khớp kích thước tensor.
    - Lớp gộp: `GlobalAveragePooling1D` để giảm chiều dữ liệu trước khi đưa vào lớp phân loại.
- **Kỹ thuật tối ưu**:
    - **Class Weights**: Sử dụng chiến lược `balanced` để giải quyết vấn đề mất cân bằng dữ liệu giữa các lớp.
    - **Tối ưu hóa**: Sử dụng bộ tối ưu `Adam` (learning rate 1e-3) với hàm mất mát `sparse_categorical_crossentropy`.
    - **Regularization**: Sử dụng `Dropout (0.3)` trước lớp Dense cuối cùng để ngăn chặn overfitting.

---

# 🏷️ Phiên bản: `train_v17_kq`

## 1. Tổng quan Model
- **Mục đích**: Phân loại hoạt động con người (HAR - Human Activity Recognition) dựa trên dữ liệu cảm biến IMU (6 features).
- **Quy mô**: 6 lớp đầu ra (classes).

## 2. Chiến lược Data & Gắn nhãn
Dữ liệu được phân loại dựa trên hậu tố/tiền tố của tên tệp tin:
- **Trans**: Chuyển trạng thái (Transition).
- **StandSit**: Đứng hoặc ngồi.
- **Lie**: Nằm.
- **Walk**: Các tệp có tiền tố `D01, D02, D05, D06`.
- **Run**: Các tệp có tiền tố `D03, D04`.
- **Fall**: Các tệp có tiền tố `F`.
- **Độ dài cửa sổ**: Cố định 200 mẫu (samples) mỗi tệp tin.

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc dữ liệu**: Mỗi đầu vào có 6 features (được nạp trực tiếp từ file CSV sau khi kiểm tra độ dài 200 samples).
- **Phân chia dữ liệu**: Dữ liệu được chia theo `subject_id` (Train/Val/Test) để đảm bảo không bị rò rỉ dữ liệu giữa các tập.
- **Cache**: Có cơ chế lưu trữ cache dữ liệu (`.npy`) để tối ưu hóa quá trình tải dữ liệu lặp lại.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng**: Temporal Convolutional Network (TCN) sử dụng các lớp `Conv1D` với kỹ thuật **Dilation** (tăng dần từ 1, 2, 4, 8) và kết nối tắt (**Residual connections**) để giải quyết bài toán vanishing gradient.
- **Cấu trúc chi tiết**:
    - Sử dụng `BatchNormalization` và `Dropout` (0.2) sau các lớp Convolution.
    - Cấu trúc Residual gồm 2 stacks, mỗi stack có 4 lớp TCN.
    - `GlobalAveragePooling1D` ở tầng cuối trước lớp `Dense`.
- **Kỹ thuật tối ưu**:
    - **Class Weighting**: Sử dụng `class_weight='balanced'` để xử lý dữ liệu bị mất cân bằng giữa các lớp.
    - **Optimizer**: Adam với learning rate `1e-3`.
    - **Loss Function**: `sparse_categorical_crossentropy`.

---

# 🏷️ Phiên bản: `train_v18_kq`

## 1. Tổng quan Model
- **Mục đích:** Phân loại hành động của con người dựa trên dữ liệu IMU (cảm biến gia tốc và con quay hồi chuyển).
- **Quy mô:** Mô hình phân loại thành **5 lớp** hành động đầu ra (Softmax).

## 2. Chiến lược Data & Gắn nhãn
Dữ liệu được gán nhãn dựa trên thông tin tên file (filename parsing):
- **Fall**: Các file có tiền tố bắt đầu bằng 'F'.
- **Walk**: Tiền tố 'D01', 'D02', 'D05', 'D06'.
- **Run**: Tiền tố 'D03', 'D04'.
- **Idle**: Các trạng thái tĩnh ('StandSit', 'Lie').
- **Trans**: Các trạng thái chuyển tiếp ('Trans').

## 3. Tiền xử lý (Preprocessing)
- **Cửa sổ dữ liệu (Windowing):** Dữ liệu được chuẩn hóa thành các clip có độ dài cố định là **200 mẫu (samples)** cho mỗi file.
- **Tính năng đầu vào:** 6 features (3 trục gia tốc `ax, ay, az` và 3 trục con quay hồi chuyển `gx, gy, gz`).
- **Lưu trữ:** Sử dụng cơ chế Cache (file `.npy`) để tăng tốc độ tải dữ liệu cho các lần chạy sau.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Sử dụng kiến trúc **TCN (Temporal Convolutional Network)** với các khối Residual nối tiếp.
- **Cấu trúc chi tiết:**
    - Gồm 2 stack, mỗi stack chứa các lớp `Conv1D` với `dilation_rate` lần lượt là 1, 2, 4, 8 để mở rộng trường nhìn (receptive field).
    - Sử dụng `BatchNormalization` sau các lớp tích chập và `Dropout` (0.2 - 0.3) để chống overfitting.
    - Kết nối tắt (Residual connection) với `Cropping1D` để khớp kích thước tensor sau các lớp giãn nở.
    - Lớp cuối: `GlobalAveragePooling1D` kết hợp với `Dense` layer (Softmax).
- **Kỹ thuật tối ưu:**
    - **Class Weights:** Sử dụng trọng số cân bằng (`class_weight='balanced'`) để xử lý vấn đề mất cân bằng dữ liệu giữa các class.
    - **Optimizer:** Adam (learning rate = 1e-3).
    - **Loss Function:** `sparse_categorical_crossentropy`.

---

# 🏷️ Phiên bản: `train_v19_kq`

## 1. Tổng quan Model
- **Mục đích:** Phân loại hoạt động con người dựa trên dữ liệu cảm biến IMU (cảm biến quán tính).
- **Quy mô:** Mô hình đầu ra phân loại 5 lớp (5 classes).

## 2. Chiến lược Data & Gắn nhãn
Dữ liệu được gán nhãn dựa trên tiền tố của tên file CSV, được chia thành 5 nhóm lớp chính:
- **Trans:** Dành cho các tệp có chứa chuỗi `_Trans_`.
- **Idle:** Dành cho các trạng thái tĩnh (`_StandSit_` hoặc `_Lie_`).
- **Walk:** Dành cho các hoạt động di chuyển (các tiền tố file: `D01`, `D02`, `D05`, `D06`).
- **Run:** Dành cho các hoạt động chạy (các tiền tố file: `D03`, `D04`).
- **Fall:** Dành cho các sự kiện té ngã (tiền tố file bắt đầu bằng `F`).

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc dữ liệu:** Mỗi mẫu (window) dữ liệu yêu cầu cố định độ dài là 200 điểm thời gian (time steps) với 6 features đầu vào.
- **Phân bổ:** Dữ liệu được chia theo `subject_id` (Train/Val/Test) để đảm bảo tính độc lập của người dùng trong quá trình huấn luyện và kiểm thử.
- **Định dạng:** Sử dụng cache (định dạng `.npy`) để tăng tốc độ tải dữ liệu trong các lần chạy sau.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Temporal Convolutional Network (TCN) sử dụng các khối Residual Conv1D với kỹ thuật Dilation (tăng dần từ 1, 2, 4, 8) qua 2 stack.
- **Cấu phần:**
    - Sử dụng `BatchNormalization` để ổn định training.
    - `Dropout` (0.2 - 0.3) được thêm vào sau lớp tích chập và lớp Global Average Pooling để chống overfitting.
    - `GlobalAveragePooling1D` được sử dụng thay vì Flatten để giảm số lượng tham số trước lớp Dense cuối cùng.
- **Kỹ thuật tối ưu:**
    - **Class Weights:** Sử dụng `compute_class_weight='balanced'` để xử lý vấn đề mất cân bằng dữ liệu giữa các lớp.
    - **Optimizer:** Adam với learning rate = 1e-3.
    - **Loss:** `sparse_categorical_crossentropy` cho phân loại đa lớp.

---

# 🏷️ Phiên bản: `train_v20_kq`

## 1. Tổng quan Model
- **Mục đích:** Phân loại hành động dựa trên dữ liệu cảm biến IMU (6 features).
- **Quy mô:** Hỗ trợ 5 lớp đầu ra (5 classes) thông qua tầng Dense với hàm kích hoạt Softmax.

## 2. Chiến lược Data & Gắn nhãn
Dữ liệu được gán nhãn dựa trên tiền tố hoặc từ khóa trong tên file:
- **Fall:** Các file có tiền tố bắt đầu bằng 'F'.
- **Walk:** Tiền tố 'D01', 'D02', 'D05', 'D06'.
- **Run:** Tiền tố 'D03', 'D04'.
- **Idle:** Chứa từ khóa '_StandSit_' hoặc '_Lie_'.
- **Trans:** Chứa từ khóa '_Trans_'.

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc dữ liệu:** Mỗi mẫu (sample) là một chuỗi thời gian cố định với độ dài 200 bước (200 timesteps).
- **Số lượng đặc trưng:** Dữ liệu đầu vào bao gồm 6 features từ cảm biến IMU.
- **Phân chia dữ liệu:** Chia tập Train/Val/Test dựa trên định danh đối tượng (`subject_id`) để đảm bảo tính độc lập giữa các tập dữ liệu.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Sử dụng kiến trúc **TCN (Temporal Convolutional Network)** tối ưu cho MCU:
    - 2 chồng (stack) lớp `Conv1D` với các độ trễ (dilation rates) lần lượt là [1, 2, 4, 8].
    - Sử dụng các khối Residual (tương đương kiến trúc ResNet) để tránh triệt tiêu gradient.
    - `BatchNormalization` và `Dropout` (0.2) được áp dụng sau mỗi lớp Conv1D để ổn định huấn luyện.
- **Kết hợp đặc trưng:** Sử dụng đồng thời `GlobalAveragePooling1D` (GAP) và `GlobalMaxPooling1D` (GMP) để bắt được cả đặc trưng nền (Idle) và các xung đỉnh (Fall/Trans).
- **Kỹ thuật tối ưu:**
    - Sử dụng **Class Weights** (chiến lược 'balanced') để xử lý sự mất cân bằng giữa các lớp.
    - Optimizer: **Adam** (learning rate = 1e-3).
    - Loss Function: **Sparse Categorical Crossentropy**.

---

# 🏷️ Phiên bản: `train_v21_kq`

## 1. Tổng quan Model
- **Mục đích:** Phân loại hành vi người dùng từ dữ liệu cảm biến IMU (6-axis).
- **Quy mô:** Hệ thống phân loại 5 lớp (5-class classification).

## 2. Chiến lược Data & Gắn nhãn
Dữ liệu được gán nhãn dựa trên tiền tố của tên file CSV, được chia thành 5 nhóm hành vi cụ thể:
- **Fall:** Các file có tiền tố bắt đầu bằng 'F'.
- **Walk:** Các file có tiền tố 'D01', 'D02', 'D05', 'D06'.
- **Run:** Các file có tiền tố 'D03', 'D04'.
- **Idle:** Bao gồm các trạng thái tĩnh ('_StandSit_', '_Lie_').
- **Trans:** Các file chuyển động trung gian ('_Trans_').
*Lưu ý: Dữ liệu được giới hạn cố định ở độ dài cửa sổ (window size) là 200 mẫu.*

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc dữ liệu:** Sử dụng 6 features đầu vào (dữ liệu thô từ cảm biến).
- **Phân chia tập dữ liệu:** Sử dụng chiến lược tách theo chủ thể (Subject-based split) thành 3 tập: Train, Validation, Test.
- **Cân bằng lớp:** Sử dụng kỹ thuật `class_weight='balanced'` để xử lý vấn đề mất cân bằng dữ liệu trong quá trình huấn luyện.
- **Cache:** Hệ thống tự động lưu/tải dữ liệu qua file `.npy` để tăng tốc độ khởi tạo.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Temporal Convolutional Network (TCN) được tùy biến cho thiết bị nhúng (MCU).
- **Cấu trúc lớp:**
    - Sử dụng 2 stack, mỗi stack bao gồm 4 lớp `Conv1D` với các dilation rate tương ứng [1, 2, 4, 8].
    - `Kernel size = 5` nhằm mở rộng trường tiếp nhận (receptive field).
    - Có sử dụng **Residual Connection** với kỹ thuật `Cropping1D` để khớp chiều dữ liệu sau khi convolution.
    - `BatchNormalization` và `Dropout` (0.2 - 0.4) được áp dụng tại các lớp để chống overfitting.
- **Tối ưu đầu ra:** 
    - Kết hợp cả `GlobalAveragePooling1D` và `GlobalMaxPooling1D` (Concatenate) để trích xuất đặc trưng trung bình và đặc trưng mạnh nhất (chống spikes).
- **Compile:** Sử dụng hàm loss `sparse_categorical_crossentropy` với bộ tối ưu `Adam` (learning rate 1e-3).

---

# 🏷️ Phiên bản: `train_v22_kq`

## 1. Tổng quan Model
- **Mục đích:** Phân loại hoạt động của con người dựa trên dữ liệu cảm biến IMU (6 trục).
- **Quy mô:** Mô hình đầu ra gồm **5 lớp (classes)**.

## 2. Chiến lược Data & Gắn nhãn
Dữ liệu được quy đổi về 5 nhãn chính dựa trên tên file:
- **Fall:** Các file bắt đầu bằng tiền tố 'F'.
- **Walk:** Các file bắt đầu bằng 'D01', 'D02', 'D05', 'D06'.
- **Run:** Các file bắt đầu bằng 'D03', 'D04'.
- **Idle:** Các file chứa '_StandSit_' hoặc '_Lie_'.
- **Trans:** Các file chứa '_Trans_'.
- *Lưu ý:* Chỉ xử lý các cửa sổ dữ liệu có độ dài cố định là 200 mẫu.

## 3. Tiền xử lý (Preprocessing)
- **Cấu trúc:** Sử dụng dữ liệu đầu vào dạng windowed (200, 6) gồm 6 features từ cảm biến.
- **Cân bằng dữ liệu:** Tính toán `class_weights` dựa trên phân phối thực tế của tập huấn luyện (`class_weight='balanced'`) để xử lý mất cân bằng dữ liệu.
- **Định dạng:** Nhãn được chuyển sang dạng **One-hot encoding** phục vụ cho quá trình huấn luyện với hàm lỗi Categorical Crossentropy.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** 
    - Kiến trúc **TCN (Temporal Convolutional Network)** với các lớp `Conv1D` sử dụng `dilation_rate` (1, 2, 4, 8) nhằm tăng trường nhìn (receptive field) của mô hình.
    - Tích hợp **SE Block (Squeeze-and-Excitation)** để tăng cường khả năng chú ý (attention) trên các kênh đặc trưng.
    - Kết hợp `GlobalAveragePooling1D` và `GlobalMaxPooling1D` để tận dụng cả thông tin đặc trưng trung bình và cực đại.
- **Kỹ thuật tối ưu:**
    - **Regularization:** Sử dụng `BatchNormalization` sau mỗi lớp Conv và `Dropout` (0.2 - 0.4) để tránh overfitting.
    - **Loss function:** Sử dụng `CategoricalCrossentropy` kết hợp với kỹ thuật **Label Smoothing (0.1)** để cải thiện khả năng tổng quát hóa.
    - **Cấu trúc nhánh:** Sử dụng cơ chế `Residual Connection` (Add) để hỗ trợ hội tụ cho các mạng sâu.

---

# 🏷️ Phiên bản: `train_v23_kq`

## 1. Tổng quan Model
- **Mục đích**: Nhận diện hoạt động người (Human Activity Recognition - HAR) dựa trên dữ liệu cảm biến IMU (6 trục).
- **Quy mô đầu ra**: Phân loại 5 lớp hoạt động (Classes).

## 2. Chiến lược Data & Gắn nhãn
Dữ liệu được gắn nhãn dựa trên tiền tố của tên tệp (filename) và cấu trúc thư mục, cụ thể:
- **Fall**: Các tệp có tiền tố bắt đầu bằng 'F'.
- **Walk**: Các tệp có tiền tố 'D01', 'D02', 'D05', 'D06'.
- **Run**: Các tệp có tiền tố 'D03', 'D04'.
- **Idle**: Các tệp chứa '_StandSit_' hoặc '_Lie_'.
- **Trans**: Các tệp chứa '_Trans_'.
- **Yêu cầu dữ liệu**: Window size cố định là 200 mẫu (samples) trên mỗi tệp CSV.

## 3. Tiền xử lý (Preprocessing)
- **Input**: Dữ liệu chuỗi thời gian 6 features.
- **Cấu trúc**: Phân chia tập Train/Val/Test dựa trên định danh chủ thể (`subject_id`).
- **Cân bằng dữ liệu**: Sử dụng kỹ thuật `compute_class_weight` với tham số `'balanced'` để xử lý sự mất cân bằng giữa các lớp.
- **Định dạng**: Chuyển đổi nhãn sang dạng One-Hot Encoding (`to_categorical`) cho quá trình huấn luyện.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng**: Kiến trúc dựa trên **Residual Conv1D** kết hợp với **Squeeze-and-Excitation (SE) Block**.
- **Đặc điểm nổi bật**:
    - **TCN-like**: Sử dụng 4 khối Residual Conv1D (kernel size 7), thay thế Dilation bằng Strides để giảm chiều không gian.
    - **SE Block**: Áp dụng attention cơ chế Squeeze-and-Excitation (tỉ lệ giảm 4) để tối ưu hóa trọng số kênh (phù hợp cho thiết bị MCU).
    - **Pooling**: Kết hợp `GlobalAveragePooling1D` và `GlobalMaxPooling1D` (Concatenate) ở lớp cuối để trích xuất đặc trưng đa dạng.
- **Kỹ thuật tối ưu**:
    - **Label Smoothing**: Áp dụng `label_smoothing=0.1` trong hàm mất mát `CategoricalCrossentropy` để chống overfitting.
    - **Regularization**: Sử dụng `BatchNormalization` và `Dropout` (0.2 - 0.4) giữa các lớp.
    - **Optimizer**: Adam với `learning_rate=1e-3`.

---

# 🏷️ Phiên bản: `train_v24_kq`

## 1. Tổng quan Model
- **Mục đích:** Nhận diện hoạt động con người (HAR) dựa trên dữ liệu cảm biến IMU (cảm biến gia tốc và con quay hồi chuyển).
- **Quy mô:** Phân loại 5 lớp hoạt động (Classes).

## 2. Chiến lược Data & Gắn nhãn
Dữ liệu được gắn nhãn tự động dựa trên tên tệp tin:
- **Fall:** Các tệp bắt đầu bằng ký tự 'F'.
- **Walk:** Các tệp bắt đầu bằng mã 'D01', 'D02', 'D05', 'D06'.
- **Run:** Các tệp bắt đầu bằng mã 'D03', 'D04'.
- **Idle:** Các tệp có chứa '_StandSit_' hoặc '_Lie_'.
- **Trans:** Các tệp có chứa '_Trans_'.
- **Định dạng dữ liệu:** Cửa sổ trượt (window) có độ dài cố định 200 mẫu (samples) cho mỗi tệp.

## 3. Tiền xử lý (Preprocessing)
- **Cắt gọt (Clipping):** Dữ liệu cảm biến gia tốc (3 trục đầu) được giới hạn trong khoảng [-8.0, 8.0]g để loại bỏ nhiễu vượt ngưỡng vật lý.
- **Scaling:**
    - **Accelerometer (0:3):** Chia cho 8.0.
    - **Gyroscope (3:6):** Chia cho 2000.0.

## 4. Kiến trúc mô hình (Architecture)
- **Loại mạng:** Residual Conv1D (phong cách TCN) kết hợp Squeeze-and-Excitation (SE) Block.
- **Chi tiết cấu trúc:**
    - Sử dụng 4 khối Residual với Conv1D (kernel size 7), xen kẽ với các lớp BatchNormalization và Dropout (0.2).
    - Sử dụng **SE Block** (ratio=4) để tối ưu hóa trọng số kênh (channel attention).
    - Giảm chiều dữ liệu bằng `strides=2` thay vì Dilation.
    - Đầu ra kết hợp `GlobalAveragePooling1D` và `GlobalMaxPooling1D` (Concatenate) để tối ưu đặc trưng.
- **Kỹ thuật tối ưu:**
    - **Label Smoothing:** 0.1 nhằm tăng khả năng tổng quát hóa.
    - **Class Weights:** Sử dụng trọng số cân bằng (`class_weight='balanced'`) để xử lý bài toán dữ liệu mất cân bằng.
    - **Activation:** `relu6` (phù hợp cho các thiết bị nhúng).
    - **Regularization:** L2 regularization (1e-4) trên các lớp Conv/Dense.

---

