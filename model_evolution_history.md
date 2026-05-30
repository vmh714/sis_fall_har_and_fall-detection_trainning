# Báo Cáo Lịch Sử Tiến Hóa Kiến Trúc TCN Dành Cho ESP32-S3 (v16 - v22)

Tài liệu này ghi chú lại quá trình tiến hóa của kiến trúc mạng Temporal Convolutional Network (TCN) dành cho hệ thống Nhận diện hành động (HAR) và Phát hiện té ngã (Fall Detection) chạy trên vi điều khiển ESP32-S3.

Quá trình này bắt đầu từ khi chuyển sang sử dụng **Dataset mới (SisFall_dataset_Windowed - 6 kênh thuần, 200 mẫu/window)**, kế thừa nền tảng từ v12/v14 và tiến hóa dần lên phiên bản hoàn thiện v22.

---

## 1. Giai đoạn Chuyển giao & Tương thích (v16 - v17)
* **Kế thừa từ v12 & v14:** 
  * Các phiên bản v12, v14 mang lại kết quả khá tốt trên máy tính nhưng gặp vấn đề "chết người" khi deploy lên vi điều khiển: chứa các lớp `Lambda` phức tạp và các phép toán không được TFLite Micro hỗ trợ nguyên bản.
  * Ngoài ra, v14 sử dụng thêm đặc trưng Jerk (đạo hàm gia tốc) làm phình to số kênh đầu vào (lên 9 kênh), gây tốn kém tài nguyên tính toán (RAM/FLOPs).
* **Cải tiến ở v16 & v17:**
  * **Dataset mới:** Chuyển sang dùng dataset mới cắt cửa sổ chuẩn 200 mẫu (2 giây ở tần số 100Hz).
  * **Tối giản hóa (MCU-Friendly):** Loại bỏ hoàn toàn các lớp `Lambda` và mixed precision (`float16`). Ép mô hình sử dụng chuẩn `float32` thuần túy.
  * **Cố định Input:** Giữ nguyên 6 kênh thô ($Acc_{x,y,z}, Gyro_{x,y,z}$), không tính toán thêm Jerk để tiết kiệm chu kỳ máy (CPU cycles) trên ESP32-S3.

## 2. Giai đoạn Gộp Nhãn & Xử lý nhầm lẫn (v18 - v19)
* **Vấn đề:** Ban đầu mô hình nhận diện 6 nhãn (Walk, Run, StandSit, Lie, Trans, Fall). Tuy nhiên, `StandSit` và `Lie` có đặc tính tĩnh (gia tốc xoay quanh trục G, ít dao động) rất giống nhau, khiến mô hình lãng phí tài nguyên để phân biệt hai trạng thái không mang nhiều ý nghĩa sống còn.
* **Cải tiến:**
  * Gộp `StandSit` và `Lie` thành một siêu nhãn **`Idle`** (Trạng thái tĩnh). Bài toán rút gọn còn 5 nhãn: `Walk, Run, Idle, Trans, Fall`.
  * Sự tập trung lúc này dồn vào việc phân định giữa tĩnh (`Idle`) và chuyển trạng thái (`Trans`).

## 3. Giai đoạn Định hình Đặc trưng (v20)
* **Vấn đề:** Tín hiệu chuyển trạng thái (`Trans`) chứa cả yếu tố tĩnh (trước/sau khi chuyển) và yếu tố động (khoảnh khắc đứng lên/ngồi xuống). Nếu chỉ dùng GAP (Global Average Pooling), các gai tín hiệu đột ngột bị làm mờ.
* **Cải tiến:**
  * Kết hợp **GAP (Global Average Pooling)** và **GMP (Global Max Pooling)** bằng lớp `Concatenate()`.
  * GAP giúp hiểu bản chất trung bình của tín hiệu. GMP giúp bắt các "đỉnh" (peaks) đột biến.
* **Kết quả:** Độ chính xác tổng thể tăng lên, nhưng lượng nhầm lẫn chéo giữa `Idle` và `Trans` vẫn còn quá cao (gần 300 ca nhầm lẫn qua lại). F1-Score của `Trans` chỉ lẹt đẹt ở mức ~79%.

## 4. Giai đoạn Đột phá Trường Nhìn (Receptive Field) (v21)
* **Vấn đề cốt lõi:** Lớp tích chập giãn nở (Dilated CNN) đang dùng `kernel_size = 5` với 2 block (dilation = 1, 2, 4, 8). Tính toán theo công thức toán học, trường nhìn tổng (Receptive Field - RF) chỉ đạt **121 mẫu** (~1.2 giây). Trong khi đó, một hành động `Trans` thường diễn ra chậm và kéo dài tới 1.5 - 2 giây. Mô hình bị "cận thị".
* **Cải tiến:**
  * **Tăng `kernel_size` từ 5 lên 7.**
  * Phép toán thần kỳ này giúp mở rộng RF từ 121 mẫu lên **181 mẫu**, bao phủ 90.5% khung thời gian đầu vào.
  * Thay đổi `padding='causal'/'same'` thành `padding='valid'` kết hợp `Cropping1D` để bảo toàn kích thước tensor mà không sinh ra lỗi biên.
  * Tinh chỉnh lại tỷ lệ **Dropout** (0.2 cho Conv, 0.4 cho FC) để chống Overfitting khi số tham số tăng thêm ~16KB.
* **Kết quả:** F1-score của `Trans` tăng vọt lên ~81.9%. Các nhãn khác bắt đầu tiệm cận rất sát mốc 90%.

## 5. Trạng thái Hoàn mỹ (The Sweet Spot) (v22)
* **Mục tiêu:** Kéo tất cả 5 nhãn vượt ngưỡng 90% (F1-Score), đồng thời duy trì cấu trúc cực nhẹ cho MCU.
* **Cải tiến:**
  * **SE Block (Squeeze-and-Excitation):** Tích hợp thêm cơ chế Attention siêu nhẹ (chỉ dùng GAP và Dense) ở cuối mỗi Block Residual. SE Block giúp mô hình biết "Kênh nào đang quan trọng nhất" tại từng thời điểm. Overhead trên MCU gần như bằng 0.
  * **Label Smoothing (0.1):** Làm mềm nhãn mục tiêu (vd: $1.0 \rightarrow 0.9$) giúp hàm Loss không ép mô hình tự tin thái quá, chống học vẹt (Overfitting) cực kỳ hiệu quả đối với các ranh giới mờ giữa `Idle` và `Trans`.
  * **Tối ưu Class Weights:** 
    * Phạt cực nặng `Fall` ($\times 3.0$) để bảo vệ an toàn sinh mạng (Đạt Recall ngã > 97%).
    * Hủy bỏ phạt `Trans` (chuyển về chuẩn $\times 1.0$) để ngăn mô hình "đoán bừa". Precision của `Trans` lập tức bay từ 78% lên 83.6%, trả lại số ca đúng cho `Idle`.
* **Kết quả Cuối cùng (Test Set):**
  * **Walk:** 92.2%
  * **Run:** 96.9%
  * **Idle:** 90.8% 🎉
  * **Fall:** 97.8% (Recall 97.3%) 🛡️
  * **Trans:** 85.5% (Tối ưu nhất có thể cho hành động fuzzing)
  * **Total Accuracy:** 91.56%

---
**TỔNG KẾT:** Kiến trúc v22 là điểm giao thoa hoàn hảo giữa **Hiệu năng Toán học** và **Giới hạn Phần cứng (TinyML)**. Sẵn sàng 100% để biên dịch qua C++ (TFLite Micro) và nạp lên ESP32-S3.
