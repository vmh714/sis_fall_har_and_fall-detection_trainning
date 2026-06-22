# Hướng Dẫn Sử Dụng Dataset SisFall (Windowed & Preprocessed)

Tài liệu này mô tả chi tiết về bộ dữ liệu SisFall sau khi đã được tiền xử lý, downsample và áp dụng kỹ thuật Sliding Window (trích xuất cửa sổ thời gian) để sẵn sàng cho việc huấn luyện các mô hình Machine Learning / Deep Learning có thể triển khai lên vi điều khiển (MCU).

## 1. Thông số kỹ thuật (Specifications)
- **Tần số lấy mẫu (Sampling Rate):** 100 Hz (đã downsample từ gốc 200 Hz để giảm tải cho MCU).
- **Kích thước Window (Window Size):** 200 mẫu (tương đương 2.0 giây tín hiệu).
- **Độ dời Window (Step Size):**
  - **100 mẫu (1.0 giây - Overlap 50%)**: Dành cho các hành động liên tục (Đi bộ, Chạy) và Té ngã.
  - **50 mẫu (0.5 giây - Overlap 75%)**: Dành cho các hành vi chuyển trạng thái (Ngồi xuống, Đứng lên, Nằm xuống) nhằm lấy được nhiều frame dữ liệu hơn tại các khe thời gian hẹp.
- **Features (6 Trục):** `ax, ay, az, gx, gy, gz`. (Đã loại bỏ 3 trục từ cảm biến la bàn MMA).
- **Định dạng dữ liệu:** Các giá trị gia tốc và góc quay đã được convert sang các đơn vị vật lý chuẩn (g, rad/s hoặc deg/s tùy thuật toán preprocess trước đó).

## 2. Hệ thống Nhãn (6 Classes)
Bộ dữ liệu được chia làm 6 nhãn (Classes) chuyên biệt, cực kỳ phù hợp cho bài toán theo dõi người cao tuổi:

1. **Walk (Đi bộ):** Các file bắt đầu bằng `D01, D02, D05, D06`. Bao gồm đi bộ nhanh, chậm, lên/xuống cầu thang.
2. **Run (Chạy):** Các file bắt đầu bằng `D03, D04`.
3. **Fall (Té ngã):** Các file bắt đầu bằng `F01` đến `F15`. (Té ngã đơn điệu như vấp ngã, trượt chân).
4. **Transition (Chuyển trạng thái):** Các file có hậu tố `_Trans_`. Bao gồm khoảnh khắc đang đứng chuyển sang ngồi, đang ngồi chuyển sang nằm, và ngược lại.
5. **Idle_StandSit (Đứng/Ngồi yên):** Các file có hậu tố `_StandSit_`. Trạng thái tĩnh khi phương của trọng lực (1g) chủ yếu nằm trên trục Y (dọc theo cơ thể).
6. **Idle_Lie (Nằm yên):** Các file có hậu tố `_Lie_`. Trạng thái tĩnh khi phương của trọng lực (1g) phân tán sang trục X hoặc Z.

## 3. Quy ước Đặt tên File (Naming Convention)
Mỗi file `.csv` trong thư mục `SisFall_dataset_Windowed` đại diện cho một window (200x6). Tên file chứa đầy đủ thông tin để parse label:

- **Các hành động liên tục & Ngã:** `[Mã Hành Động]_[Mã Người Tập]_[Lần Thử]_W[Số Thứ Tự].csv`
  - *Ví dụ:* `D01_SA01_R01_W000.csv` -> Đi bộ, Người SA01, Lần thử 1, Window số 0.
  - *Ví dụ:* `F01_SA02_R05_W002.csv` -> Ngã, Người SA02, Lần thử 5, Window số 2.

- **Các kịch bản phức tạp (D07 - D16):** `[Mã Hành Động]_[Mã Người Tập]_[Lần Thử]_[Trạng Thái]_W[Số Thứ Tự].csv`
  - *Ví dụ:* `D12_SA01_R01_StandSit_W000.csv` -> Ngồi yên (trước khi nằm).
  - *Ví dụ:* `D12_SA01_R01_Trans_W005.csv` -> Khoảnh khắc đang chuyển từ Ngồi sang Nằm.
  - *Ví dụ:* `D12_SA01_R01_Lie_W008.csv` -> Trạng thái đã nằm yên trên giường.

## 4. Giải thuật Trích xuất Window tĩnh/động (Windowing Logic)
Đối với các kịch bản chuyển trạng thái phức tạp (VD: D12 - Ngồi -> Nằm -> Ngồi dậy), dữ liệu được bóc tách tự động cực kỳ tinh vi:

1. **Tìm đỉnh gia tốc (Peak Detection):** Tính toán Vector Magnitude (SVM) của 3 trục gia tốc và tìm 2 đỉnh có sự biến thiên mạnh nhất (chính là lúc cơ thể thay đổi tư thế).
2. **Cắt Trans Window:** Tại mỗi đỉnh, trích xuất 3 windows xen phủ nhau (shifts = -50, 0, 50) và gán nhãn là `Transition`.
3. **Cắt Static Window:** Chạy một sliding window (step=50) qua toàn bộ file. Các window có tâm cách xa đỉnh gia tốc một khoảng an toàn (>= 80 mẫu) được coi là trạng thái tĩnh.
4. **Phân loại StandSit / Lie:** Tính giá trị tuyệt đối trung bình của X, Y, Z trong đoạn tĩnh. Nếu Y lớn nhất -> Người đang đứng/ngồi (`StandSit`). Nếu X hoặc Z lớn nhất -> Người đang nằm (`Lie`).

## 5. Ứng dụng & Hướng phát triển
- Tập dữ liệu này đã được tối ưu hóa để loại bỏ tối đa nhiễu và những nhãn gây nhầm lẫn.
- Hoàn toàn phù hợp để nạp vào các mô hình học sâu (CNN 1D) nhằm triển khai (deploy) lên các dòng vi điều khiển (MCU) có bộ nhớ thấp nhờ số lượng features ít (6 trục) và dữ liệu được cắt sẵn gọn gàng (200 mẫu/khung).
