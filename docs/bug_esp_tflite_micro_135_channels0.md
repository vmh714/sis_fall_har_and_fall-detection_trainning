# Bug: ESP-NN không được tận dụng do esp-tflite-micro 1.3.5 (`channels = 0`)

> Ghi nhận: 2026-06-25 · Phạm vi: firmware ESP32-S3 (TFLM + ESP-NN), project `sis_fall_firmware_inference`
> Mức độ: hiệu năng (không crash, không sai kết quả) — **làm inference chậm ~3–5×** mà mọi dấu hiệu cấu hình đều "trông như đúng".

## 1. TL;DR

`esp-tflite-micro` **1.3.5** truyền `filter_dims.channels = 0` cho ESP-NN ở lớp wrapper `Conv2D`/`DepthwiseConv2D`. Vì thiếu số kênh, ESP-NN **không dispatch được vào kernel SIMD tối ưu theo kênh** mà rớt về vòng generic → chậm ~3–5× cycle/MAC, **dù** ESP-NN đã được biên dịch và link đầy đủ. Bản **1.3.7** sửa thành `channels = filter->dims->data[3]` → đúng tốc độ.

Khắc phục: nâng `esp-tflite-micro >= 1.3.7`.

## 2. Triệu chứng

Cùng một model CNN (kiến trúc y hệt, chỉ khác trọng số) đo trên 2 build:

| Build | esp-tflite-micro | Cấu hình | Thời gian/infer |
|---|---|---|---|
| Test tool | **1.3.5** | 240 MHz, `-O2`, arena SRAM | **56.7 ms** |
| Firmware thật | **1.3.7** | 160 MHz, `-Og` (Debug), arena PSRAM | **17.1 ms** |

Nghịch lý: build test có **cấu hình tốt hơn về mọi mặt** (CPU nhanh hơn 1.5×, `-O2` thay vì `-Og`, SRAM nhanh hơn PSRAM) nhưng lại **chậm hơn 3.3×**. Điều này là bất khả nếu cùng kernel ⇒ phải có khác biệt ở tầng thực thi.

## 3. Loại trừ các giả thuyết sai (quy trình debug)

1. **Khác model?** Không. Parse `.tflite` cả hai: **28 op giống hệt**, shape/stride/filter giống hệt, **dilation = (1,1) toàn bộ** (không có dilated conv), arena ~28 KB như nhau. Chỉ khác trọng số.
2. **Khác sdkconfig?** Không liên quan. `diff` đầy đủ 2 sdkconfig chỉ khác CPU freq / compiler opt / flash / partition / modem — **đều thiên về test tool phải nhanh hơn**, không có thiết lập nào làm chậm compute.
3. **ESP-NN không được biên dịch/không link?** Không. Cả hai build: `CONFIG_NN_OPTIMIZED=y`, esp-nn **1.2.3** với **assembly ESP32-S3 được compile** (kiểm object `*esp32s3*.obj`: 33 ở test, 24 ở real), và `-DESP_NN` được định nghĩa trong CMake của esp-tflite-micro (196 lần). ⇒ ESP-NN "ACTIVE" về mặt biên dịch ở cả hai.

Sau khi loại hết, khác biệt **duy nhất** còn lại: **version esp-tflite-micro (1.3.5 vs 1.3.7)**.

## 4. Nguyên nhân gốc

`diff` lớp kernel wrapper ESP-NN giữa 2 version:

`tensorflow/lite/micro/kernels/esp_nn/conv.cc` (2 chỗ) và `depthwise_conv.cc` (2 chỗ):

```c
// esp-tflite-micro 1.3.5 (CHẬM):
data_dims_t filter_dims = {
    .width = filter_width, .height = filter_height,
    .channels = 0, .extra = 0          // <-- BUG: số kênh = 0
};

// esp-tflite-micro 1.3.7 (ĐÚNG):
data_dims_t filter_dims = {
    .width = filter_width, .height = filter_height,
    .channels = filter->dims->data[3], .extra = 0   // <-- số kênh thật
};
```

`esp_nn_conv_s8()` / `esp_nn_depthwise_conv_s8()` dùng `filter_dims.channels` để chọn nhánh kernel SIMD tối ưu theo kênh (vd nhánh pointwise `mult8_1x1`, các nhánh align kênh bội 8/16). Khi `channels = 0`, các điều kiện dispatch không thỏa ⇒ ESP-NN **chạy nhánh generic** thay vì SIMD chuyên dụng.

## 5. Bằng chứng định lượng

Model ≈ **1,0 triệu MAC** (chủ yếu là pointwise 1×1 — vốn được ESP-NN tăng tốc mạnh nhất).

| Build | Thời gian | CPU | Cycle/MAC quy đổi |
|---|---|---|---|
| 1.3.7 | 17 ms | 160 MHz | **~2.7** (đúng ESP-NN S3) |
| 1.3.5 | 56.7 ms | 240 MHz | **~13.4** (generic, mất tăng tốc) |

Chênh ~5× cycle/MAC — đúng đặc trưng "ESP-NN không vào được fast-path".

## 6. Vì sao khó phát hiện

Mọi tín hiệu cấu hình đều **báo xanh** dù thực tế đang chậm:
- `CONFIG_NN_OPTIMIZED=y` ✔ (esp-nn *có* biên dịch bản optimized)
- File assembly `*esp32s3*.S/.c` được compile ✔
- `-DESP_NN` được định nghĩa ✔
- Không crash, kết quả phân loại vẫn đúng ✔

Bug nằm ở **tầng wrapper truyền sai metadata**, không phải ở chỗ "có hay không có ESP-NN". ⇒ Một dòng log kiểu "ESP-NN: ACTIVE" suy từ `CONFIG_NN_OPTIMIZED` **không đủ** để phân biệt build nhanh/chậm. Chỉ **độ trễ đo thực** hoặc **version wrapper** mới lộ ra.

## 7. Cách khắc phục

**Chuẩn (khuyến nghị):** nâng version trong `main/idf_component.yml`:
```yaml
dependencies:
  espressif/esp-tflite-micro: "^1.3.7"
```
rồi **xóa `dependencies.lock`** và `idf.py reconfigure` để re-resolve (xem §8).

**Hotfix tại chỗ (khi không đổi version được):** sửa 4 dòng trong `managed_components/.../kernels/esp_nn/conv.cc` (2) và `depthwise_conv.cc` (2): `.channels = 0` → `.channels = filter->dims->data[3]`. Lưu ý: thư mục `managed_components/` có thể bị ghi đè khi component manager re-resolve.

## 8. Bài học: `dependencies.lock` mới là thứ quyết định, không phải `.yml`

Điểm gây bối rối nhất: **cả test tool lẫn firmware thật đều ghi `"^1.3.5"`** trong `idf_component.yml`, nhưng resolve ra 2 version khác nhau:

| | Constraint `.yml` | Version THỰC compile (`dependencies.lock`) |
|---|---|---|
| Firmware thật | `^1.3.5` | **1.3.7** (resolve muộn, lúc registry đã có 1.3.7) |
| Test tool | `^1.3.5` | **1.3.5** (resolve sớm, lock đóng băng) |

`^1.3.5` nghĩa là `>=1.3.5, <2.0.0` — chỉ là **sàn tối thiểu**. `dependencies.lock` đóng băng version đã resolve và **không tự nâng** kể cả khi có bản mới hơn thỏa constraint. ⇒ Muốn nâng phải **xóa lock** rồi reconfigure; sửa mỗi dòng `^...` trong `.yml` là **chưa đủ**.

## 9. Cách tự kiểm tra về sau (đã tích hợp vào tool test)

Tool `firmware_test_tool/test_inference_uart.py` đọc version esp-tflite-micro thực biên dịch từ `managed_components/.../idf_component.yml` và in vào report:

```
CẤU HÌNH BUILD (điều kiện đo hiệu năng):
  - ESP-NN            : ACTIVE (optimized assembly kernels)
  - esp-tflite-micro  : 1.3.7
  - ESP-NN ACCELERATION : ON (esp-nn optimized + wrapper >=1.3.7)
```
Verdict `ESP-NN ACCELERATION` xét **2 trục**: (1) esp-nn build optimized/ANSI-C; (2) wrapper `>=1.3.7`. Chỉ "ON" khi cả hai đạt; `<=1.3.6` → cảnh báo `OFF/CRIPPLED (bug channels=0)`.

## 10. Liên quan

- Phân tích kiến trúc nào tận dụng ESP-NN: [`firmware_deployment_espnn_analysis.md`](firmware_deployment_espnn_analysis.md)
- Danh sách op được ESP-NN tăng tốc & ràng buộc thiết kế: `AGENTS.md` §4.2
