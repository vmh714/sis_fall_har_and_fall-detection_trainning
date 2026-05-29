#ifndef TFLITE_WRAPPER_H
#define TFLITE_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

// Khởi tạo mô hình TFLite Micro và cấp phát bộ nhớ Arena
// Trả về 0 nếu thành công, -1 nếu thất bại
int tflite_init(void);

// Thực thi mô hình với dữ liệu giả định và đo thời gian
void tflite_run_inference(void);

#ifdef __cplusplus
}
#endif

#endif // TFLITE_WRAPPER_H
