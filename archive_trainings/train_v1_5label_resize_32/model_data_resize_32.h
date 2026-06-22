// ============================================================
// AUTO-GENERATED model data  --  resize_32
// Sinh tu notebook (KHONG sua tay).
// TFLite INT8 size : 39192 bytes
// ------------------------------------------------------------
// Input  : shape=[np.int32(1), np.int32(200), np.int32(6)] dtype=int8 quant(scale=0.0078431377, zero_point=-1)
// Output : shape=[np.int32(1), np.int32(5)] dtype=int8 quant(scale=0.00390625, zero_point=-128)
// INT8<->real:  q = round(real/scale)+zero_point ; real = (q-zero_point)*scale
// ------------------------------------------------------------
// OPS cua mang (7) -- dang ky DUNG nhung op nay:
//   - CONV_2D
//   - FULLY_CONNECTED
//   - MAX_POOL_2D
//   - RESHAPE
//   - SOFTMAX
//   - STRIDED_SLICE
//   - UNIDIRECTIONAL_SEQUENCE_LSTM
// (Bo qua pseudo-op: DELEGATE)
//
// GOI Y MicroMutableOpResolver (so op = 7):
//   static tflite::MicroMutableOpResolver<7> resolver;
//   resolver.AddConv2D();
//   resolver.AddFullyConnected();
//   resolver.AddMaxPool2D();
//   resolver.AddReshape();
//   resolver.AddSoftmax();
//   resolver.AddStridedSlice();
//   resolver.AddUnidirectionalSequenceLSTM();
// ============================================================
#ifndef MODEL_DATA_H_
#define MODEL_DATA_H_

#define MODEL_INPUT_LEN   1200
#define MODEL_OUTPUT_LEN  5
#define MODEL_NUM_OPS     7

#ifdef __cplusplus
extern "C" {
#endif

extern const unsigned char g_model_data[];
extern const unsigned int g_model_data_len;

#ifdef __cplusplus
}
#endif

#endif  // MODEL_DATA_H_
