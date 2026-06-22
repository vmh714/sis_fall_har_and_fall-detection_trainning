// ============================================================
// AUTO-GENERATED model data  --  v30_tcn_optimize_v2
// TFLite INT8 size : 65648 bytes
// ------------------------------------------------------------
// Input  : shape=[np.int32(1), np.int32(200), np.int32(6)] dtype=int8 quant(scale=0.0078431377, zero_point=-1)
// Output : shape=[np.int32(1), np.int32(5)] dtype=int8 quant(scale=0.00390625, zero_point=-128)
// INT8<->real:  q = round(real/scale)+zero_point ; real = (q-zero_point)*scale
// ------------------------------------------------------------
// OPS cua mang (12) -- dang ky DUNG nhung op nay:
//   - ADD
//   - CONCATENATION
//   - CONV_2D
//   - DEPTHWISE_CONV_2D
//   - FULLY_CONNECTED
//   - LOGISTIC
//   - MAX_POOL_2D
//   - MEAN
//   - MUL
//   - RESHAPE
//   - SOFTMAX
//   - STRIDED_SLICE
// (Bo qua pseudo-op: DELEGATE)
//
// GOI Y MicroMutableOpResolver (so op = 12):
//   static tflite::MicroMutableOpResolver<12> resolver;
//   resolver.AddAdd();
//   resolver.AddConcatenation();
//   resolver.AddConv2D();
//   resolver.AddDepthwiseConv2D();
//   resolver.AddFullyConnected();
//   resolver.AddLogistic();
//   resolver.AddMaxPool2D();
//   resolver.AddMean();
//   resolver.AddMul();
//   resolver.AddReshape();
//   resolver.AddSoftmax();
//   resolver.AddStridedSlice();
// ============================================================
#ifndef MODEL_DATA_H_
#define MODEL_DATA_H_

#define MODEL_INPUT_LEN   1200
#define MODEL_OUTPUT_LEN  5
#define MODEL_NUM_OPS     12

#ifdef __cplusplus
extern "C" {
#endif

extern const unsigned char g_model_data[];
extern const unsigned int g_model_data_len;

#ifdef __cplusplus
}
#endif

#endif  // MODEL_DATA_H_
