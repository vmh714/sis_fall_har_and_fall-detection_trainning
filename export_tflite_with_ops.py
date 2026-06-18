# -*- coding: utf-8 -*-
"""
export_tflite_with_ops.py
-------------------------
Convert model Keras -> INT8 TFLite VA sinh model_data.cc / .h co NHUNG SAN:
  - Danh sach ops thuc su cua mang (da loc pseudo-op nhu DELEGATE)
  - Input/Output: shape, dtype, quant (scale, zero_point)
  - Goi y MicroMutableOpResolver (so op + tung dong resolver.AddXxx())
  - Cac #define tien dung cho wrapper (INPUT_LEN, OUTPUT_LEN, NUM_OPS)

Muc dich: nguoi viet TFLM wrapper KHONG phai doan mo cac phep toan cua mang.

Dung trong notebook (thay cho cell export cu):
    from export_tflite_with_ops import export_int8_with_ops
    export_int8_with_ops(model, X_train, out_dir=trainer.out_dir,
                         version="resize_64_32",
                         arch_note="CNN-LSTM (LSTM 64->32) - 5 nhan: Walk,Run,Idle,Trans,Fall")
LUU Y: notebook phai dang chay Keras 2 (TF_USE_LEGACY_KERAS=1) de LSTM fuse.
"""
from pathlib import Path
import numpy as np
import tensorflow as tf

# Pseudo-op KHONG can dang ky trong MicroMutableOpResolver -> loai khoi goi y
_NON_REGISTRABLE = {"DELEGATE", "CALL_ONCE", "CUSTOM"}

# Map TFLite op_name -> ham MicroMutableOpResolver tuong ung
_OP2ADD = {
    "CONV_2D": "AddConv2D",
    "DEPTHWISE_CONV_2D": "AddDepthwiseConv2D",
    "MAX_POOL_2D": "AddMaxPool2D",
    "AVERAGE_POOL_2D": "AddAveragePool2D",
    "FULLY_CONNECTED": "AddFullyConnected",
    "SOFTMAX": "AddSoftmax",
    "LOGISTIC": "AddLogistic",
    "TANH": "AddTanh",
    "RELU": "AddRelu",
    "RESHAPE": "AddReshape",
    "UNIDIRECTIONAL_SEQUENCE_LSTM": "AddUnidirectionalSequenceLSTM",
    "QUANTIZE": "AddQuantize",
    "DEQUANTIZE": "AddDequantize",
    "STRIDED_SLICE": "AddStridedSlice",
    "CONCATENATION": "AddConcatenation",
    "ADD": "AddAdd",
    "MUL": "AddMul",
    "SUB": "AddSub",
    "PACK": "AddPack",
    "UNPACK": "AddUnpack",
    "SHAPE": "AddShape",
    "FILL": "AddFill",
    "GATHER": "AddGather",
    "SLICE": "AddSlice",
    "SPLIT": "AddSplit",
    "TRANSPOSE": "AddTranspose",
    "MEAN": "AddMean",
    "PAD": "AddPad",
    "MINIMUM": "AddMinimum",
    "MAXIMUM": "AddMaximum",
    "EXPAND_DIMS": "AddExpandDims",
    "SQUEEZE": "AddSqueeze",
}


def _c_array_body(buf, per_line=12):
    out, line = [], "  "
    for i, b in enumerate(buf):
        line += f"0x{b:02x}, "
        if (i + 1) % per_line == 0:
            out.append(line.rstrip())
            line = "  "
    if line.strip():
        out.append(line.rstrip())
    return "\n".join(out).rstrip(", ")


def export_int8_with_ops(model, X_rep, out_dir, version,
                         input_shape=(200, 6), n_rep=300, seed=42,
                         arch_note="", var_name="g_model_data"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Representative dataset (X_rep gia su DA scale) ---
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_rep), size=min(n_rep, len(X_rep)), replace=False)

    def rep_gen():
        for i in idx:
            yield [X_rep[i:i + 1].astype(np.float32)]

    # --- Clone sang batch co dinh = 1 de LSTM chac chan fuse ---
    static_in = tf.keras.Input(batch_shape=(1,) + tuple(input_shape), dtype="float32")
    static_model = tf.keras.models.clone_model(model, input_tensors=static_in)
    static_model.set_weights(model.get_weights())

    conv = tf.lite.TFLiteConverter.from_keras_model(static_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tflite_model = conv.convert()

    tflite_path = out_dir / f"model_{version}_int8.tflite"
    tflite_path.write_bytes(tflite_model)

    # --- Doc metadata that tu interpreter ---
    it = tf.lite.Interpreter(model_content=tflite_model)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    outp = it.get_output_details()[0]
    all_ops = sorted({d["op_name"] for d in it._get_ops_details()})
    reg_ops = [op for op in all_ops if op not in _NON_REGISTRABLE]  # ops can dang ky

    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = outp["quantization"]
    in_len = int(np.prod(inp["shape"]))
    out_len = int(np.prod(outp["shape"]))

    # --- Goi y resolver ---
    resolver_lines = []
    for op in reg_ops:
        m = _OP2ADD.get(op)
        if m:
            resolver_lines.append(f"//   resolver.{m}();")
        else:
            resolver_lines.append(f"//   /* op '{op}' chua co trong map -> tu tra cuu Add method tuong ung */")

    nl = "\n"
    meta = (
        "// ============================================================\n"
        f"// AUTO-GENERATED model data  --  {version}\n"
        "// Sinh boi export_tflite_with_ops.py  --  KHONG sua tay.\n"
        + (f"// {arch_note}\n" if arch_note else "")
        + f"// TFLite INT8 size : {len(tflite_model)} bytes\n"
        + "// ------------------------------------------------------------\n"
        + f"// Input  : shape={list(inp['shape'])} dtype={inp['dtype'].__name__} "
          f"quant(scale={in_scale:.8g}, zero_point={in_zp})\n"
        + f"// Output : shape={list(outp['shape'])} dtype={outp['dtype'].__name__} "
          f"quant(scale={out_scale:.8g}, zero_point={out_zp})\n"
        + "// De doc/ghi tensor INT8:  q = round(real/scale) + zero_point ; real = (q - zero_point)*scale\n"
        + "// ------------------------------------------------------------\n"
        + f"// OPS cua mang ({len(reg_ops)}) -- dang ky DUNG nhung op nay (da loc pseudo-op):\n"
        + nl.join(f"//   - {op}" for op in reg_ops) + nl
        + (("// (Bo qua pseudo-op khong can dang ky: "
            + ", ".join(op for op in all_ops if op in _NON_REGISTRABLE) + ")\n")
           if any(op in _NON_REGISTRABLE for op in all_ops) else "")
        + "//\n"
        + f"// GOI Y MicroMutableOpResolver (so op = {len(reg_ops)}):\n"
        + f"//   static tflite::MicroMutableOpResolver<{len(reg_ops)}> resolver;\n"
        + nl.join(resolver_lines) + nl
        + "// ============================================================\n"
    )

    body = _c_array_body(tflite_model)
    cc = (
        meta
        + '#include "model_data.h"\n\n'
        + f"const unsigned char {var_name}[] alignas(16) = {{\n"
        + body
        + "\n};\n\n"
        + f"const unsigned int {var_name}_len = {len(tflite_model)};\n"
    )
    (out_dir / f"model_data_{version}.cc").write_text(cc, encoding="utf-8")

    guard = "MODEL_DATA_H_"
    h = (
        meta
        + f"#ifndef {guard}\n#define {guard}\n\n"
        + f"#define MODEL_INPUT_LEN   {in_len}   // = prod{tuple(int(x) for x in inp['shape'])}\n"
        + f"#define MODEL_OUTPUT_LEN  {out_len}\n"
        + f"#define MODEL_NUM_OPS     {len(reg_ops)}\n\n"
        + '#ifdef __cplusplus\nextern "C" {\n#endif\n\n'
        + f"extern const unsigned char {var_name}[];\n"
        + f"extern const unsigned int {var_name}_len;\n\n"
        + "#ifdef __cplusplus\n}\n#endif\n\n"
        + f"#endif  // {guard}\n"
    )
    (out_dir / f"model_data_{version}.h").write_text(h, encoding="utf-8")

    print(f"[*] Saved {tflite_path} ({len(tflite_model)} bytes)")
    print(f"[*] Ops can dang ky ({len(reg_ops)}): {reg_ops}")
    print(f"[*] Da sinh model_data_{version}.cc / .h (co nhung ops + quant + goi y resolver)")
    return tflite_model, reg_ops
