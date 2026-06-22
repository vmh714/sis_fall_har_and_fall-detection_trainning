import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="c:/Users/hung.vumanh2/Documents/SisFall-PreProcessing/train_v6_kq/model_dynamic_range.tflite")
interpreter.allocate_tensors()

print("Model Ops:")
for details in interpreter.get_tensor_details():
    pass # we can't easily get op names from tensor details

# A better way is to parse the tflite schema or just print interpreter details
ops = set()
for d in interpreter._get_ops_details():
    ops.add(d['op_name'])
print(ops)
