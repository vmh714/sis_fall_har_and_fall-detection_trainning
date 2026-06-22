import re
import tflite
import sys

def analyze_cc(cc_path):
    text = open(cc_path).read()
    m = re.search(r'\{([0-9a-fxA-F, \n]+)\}', text)
    buf = bytes.fromhex(m.group(1).replace('0x','').replace(',','').replace('\n','').replace(' ',''))
    model = tflite.Model.GetRootAsModel(buf, 0)
    subgraph = model.Subgraphs(0)
    op_codes = [model.OperatorCodes(i).BuiltinCode() for i in range(model.OperatorCodesLength())]
    
    print(f'--- {cc_path.split("/")[-1] if "/" in cc_path else cc_path.split(chr(92))[-1]} ---')
    print(f'Tensors: {subgraph.TensorsLength()}, Ops: {subgraph.OperatorsLength()}')
    
    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)
        code = op_codes[op.OpcodeIndex()]
        op_name = [n for n in dir(tflite.BuiltinOperator) if getattr(tflite.BuiltinOperator, n) == code]
        op_name = op_name[0] if op_name else str(code)
        
        if code in [tflite.BuiltinOperator.CONV_2D, tflite.BuiltinOperator.DEPTHWISE_CONV_2D]:
            in_t = subgraph.Tensors(op.Inputs(0))
            fil_t = subgraph.Tensors(op.Inputs(1))
            out_t = subgraph.Tensors(op.Outputs(0))
            
            in_q = in_t.Quantization()
            out_q = out_t.Quantization()
            
            in_scale = in_q.Scale(0) if in_q and in_q.ScaleLength() > 0 else 0
            out_scale = out_q.Scale(0) if out_q and out_q.ScaleLength() > 0 else 0
            
            print(f'{op_name} -> in_scale: {in_scale:.6f}, out_scale: {out_scale:.6f}, filter_shape: {[fil_t.Shape(j) for j in range(fil_t.ShapeLength())]}')

analyze_cc(r'd:\New folder\sis_fall_har_and_fall-detection_trainning\train_v25_kq\model_data_v25.cc')
analyze_cc(r'd:\New folder\sis_fall_har_and_fall-detection_trainning\train_v30_resnet1d\model_data_v30_resnet1d.cc')
