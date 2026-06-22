import re
import tflite

with open('sis_fall_firmware_inference/main/model_data.cc', 'r') as f:
    text = f.read()

match = re.search(r'\{([0-9a-fxA-F, \n]+)\}', text)
if match:
    bytes_str = match.group(1).replace('0x', '').replace(',', '').replace('\n', '').replace(' ', '')
    model_fb = bytes.fromhex(bytes_str)
    
    model = tflite.Model.GetRootAsModel(model_fb, 0)
    for i in range(model.OperatorCodesLength()):
        opcode = model.OperatorCodes(i)
        
        # Try different ways to get the code
        try:
            code1 = opcode.BuiltinCode()
        except:
            code1 = -1
        try:
            code2 = opcode.DeprecatedBuiltinCode()
        except:
            code2 = -1
            
        print(f"Op {i}: BuiltinCode={code1}, DeprecatedBuiltinCode={code2}")
