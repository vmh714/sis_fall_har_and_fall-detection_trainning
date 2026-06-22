import json
import codecs

nb = json.load(open(r'd:\New folder\sis_fall_har_and_fall-detection_trainning\train_v30_resnet1d_optimize\keras2\train_v30_resnet1d_optimize.ipynb', encoding='utf-8'))
with codecs.open(r'd:\New folder\sis_fall_har_and_fall-detection_trainning\v30_extract_script.py', 'w', 'utf-8') as f:
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if 'def build_model' in src or 'def resnet1d_block' in src or 'def se_block' in src:
                f.write(src + '\n')
