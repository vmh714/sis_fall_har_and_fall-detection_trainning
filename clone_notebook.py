import json
import re
import os

model_code = """import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, SeparableConv1D, GlobalAveragePooling1D,
                                     MaxPooling1D, Flatten, Concatenate, Dense, Dropout,
                                     BatchNormalization, Activation, Add, Cropping1D, Multiply, Reshape)
from tensorflow.keras.models import Model

def se_block_v25(x, c, ratio=4):
    \"\"\"SE Block tối ưu hóa hoàn toàn bằng Conv1D 1x1 (Pointwise) cho ESP-NN\"\"\"
    se = GlobalAveragePooling1D()(x)
    se = Reshape((1, c))(se)
    # Tăng tốc pointwise 14.2x
    se = Conv1D(c // ratio, kernel_size=1, use_bias=False)(se)
    se = Activation('relu6')(se) # Tăng tốc 11.5x
    se = Conv1D(c, kernel_size=1, use_bias=False)(se)
    se = Activation('sigmoid')(se)
    return Multiply()([x, se])

def build_model(input_shape=(200, 6), n_classes=5):
    # TCN v30 Optimize V2: 
    # 1. kernel_size=3 de kich hoat depthwise conv 3x3 optimized cua ESP-NN
    # 2. se_block dung Conv1D 1x1 (tham khao tu v25) thay vi Dense de tranh overhead
    inputs = Input(shape=input_shape, name='input_layer')
    x = BatchNormalization()(inputs)
    for stack in range(2):
        for d in [1, 2, 4, 8]:
            residual = x
            # separable conv k=3, dilation d (Depthwise k=3 optimized 6.3x)
            x = SeparableConv1D(32, kernel_size=3, dilation_rate=d, padding='valid', use_bias=False)(x)
            x = BatchNormalization()(x); x = Activation('relu6')(x)
            x = Dropout(0.2)(x)
            
            # crop_size = (k-1)*d = 2*d
            crop_size = 2 * d
            residual = Cropping1D(cropping=(crop_size, 0))(residual)
            if residual.shape[-1] != 32:
                # pointwise 1x1 conv tang toc 14.24x
                residual = Conv1D(32, 1, padding='valid', use_bias=False)(residual)
            
            # Dung se_block_v25 thay vi se_block cu
            x = se_block_v25(x, c=32, ratio=4)
            x = Add()([x, residual]); x = Activation('relu6')(x)
            
    gap = GlobalAveragePooling1D()(x); gap = Dropout(0.2)(gap)
    m = MaxPooling1D(pool_size=int(x.shape[1]))(x); m = Flatten()(m); m = Dropout(0.4)(m)
    x = Concatenate()([gap, m])
    outputs = Dense(n_classes, activation='softmax', name='output_layer')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])
    return model
"""

with open(r'd:\New folder\sis_fall_har_and_fall-detection_trainning\train_v30_tcn_optimize\train_v30_tcn_optimize.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if 'def build_model' in source:
            cell['source'] = [line + '\n' for line in model_code.split('\n')]
            cell['source'][-1] = cell['source'][-1].rstrip('\n')
        elif 'class FallDetectionTrainerResize(DataPreprocessor, OutputReporter):' in source or 'out_dir' in source or 'self.version' in source:
            new_source = []
            for line in cell.get('source', []):
                line = re.sub(r'out_dir\s*=\s*.*', "out_dir   = f'{WORKSPACE}/output_v30_tcn_optimize_v2'\n", line)
                line = re.sub(r'self.version\s*=\s*.*', 'self.version = "v30_tcn_optimize_v2"\n', line)
                new_source.append(line)
            cell['source'] = new_source

os.makedirs(r'd:\New folder\sis_fall_har_and_fall-detection_trainning\train_v30_tcn_optimize_v2', exist_ok=True)
with open(r'd:\New folder\sis_fall_har_and_fall-detection_trainning\train_v30_tcn_optimize_v2\train_v30_tcn_optimize_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated.")
