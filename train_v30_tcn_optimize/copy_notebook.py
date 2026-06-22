import json

src_file = r'd:\New folder\sis_fall_har_and_fall-detection_trainning\train_v30_tcn\SisFall_KFold_Experiments_v4.ipynb'
dst_file = r'd:\New folder\sis_fall_har_and_fall-detection_trainning\train_v30_tcn_optimize\SisFall_KFold_Experiments_v4_optimize.ipynb'

with open(src_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_model_code = [
    "import tensorflow as tf\n",
    "from tensorflow.keras.layers import (Input, Conv1D, SeparableConv1D, GlobalAveragePooling1D,\n",
    "                                     MaxPooling1D, Flatten, Concatenate, Dense, Dropout,\n",
    "                                     BatchNormalization, Activation, Add, Cropping1D, Multiply, Reshape)\n",
    "from tensorflow.keras.models import Model\n",
    "\n",
    "def se_block(input_tensor, c=32, ratio=4):\n",
    "    se = GlobalAveragePooling1D()(input_tensor)\n",
    "    se = Dense(c // ratio, activation='relu', use_bias=False)(se)\n",
    "    se = Dense(c, activation='sigmoid', use_bias=False)(se)\n",
    "    se = Reshape((1, c))(se)\n",
    "    return Multiply()([input_tensor, se])\n",
    "\n",
    "def build_model(input_shape=(200, 6), n_classes=5):\n",
    "    # TCN TOI UU ESP-NN: Conv k7 thuong -> SeparableConv (depthwise dilated + pointwise 1x1 14x);\n",
    "    #   relu -> relu6 (11x); head GMP(REDUCE_MAX) -> MaxPool(full)+Flatten (MAX_POOL_2D 7.83x).\n",
    "    #   GIU dilation + SE (nguon suc manh Trans). Dilated depthwise van reference nhung nhe hon conv thuong.\n",
    "    inputs = Input(shape=input_shape, name='input_layer')\n",
    "    x = BatchNormalization()(inputs)\n",
    "    for stack in range(2):\n",
    "        for d in [1, 2, 4, 8]:\n",
    "            residual = x\n",
    "            x = SeparableConv1D(32, kernel_size=7, dilation_rate=d, padding='valid', use_bias=False)(x)\n",
    "            x = BatchNormalization()(x); x = Activation('relu6')(x)\n",
    "            x = Dropout(0.2)(x)\n",
    "            crop_size = 6 * d\n",
    "            residual = Cropping1D(cropping=(crop_size, 0))(residual)\n",
    "            if residual.shape[-1] != 32:\n",
    "                residual = Conv1D(32, 1, padding='valid', use_bias=False)(residual)\n",
    "            x = se_block(x, c=32, ratio=4)\n",
    "            x = Add()([x, residual]); x = Activation('relu6')(x)\n",
    "    gap = GlobalAveragePooling1D()(x); gap = Dropout(0.2)(gap)\n",
    "    m = MaxPooling1D(pool_size=int(x.shape[1]))(x); m = Flatten()(m); m = Dropout(0.4)(m)\n",
    "    x = Concatenate()([gap, m])\n",
    "    outputs = Dense(n_classes, activation='softmax', name='output_layer')(x)\n",
    "    model = Model(inputs=inputs, outputs=outputs)\n",
    "    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),\n",
    "                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),\n",
    "                  metrics=['accuracy'])\n",
    "    return model\n"
]

for cell in nb['cells']:
    # Replace markdown titles to show it's the optimize version
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if "# K-Fold v4 — TCN" in source:
            cell['source'] = [source.replace("# K-Fold v4 — TCN (cross-validate v30_tcn)", "# K-Fold v4 — TCN Optimize (v30_tcn_optimize)")]
        elif "Kiến trúc Model — **TCN v22**" in source:
            cell['source'] = ["# 6. Kiến trúc Model — **TCN Optimize** (SeparableConv + MaxPool Flatten)"]
            
    # Replace the build_model code
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "def build_model(" in source and "def se_block(" in source:
            cell['source'] = new_model_code
            
        if "DRIVE_OUT = Path('/media/data3/users/hungvm/dataset/sisfall/KFold_Results_v4')" in source:
            # Modify DRIVE_OUT
            new_source = []
            for line in cell['source']:
                if "DRIVE_OUT = Path('/media/data3/users/hungvm/dataset/sisfall/KFold_Results_v4')" in line:
                    new_source.append("DRIVE_OUT = Path('/media/data3/users/hungvm/dataset/sisfall/KFold_Results_v30_tcn_optimize')\n")
                elif "print(f\"[*] Ket qua se duoc luu tai: {DRIVE_OUT}\")" in line:
                    new_source.append(line)
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open(dst_file, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
