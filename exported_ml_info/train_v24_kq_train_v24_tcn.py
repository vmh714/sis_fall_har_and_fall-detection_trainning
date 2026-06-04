# ==================================================
# TRÍCH XUẤT TỪ FILE: train_v24_tcn.py
# ==================================================

# --- LOGIC GẮN NHÃN & PREPARE DATA ---
def parse_filename_info(filename):
    if '_Trans_' in filename:
        return 'Trans'
    if '_StandSit_' in filename or '_Lie_' in filename:
        return 'Idle'
    
    prefix = filename[:3]
    if prefix in ['D01', 'D02', 'D05', 'D06']:
        return 'Walk'
    if prefix in ['D03', 'D04']:
        return 'Run'
    if prefix.startswith('F'):
        return 'Fall'
        
    return None

# --- LOGIC GẮN NHÃN & PREPARE DATA ---
def load_single_csv(file_path):
    try:
        filename = file_path.stem
        parts = filename.split('_')
        subject_id = parts[1]
        
        label_str = parse_filename_info(filename)
        if label_str is None:
            return None
            
        label = CLASS_NAMES.index(label_str)
        
        df = pd.read_csv(file_path)
        if len(df) != 200:
            return None
            
        # Dữ liệu 6 features
        data = df.to_numpy()
        return data, label, subject_id
    except Exception as e:
        return None

# --- LOGIC GẮN NHÃN & PREPARE DATA ---
def prepare_dataset():
    train_cache_x = CACHE_DIR / 'X_train.npy'
    train_cache_y = CACHE_DIR / 'y_train.npy'
    val_cache_x = CACHE_DIR / 'X_val.npy'
    val_cache_y = CACHE_DIR / 'y_val.npy'
    test_cache_x = CACHE_DIR / 'X_test.npy'
    test_cache_y = CACHE_DIR / 'y_test.npy'
    
    if (train_cache_x.exists() and train_cache_y.exists() and 
        val_cache_x.exists() and val_cache_y.exists() and 
        test_cache_x.exists() and test_cache_y.exists()):
        print("[*] Phát hiện cache dữ liệu (5 classes). Đang tải...")
        X_train = np.load(train_cache_x)
        y_train = np.load(train_cache_y)
        X_val = np.load(val_cache_x)
        y_val = np.load(val_cache_y)
        X_test = np.load(test_cache_x)
        y_test = np.load(test_cache_y)
        return X_train, y_train, X_val, y_val, X_test, y_test

    print("[*] Đang quét thư mục dữ liệu windowed mới...")
    all_files = list(DATA_DIR.rglob('*.csv'))
    
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(load_single_csv, all_files)
        for res in results:
            if res is None: continue
            data, label, subject_id = res
            if subject_id in TRAIN_SUBJECTS:
                X_train_list.append(data)
                y_train_list.append(label)
            elif subject_id in VAL_SUBJECTS:
                X_val_list.append(data)
                y_val_list.append(label)
            elif subject_id in TEST_SUBJECTS:
                X_test_list.append(data)
                y_test_list.append(label)
                
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    X_val = np.array(X_val_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.int32)
    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.int32)
    
    np.save(train_cache_x, X_train)
    np.save(train_cache_y, y_train)
    np.save(val_cache_x, X_val)
    np.save(val_cache_y, y_val)
    np.save(test_cache_x, X_test)
    np.save(test_cache_y, y_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# --- KIẾN TRÚC MÔ HÌNH (ARCHITECTURE) ---
def se_block(input_tensor, c=32, ratio=4):
    """
    Squeeze-and-Excitation Block (Attention siêu nhẹ cho MCU)
    """
    se = GlobalAveragePooling1D()(input_tensor)
    se = Dense(c // ratio, activation=tf.nn.relu6, use_bias=False, kernel_regularizer=l2(1e-4))(se)
    se = Dense(c, activation='sigmoid', use_bias=False)(se)
    se = Reshape((1, c))(se)
    return Multiply()([input_tensor, se])

# --- KIẾN TRÚC MÔ HÌNH (ARCHITECTURE) ---
def build_tcn_mcu(input_shape, n_classes=5):
    inputs = Input(shape=input_shape, name='input_layer')
    x = BatchNormalization()(inputs)
    
    # Ép chuỗi thời gian xuống còn một nửa ngay từ đầu
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    
    # 4 khối Residual với Conv1D thuần túy, dùng Strides thay cho Dilation
    for i in range(4):
        residual = x
        
        # Ở block 1 và 3, ta dùng strides=2 để giảm chiều không gian, block 2 và 4 giữ nguyên strides=1
        s = 2 if i in [0, 2] else 1 
        
        x = Conv1D(32, kernel_size=7, strides=s, padding='same', activation=tf.nn.relu6, kernel_regularizer=l2(1e-4), use_bias=True)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Conv1D(32, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu6, kernel_regularizer=l2(1e-4), use_bias=True)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Match residual shape nếu có s > 1 hoặc số lượng kênh đổi
        if residual.shape[-1] != 32 or s != 1:
            residual = Conv1D(32, 1, strides=s, padding='same', kernel_regularizer=l2(1e-4))(residual)
            
        # SE Block
        x = se_block(x, c=32, ratio=4)
        
        x = Add()([x, residual])

    gap = GlobalAveragePooling1D()(x)
    gap = Dropout(0.2)(gap)
    
    gmp = GlobalMaxPooling1D()(x)
    gmp = Dropout(0.4)(gmp)
    
    x = Concatenate()([gap, gmp])
    
    outputs = Dense(n_classes, activation='softmax', name='output_layer')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    # SỬ DỤNG CATEGORICAL CROSSENTROPY CÓ LABEL SMOOTHING = 0.1
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), 
        metrics=['accuracy']
    )
    return model

# --- TIỀN XỬ LÝ (PREPROCESSING) TRƯỚC KHI TRAIN ---
def main_preprocessing_logic():
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_dataset()
        for X in [X_train, X_val, X_test]:
            # Clip giới hạn vật lý của IMU cấu hình 8g (nếu lớn hơn 8g thì bị cắt)
            X[:, :, 0:3] = np.clip(X[:, :, 0:3], -8.0, 8.0)
            
            # Scaling: Accel chia 8.0, Gyro chia 2000.0
            X[:, :, 0:3] = X[:, :, 0:3] / 8.0
            X[:, :, 3:6] = X[:, :, 3:6] / 2000.0
        print(f"[*] Dữ liệu Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        class_weights_vals = class_weight.compute_class_weight(
            class_weight='balanced', classes=np.unique(y_train), y=y_train
        )
        class_weights = dict(zip(np.unique(y_train), class_weights_vals))
        y_train_oh = to_categorical(y_train, num_classes=len(CLASS_NAMES))
        y_val_oh = to_categorical(y_val, num_classes=len(CLASS_NAMES))
        input_shape = (X_train.shape[1], X_train.shape[2])
        y_pred_probs = best_model.predict(X_test, batch_size=256)