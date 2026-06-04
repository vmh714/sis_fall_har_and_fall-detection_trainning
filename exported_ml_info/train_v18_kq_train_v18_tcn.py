# ==================================================
# TRÍCH XUẤT TỪ FILE: train_v18_tcn.py
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
            
        # Dữ liệu mới đã có 6 features: ax, ay, az, gx, gy, gz
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
def build_tcn_mcu(input_shape, n_classes=5):
    inputs = Input(shape=input_shape, name='input_layer')
    x = BatchNormalization()(inputs)
    for stack in range(2):
        for d in [1, 2, 4, 8]:
            residual = x
            x = Conv1D(32, kernel_size=3, dilation_rate=d, padding='valid', activation='relu', use_bias=True)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            crop_size = 2 * d
            residual = Cropping1D(cropping=(crop_size, 0))(residual)
            
            if residual.shape[-1] != 32:
                residual = Conv1D(32, 1, padding='valid')(residual)
            x = Add()([x, residual])

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(n_classes, activation='softmax', name='output_layer')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- TIỀN XỬ LÝ (PREPROCESSING) TRƯỚC KHI TRAIN ---
def main_preprocessing_logic():
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_dataset()
        print(f"[*] Dữ liệu Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        class_weights_vals = class_weight.compute_class_weight(
            class_weight='balanced', classes=np.unique(y_train), y=y_train
        )
        class_weights = dict(zip(np.unique(y_train), class_weights_vals))
        input_shape = (X_train.shape[1], X_train.shape[2])
        y_pred_probs = best_model.predict(X_test, batch_size=256)