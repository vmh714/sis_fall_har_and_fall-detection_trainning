# ==================================================
# TRÍCH XUẤT TỪ FILE: train_v12_6ch_opt.py
# ==================================================

# --- LOGIC GẮN NHÃN & PREPARE DATA ---
def load_single_csv(file_path):
    try:
        parts = file_path.stem.split('_')
        label_code = parts[0]
        subject_id = parts[1]
        
        if label_code not in LABEL_MAP:
            return None
        label = LABEL_MAP[label_code]
        
        df = pd.read_csv(file_path)
        if len(df) != 200:
            return None
            
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
        print("[*] Phát hiện cache dữ liệu. Đang tải từ cache...")
        X_train = np.load(train_cache_x)
        y_train = np.load(train_cache_y)
        X_val = np.load(val_cache_x)
        y_val = np.load(val_cache_y)
        X_test = np.load(test_cache_x)
        y_test = np.load(test_cache_y)
        return X_train, y_train, X_val, y_val, X_test, y_test

    print("[*] Cache không tồn tại, bắt buộc phải sinh lại dữ liệu...")
    return None, None, None, None, None, None

# --- KIẾN TRÚC MÔ HÌNH (ARCHITECTURE) ---
def build_tcn_mcu(input_shape, n_classes=4):
    inputs = Input(shape=input_shape, name='input_layer')
    
    # [Cải tiến C]: Thêm Data Augmentation với Gaussian Noise
    x = GaussianNoise(0.05)(inputs)
    x = BatchNormalization()(x)
    
    for stack in range(2):
        for d in [1, 2, 4, 8]:
            residual = x
            
            # [Cải tiến B]: Lớp Conv1D đầu tiên (stack=0, d=1) dùng 64 filters, các lớp sau giữ nguyên 32
            current_filters = 64 if (stack == 0 and d == 1) else 32
            
            x = Conv1D(current_filters, kernel_size=3, dilation_rate=d, padding='valid', activation='relu', use_bias=True)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            crop_size = 2 * d
            residual = Cropping1D(cropping=(crop_size, 0))(residual)
            
            if residual.shape[-1] != current_filters:
                residual = Conv1D(current_filters, 1, padding='valid')(residual)
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
        print(f"[*] Dữ liệu gốc có shape: {X_train.shape}. Tiến hành cắt 6 trục (Accel+Gyro)...")
        X_train = X_train[:, :, :6]
        X_val = X_val[:, :, :6]
        X_test = X_test[:, :, :6]
        print(f"[*] Dữ liệu sau khi cắt: {X_train.shape}")
        class_weights_vals = class_weight.compute_class_weight(
            class_weight='balanced', classes=np.unique(y_train), y=y_train
        )
        class_weights = dict(zip(np.unique(y_train), class_weights_vals))
        print(f"[*] Trọng số lớp Fall được thiết lập: {class_weights[fall_idx]:.2f}")
        input_shape = (X_train.shape[1], X_train.shape[2])
        y_pred_probs = best_model.predict(X_test, batch_size=256)