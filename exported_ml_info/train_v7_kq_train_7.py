# ==================================================
# TRÍCH XUẤT TỪ FILE: train_7.py
# ==================================================

# --- LOGIC GẮN NHÃN & PREPARE DATA ---
def parse_filename_info(filename):
    """
    Phân tích tên file để lấy loại nhãn và ID đối tượng
    Ví dụ: D01_SA01_R01_W000 -> label_code='D01', subject_id='SA01'
    """
    parts = filename.split('_')
    label_code = parts[0]
    subject_id = parts[1]
    return label_code, subject_id

# --- LOGIC GẮN NHÃN & PREPARE DATA ---
def load_single_csv(file_path):
    """Đọc dữ liệu từ một file CSV window và tính toán thêm Pitch, Roll, SVM"""
    try:
        label_code, subject_id = parse_filename_info(file_path.stem)
        if label_code not in LABEL_MAP:
            return None
            
        label = LABEL_MAP[label_code]
        
        # Đọc dữ liệu gia tốc và góc quay (6 cột: ax,ay,az,gx,gy,gz)
        df = pd.read_csv(file_path)
        if len(df) != 200:
            return None # Bỏ qua các file lỗi kích thước
            
        data = df.to_numpy()
        
        # Trích xuất các cột gia tốc
        ax = data[:, 0]
        ay = data[:, 1]
        az = data[:, 2]
        
        # Tính Pitch, Roll, SVM
        pitch = np.arctan2(az, np.sqrt(ax**2 + ay**2)) * 180.0 / np.pi
        roll = np.arctan2(ax, np.sqrt(ay**2 + az**2)) * 180.0 / np.pi
        svm = np.sqrt(ax**2 + ay**2 + az**2)
        
        # Nối vào ma trận data
        pitch = pitch.reshape(-1, 1)
        roll = roll.reshape(-1, 1)
        svm = svm.reshape(-1, 1)
        
        data_9f = np.concatenate((data, pitch, roll, svm), axis=1) # shape: (200, 9)
        return data_9f, label, subject_id
    except Exception as e:
        return None

# --- LOGIC GẮN NHÃN & PREPARE DATA ---
def prepare_dataset():
    """Gộp dữ liệu từ các file CSV và chia tập Train/Val/Test (LSO)"""
    train_cache_9f_x = CACHE_DIR / 'X_train.npy'
    train_cache_9f_y = CACHE_DIR / 'y_train.npy'
    val_cache_x = CACHE_DIR / 'X_val.npy'
    val_cache_y = CACHE_DIR / 'y_val.npy'
    test_cache_x = CACHE_DIR / 'X_test.npy'
    test_cache_y = CACHE_DIR / 'y_test.npy'
    
    # Nếu đã có cache, load lên cho cực nhanh
    if (train_cache_9f_x.exists() and train_cache_9f_y.exists() and 
        val_cache_x.exists() and val_cache_y.exists() and 
        test_cache_x.exists() and test_cache_y.exists()):
        print("[*] Phát hiện cache dữ liệu đã được gộp. Đang tải từ cache...")
        X_train = np.load(train_cache_9f_x)
        y_train = np.load(train_cache_9f_y)
        X_val = np.load(val_cache_x)
        y_val = np.load(val_cache_y)
        X_test = np.load(test_cache_x)
        y_test = np.load(test_cache_y)
        return X_train, y_train, X_val, y_val, X_test, y_test

    print("[*] Đang quét thư mục dữ liệu windowed...")
    all_files = list(DATA_DIR.rglob('*.csv'))
    total_files = len(all_files)
    print(f"[*] Tìm thấy {total_files} file windowed. Bắt đầu tải dữ liệu song song...")
    
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []
    
    # Sử dụng ThreadPool để tăng tốc độ đọc I/O đĩa
    processed_count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(load_single_csv, all_files)
        for res in results:
            processed_count += 1
            if processed_count % 10000 == 0:
                print(f"  > Đã đọc {processed_count}/{total_files} file...")
                
            if res is None:
                continue
                
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
    
    print("\n[*] Thống kê kích thước tập dữ liệu:")
    print(f"  > Train shape: {X_train.shape}, Nhãn: {np.bincount(y_train)}")
    print(f"  > Val shape:   {X_val.shape}, Nhãn: {np.bincount(y_val)}")
    print(f"  > Test shape:  {X_test.shape}, Nhãn: {np.bincount(y_test)}")
    
    # Lưu cache npy
    print("[*] Đang lưu cache dữ liệu để tái sử dụng ở các lần chạy sau...")
    np.save(train_cache_9f_x, X_train)
    np.save(train_cache_9f_y, y_train)
    np.save(val_cache_x, X_val)
    np.save(val_cache_y, y_val)
    np.save(test_cache_x, X_test)
    np.save(test_cache_y, y_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# --- KIẾN TRÚC MÔ HÌNH (ARCHITECTURE) ---
def build_model(input_shape):
    """Xây dựng kiến trúc Trunk-and-Branches (Late-Branching) cho HAR & Fall Detection"""
    inputs = Input(shape=input_shape, name='input_layer')
    
    # =========================================================
    # THÂN CÂY (TRUNK) - Lọc nhiễu và trích xuất đặc trưng chung
    # =========================================================
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    # =========================================================
    # NHÁNH 1: FALL EXPERT (Chuyên rình gai nhọn)
    # =========================================================
    # Rút trích đỉnh cao nhất từ các đặc trưng đã được làm sạch
    fall_branch = GlobalMaxPooling1D()(x)
    fall_branch = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001))(fall_branch)
    
    # =========================================================
    # NHÁNH 2: HAR EXPERT (Chuyên đếm nhịp điệu Đi/Chạy)
    # =========================================================
    # Phân tích tính chuỗi thời gian của các đặc trưng đã làm sạch
    har_branch = LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))(x)
    har_branch = LSTM(32, kernel_regularizer=regularizers.l2(0.001))(har_branch)
    har_branch = Dropout(0.4)(har_branch)
    har_branch = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001))(har_branch)
    
    # =========================================================
    # HỢP NHẤT (Merge)
    # =========================================================
    merged = Concatenate()([fall_branch, har_branch])
    merged = Dropout(0.3)(merged)
    
    outputs = Dense(4, activation='softmax', name='output_layer')(merged)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

# --- TIỀN XỬ LÝ (PREPROCESSING) TRƯỚC KHI TRAIN ---
def main_preprocessing_logic():
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_dataset()
        print(f'[*] SHAPE thực tế của X_train: {X_train.shape}')
        class_weights_vals = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = dict(zip(np.unique(y_train), class_weights_vals))
        print(f"[*] Trọng số cân bằng các lớp (Class Weights): {class_weights}")
        input_shape = (X_train.shape[1], X_train.shape[2])
        y_pred_probs = best_model.predict(X_test, batch_size=256)