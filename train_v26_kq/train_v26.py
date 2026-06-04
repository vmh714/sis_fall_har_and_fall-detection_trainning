import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, Add, GlobalAveragePooling1D, Dense, Multiply, GlobalMaxPooling1D, Concatenate, SeparableConv1D
from tensorflow.keras.models import Model
from ml_pipeline_v26 import DataPreprocessor, OutputReporter

# Cấu hình
DATA_DIR = 'SisFall_dataset_Windowed'
CACHE_DIR = 'train_cache_v26'  # Sử dụng thư mục cache mới cho v26
OUT_DIR = 'train_v26_kq'
CLASS_NAMES = ['Walk', 'Run', 'Idle', 'Trans', 'Fall']

# Khởi tạo Pipeline v26
preprocessor = DataPreprocessor(DATA_DIR, CACHE_DIR, CLASS_NAMES)
reporter = OutputReporter(OUT_DIR, CLASS_NAMES)

# Nạp và Tiền xử lý dữ liệu
X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.load_or_create_dataset()
X_train, X_val, X_test = preprocessor.apply_preprocessing(X_train, X_val, X_test)
class_weights = preprocessor.get_balanced_class_weights(y_train)

# --- KIẾN TRÚC MÔ HÌNH ResNet-1D v26 TỐI ƯU ESP-NN ---
def se_block_v26(inputs, filters):
    # Ratio động: Đảm bảo bottleneck có tối thiểu 8 nơ-ron
    ratio = max(2, filters // 8)
    
    se = GlobalAveragePooling1D()(inputs)
    se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)
    return Multiply()([inputs, se])

def resnet1d_block_v26(inputs, filters, strides=1):
    # Dùng SeparableConv1D kết hợp kernel=3 để tối ưu ESP-NN INT8
    x = SeparableConv1D(filters, kernel_size=3, strides=strides, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6.0)(x)  # Dùng ReLU6 tốt cho Quantization
    
    x = SeparableConv1D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    
    x = se_block_v26(x, filters)
    
    shortcut = inputs
    if strides != 1 or inputs.shape[-1] != filters:
        shortcut = Conv1D(filters, kernel_size=1, strides=strides, padding='same', use_bias=False)(inputs)
        shortcut = BatchNormalization()(shortcut)
        
    x = Add()([x, shortcut])
    x = ReLU(max_value=6.0)(x)
    return x

def build_resnet1d_v26(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # Stem - Vẫn giữ kernel=3 để tuân thủ ESP-NN SIMD
    x = Conv1D(16, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6.0)(x)
    
    # Blocks [16, 32, 64, 96] - Đảm bảo bội số của 8
    x = resnet1d_block_v26(x, filters=16, strides=1) # Block 1 (100x16)
    x = resnet1d_block_v26(x, filters=32, strides=2) # Block 2 (50x32)
    x = resnet1d_block_v26(x, filters=32, strides=1) # Block 3 (50x32)
    x = resnet1d_block_v26(x, filters=64, strides=2) # Block 4 (25x64)
    
    # Block 5 MỚI: Strides=1, Tăng filters lên 96 giúp nới Receptive Field tại tầng đáy
    x = resnet1d_block_v26(x, filters=96, strides=1) # Block 5 (25x96)
    
    # Global Pooling kết hợp
    x_avg = GlobalAveragePooling1D()(x)
    x_max = GlobalMaxPooling1D()(x)
    x = Concatenate()([x_avg, x_max])
    
    # Fully Connected
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

model = build_resnet1d_v26(input_shape=(200, 6), num_classes=len(CLASS_NAMES))
model.summary()

# Sử dụng Focal Loss (có sẵn trong tf.keras.losses bản mới)
# Gamma=2.0 giúp ép mô hình sửa sai các ca khó như Idle/Trans
focal_loss = tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=focal_loss,
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(OUT_DIR, 'best_model_v26.keras'), monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]

# Chuyển đổi y_train, y_val sang one-hot encoding cho FocalCrossentropy
y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=len(CLASS_NAMES))
y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes=len(CLASS_NAMES))

print("\n[*] Bắt đầu huấn luyện mô hình v26...")
history = model.fit(
    X_train, y_train_oh,
    validation_data=(X_val, y_val_oh),
    epochs=100,
    batch_size=256,
    class_weight=class_weights,
    callbacks=callbacks
)

# Đánh giá & Báo cáo
reporter.plot_training_history(history, version='v26')

# Chú ý: Evaluate cần nhãn gốc (không one-hot)
model.load_weights(os.path.join(OUT_DIR, 'best_model_v26.keras'))
reporter.evaluate_and_report(model, X_test, y_test, version='v26')
print("\n[*] Quá trình huấn luyện v26 hoàn tất!")
