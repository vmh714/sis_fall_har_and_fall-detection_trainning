import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, SeparableConv1D, BatchNormalization, Activation, Multiply, Add, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate, Dropout, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from pathlib import Path

# Add project root to sys.path to import ml_pipeline
CURRENT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CURRENT_DIR))
from ml_pipeline_v26 import DataPreprocessor, OutputReporter

tf.keras.backend.set_floatx('float32')

class FallDetectionTrainerV25V2(DataPreprocessor, OutputReporter):
    """Lớp quản lý luồng huấn luyện cho v25_v2, sử dụng Data Augmentation (Jittering, Trans) từ v26_v2"""
    def __init__(self):
        # Đường dẫn cấu hình
        data_dir = CURRENT_DIR / 'SisFall_dataset_Windowed'
        cache_dir = CURRENT_DIR / 'train_cache_v26_v2' # Mount trực tiếp vào cache của v26_v2
        out_dir = CURRENT_DIR / 'train_v25_v2_kq'
        class_names = ['Walk', 'Run', 'Idle', 'Trans', 'Fall']
        
        # Khởi tạo 2 class cha
        DataPreprocessor.__init__(self, data_dir, cache_dir, class_names)
        OutputReporter.__init__(self, out_dir, class_names)
        
        self.version = "v25_v2"

def se_block_v25(x, c, ratio=4):
    """SE Block tối ưu hóa hoàn toàn bằng Conv1D 1x1 (Pointwise) cho ESP-NN"""
    se = GlobalAveragePooling1D()(x)
    se = Reshape((1, c))(se)
    # Tăng tốc pointwise 14.2x
    se = Conv1D(c // ratio, kernel_size=1, use_bias=False)(se)
    se = Activation('relu6')(se) # Tăng tốc 11.5x
    se = Conv1D(c, kernel_size=1, use_bias=False)(se)
    se = Activation('sigmoid')(se)
    return Multiply()([x, se])

def resnet1d_block(x, filters, kernel_size, strides, apply_se=True):
    residual = x
    
    # 1. Depthwise + Pointwise (Separable) -> ESP-NN optimize Depthwise 6.3x
    y = SeparableConv1D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    y = BatchNormalization()(y)
    y = Activation('relu6')(y)
    
    # 2. SeparableConv thứ hai không đổi kích thước
    y = SeparableConv1D(filters, kernel_size, strides=1, padding='same', use_bias=False)(y)
    y = BatchNormalization()(y)
    
    # Matching dimensions cho Residual connection
    if residual.shape[-1] != filters or strides != 1:
        residual = Conv1D(filters, kernel_size=1, strides=strides, padding='same', use_bias=False)(residual)
        residual = BatchNormalization()(residual)
        
    # SE Block
    if apply_se:
        y = se_block_v25(y, c=filters, ratio=4)
        
    y = Add()([y, residual])
    y = Activation('relu6')(y)
    return Dropout(0.2)(y)

def build_resnet1d_v25(input_shape=(200, 6), n_classes=5):
    inputs = Input(shape=input_shape)
    
    # Stem Conv (Học đặc trưng cấp thấp + Downsample)
    x = Conv1D(16, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu6')(x)
    
    x = resnet1d_block(x, filters=16, kernel_size=3, strides=1)
    x = resnet1d_block(x, filters=32, kernel_size=3, strides=2)
    x = resnet1d_block(x, filters=32, kernel_size=3, strides=1)
    x = resnet1d_block(x, filters=64, kernel_size=3, strides=2)
    
    # Dual Pooling (GAP + GMP)
    gap = GlobalAveragePooling1D()(x)
    gmp = GlobalMaxPooling1D()(x)
    x = Concatenate()([gap, gmp])
    
    # Đồng bộ Dropout cho cả cụm đặc trưng
    x = Dropout(0.3)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # Compile mô hình với Label Smoothing
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), 
        metrics=['accuracy']
    )
    return model

def main():
    trainer = FallDetectionTrainerV25V2()
    
    # 1. Nạp và tiền xử lý dữ liệu
    X_train, y_train, X_val, y_val, X_test, y_test = trainer.load_or_create_dataset()
    X_train, X_val, X_test = trainer.apply_preprocessing(X_train, X_val, X_test)
    
    # 2. Tính toán class weights (Do đã dùng 'balanced' nên tự động cân bằng số lượng, 
    # chỉ cần đẩy nhẹ Fall lên để ưu tiên recall, không nên phạt Idle hay nhân đôi Trans nữa)
    weight_modifiers = {'Fall': 3.0, 'Trans': 1.0, 'Idle': 1.0}
    class_weights = trainer.get_balanced_class_weights(y_train, weight_modifiers)
    
    # 3. One-hot encoding
    y_train_oh = to_categorical(y_train, num_classes=len(trainer.class_names))
    y_val_oh = to_categorical(y_val, num_classes=len(trainer.class_names))
    
    # 4. Khởi tạo mô hình
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"Khởi tạo mô hình ResNet-1D {trainer.version}...")
    model = build_resnet1d_v25(input_shape, n_classes=len(trainer.class_names))
    model.summary()
    
    # 5. Huấn luyện mô hình
    checkpoint_path = trainer.out_dir / f'best_model_{trainer.version}.keras'
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=str(checkpoint_path), monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    print(f"\\n[*] Bắt đầu huấn luyện mô hình ResNet-1D {trainer.version}...")
    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=100,
        batch_size=64,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2 
    )
    
    # 6. Đánh giá và báo cáo
    trainer.plot_training_history(history, version=trainer.version)
    
    best_model = tf.keras.models.load_model(str(checkpoint_path))
    trainer.evaluate_and_report(best_model, X_test, y_test, version=trainer.version, fall_threshold=0.25)

if __name__ == '__main__':
    main()
