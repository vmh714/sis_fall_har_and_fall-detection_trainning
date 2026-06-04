import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, SeparableConv1D, BatchNormalization, Activation, Multiply, Add, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def se_block_v25(x, c, ratio=4):
    """SE Block tối ưu hóa hoàn toàn bằng Conv1D 1x1 (Pointwise) cho ESP-NN"""
    se = tf.reduce_mean(x, axis=1, keepdims=True)
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
    # 200x6 -> 100x16. Dùng kernel_size=3 thay cho 5 để tối ưu tối đa ESP-NN
    x = Conv1D(16, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu6')(x)
    
    # Các khối Residual (Áp dụng tăng dần Filters và full Kernel Size = 3)
    # Block 1: 100x16 -> 100x16
    x = resnet1d_block(x, filters=16, kernel_size=3, strides=1)
    
    # Block 2: 100x16 -> 50x32
    x = resnet1d_block(x, filters=32, kernel_size=3, strides=2)
    
    # Block 3: 50x32 -> 50x32
    x = resnet1d_block(x, filters=32, kernel_size=3, strides=1)
    
    # Block 4: 50x32 -> 25x64
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

def get_class_weights():
    # Điều chỉnh class weight dựa trên phân tích báo cáo tập test
    # Mục tiêu: Đẩy mạnh Fall và Trans, giảm Idle
    # Lớp Idle (dữ liệu nhiều nhất) có trọng số thấp nhất. Lớp Fall (dữ liệu ít nhất, quan trọng nhất) có trọng số cao nhất.
    # Note: Chỉnh sửa lại index tương ứng với Label Encoder trong code thực tế của bạn.
    # Giả sử: 0: Walk, 1: Run, 2: Idle, 3: Trans, 4: Fall
    return {
        0: 1.0,  # Walk
        1: 1.0,  # Run
        2: 0.5,  # Idle (Chiếm 40% dataset, penalty thấp)
        3: 2.0,  # Trans (Cần nâng precision, chống nhầm lẫn với Idle)
        4: 3.5   # Fall (Chỉ chiếm ~8% dataset, penalty cực cao để đảm bảo Recall)
    }

if __name__ == "__main__":
    print("Khởi tạo mô hình ResNet-1D v25...")
    model = build_resnet1d_v25(input_shape=(200, 6), n_classes=5)
    model.summary()

    # (Optional) Chỗ này load dữ liệu thật của bạn
    # X_train, y_train, X_val, y_val = load_your_data()
    
    # Ví dụ Callbacks
    checkpoint = ModelCheckpoint(
        'best_resnet1d_v25.keras', 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max',
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-5,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    # Ví dụ Training function call (Commented out để không báo lỗi khi chạy thử)
    """
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        class_weight=get_class_weights(),  # <--- Bắt buộc dùng Class Weights
        callbacks=[checkpoint, reduce_lr, early_stopping]
    )
    """
    print("Script khởi tạo kiến trúc thành công! Hãy nạp dữ liệu và tiến hành huấn luyện.")
