import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, SeparableConv1D, BatchNormalization, Activation,
                                     Multiply, Add, GlobalAveragePooling1D, GlobalMaxPooling1D,
                                     Concatenate, Dropout, Dense, Reshape)
from tensorflow.keras.models import Model

# Kỹ thuật Hard-Sigmoid hỗ trợ tăng tốc trên phần cứng ESP-NN
def hard_sigmoid(x):
    return tf.nn.relu6(x + 3.) * 0.16666667

def se_block_v25(x, c, ratio=4):
    # SE block (Conv1D pointwise) — nhu v25
    se = GlobalAveragePooling1D()(x)
    se = Reshape((1, c))(se)
    se = Conv1D(c // ratio, kernel_size=1, use_bias=False)(se)
    se = Activation('relu6')(se)
    se = Conv1D(c, kernel_size=1, use_bias=False)(se)
    # TỐI ƯU: Thay thế Sigmoid (chậm) bằng Hard-Sigmoid (tăng tốc bằng relu6)
    se = Activation(hard_sigmoid)(se)
    return Multiply()([x, se])

def resnet1d_block(x, filters, kernel_size, strides, apply_se=True):
    residual = x
    y = SeparableConv1D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    y = BatchNormalization()(y); y = Activation('relu6')(y)
    y = SeparableConv1D(filters, kernel_size, strides=1, padding='same', use_bias=False)(y)
    y = BatchNormalization()(y)
    if residual.shape[-1] != filters or strides != 1:
        residual = Conv1D(filters, kernel_size=1, strides=strides, padding='same', use_bias=False)(residual)
        residual = BatchNormalization()(residual)
    if apply_se:
        y = se_block_v25(y, c=filters, ratio=4)
    y = Add()([y, residual]); y = Activation('relu6')(y)
    return Dropout(0.2)(y)

def build_model(input_shape=(200, 6), n_classes=5):
    # ResNet-1D v25 nguyen ban: stem Conv16 + 4 block (16,32,32,64) + GAP+GMP + Dense
    inputs = Input(shape=input_shape)
    x = Conv1D(16, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x); x = Activation('relu6')(x)
    x = resnet1d_block(x, filters=16, kernel_size=3, strides=1)
    x = resnet1d_block(x, filters=32, kernel_size=3, strides=2)
    x = resnet1d_block(x, filters=32, kernel_size=3, strides=1)
    x = resnet1d_block(x, filters=64, kernel_size=3, strides=2)
    
    # TỐI ƯU: Bỏ GlobalMaxPooling1D vì hàm REDUCE_MAX rớt xuống Reference (không được ESP-NN tăng tốc)
    # Ta chỉ dùng GAP (GlobalAveragePooling1D) vì MEAN được tăng tốc phần cứng
    gap = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(gap)
    
    outputs = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])
    return model

