# Kế hoạch Triển khai Kiến trúc TCN/ResNet-1D cho Phiên bản v25

Dựa trên phân tích từ quá trình brainstorm (v23-v24) và báo cáo kết quả của firmware v24, dưới đây là đề xuất kế hoạch nâng cấp và triển khai cho **v25** với mục tiêu tối ưu hóa cực đại cho bộ tăng tốc **ESP32-S3 (ESP-NN)**, đồng thời giữ hoặc vượt mức độ chính xác của v24 (Accuracy > 91%, Fall Recall ~ 99.5%).

## 1. Nâng cấp Kiến trúc Mô hình (Model Architecture)

Mặc dù v24 đã có nhiều ý tưởng tuyệt vời (đổi sang `relu6`, dùng `SeparableConv1D`, `Conv1x1` cho SE block), v25 cần đi xa hơn nữa bằng cách áp dụng triết lý của **MobileNetV3**:

### 1.1. Bỏ `MaxPooling1D` ở Input, dùng **Conv1D Stem Layer**
- **Lý do:** Ở v24 (brainstorm 2), `MaxPooling1D` được đưa lại đầu mạng vì ESP-NN tăng tốc MaxPool 7.8x. Tuy nhiên, việc MaxPool trực tiếp trên dữ liệu IMU thô (6 kênh) sẽ phá hủy các đặc trưng vật lý liên tục (ví dụ: dạng sóng ngã).
- **Đề xuất v25:** Dùng `Conv1D` (hoặc `SeparableConv1D`) với `strides=2` ngay ở lớp đầu tiên. ESP-NN tăng tốc Conv1x1 (14x) và Depthwise (6x) rất tốt. Bước này vừa giúp giảm nửa chiều thời gian (200 -> 100) vừa học được *low-level features* hiệu quả.

### 1.2. Tinh chỉnh Kernel Size và Số lượng Filter
- **Lý do:** Dùng Kernel size 7 với 32 channels ở mọi block (v24) là hơi thừa khi chuỗi thời gian ngày càng ngắn lại (còn 50, 25 timestep).
- **Đề xuất v25:**
  - **Filter Mở rộng:** Dùng cấu trúc mở rộng (Bottleneck/Expansion) thay vì fix 32. Ví dụ: `Stem (16)` -> `Block 1 (16)` -> `Block 2 (32, stride 2)` -> `Block 3 (32)` -> `Block 4 (64, stride 2)`.
  - **Kernel Size:** Giảm kernel size xuống `5` cho 2 block đầu và `3` cho 2 block cuối. Xếp chồng các Conv nhỏ sẽ tạo ra Receptive Field lớn (tương đương k=7) nhưng tốn ít tham số hơn rất nhiều.

### 1.3. Cải tiến Block Residual (MobileNetV3-like)
Biến Block Residual thành dạng *Inverted Bottleneck* cực nhẹ với thứ tự chuẩn:
`Depthwise Conv1D` -> `BN` -> `ReLU6` -> `Pointwise Conv1D (1x1)` -> `BN` -> `SE_Block` -> `Add`

---

## 2. Tối ưu Pipeline Kỹ thuật (Training & Export)

### 2.1. Full INT8 Post-Training Quantization (PTQ)
- Các con số tăng tốc ấn tượng trong bảng ESP-NN (VD: elementwise_add, depthwise conv) được phát huy **tối đa** khi tính toán ở định dạng số nguyên (INT8).
- Ở v25, bắt buộc phải dùng `tf.lite.Optimize.DEFAULT` và cung cấp **Representative Dataset** khi convert sang TFLite để ép toàn bộ Inferences (Weights + Activations) về INT8. 

### 2.2. Xử lý Mất cân bằng Dữ liệu (Class Weights)
- Nhìn vào Report v24: `Walk (200), Run (200), Idle (400), Trans (200), Fall (200)`. Lớp Idle có lượng data gấp đôi.
- **Đề xuất:** Kết hợp `label_smoothing=0.1` với `class_weight` trong hàm `model.fit()` để phạt nặng hơn nếu mô hình đoán sai lớp `Fall`, nhằm duy trì ngưỡng Recall 99.5% hoặc lên 100%.

### 2.3. Data Augmentation cho Time-Series
- Thêm lớp Custom Augmentation hoặc xử lý trong Dataset: Thêm nhiễu Gaussian nhẹ (Jittering) vào tín hiệu `ax, ay, az` để mô phỏng rung lắc thực tế của cảm biến khi đeo trên người, giúp mô hình bớt overfit và robust hơn.

---

## 3. Mã nguồn Kiến trúc v25 Đề xuất (Keras)

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, SeparableConv1D, BatchNormalization, Activation, Multiply, Add, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate, Dropout, Dense
from tensorflow.keras.models import Model

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
    
    # Thay thế MaxPool bằng Stem Conv (Học đặc trưng cấp thấp + Downsample)
    # 200x6 -> 100x16. Dùng kernel_size=3 thay cho 5 để tối ưu tối đa
    x = Conv1D(16, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu6')(x)
    
    # Các khối Residual (Áp dụng tăng dần Filters và full Kernel Size 3)
    # Block 1: 100x16 -> 100x16 (ks=3)
    x = resnet1d_block(x, filters=16, kernel_size=3, strides=1)
    
    # Block 2: 100x16 -> 50x32 (ks=3)
    x = resnet1d_block(x, filters=32, kernel_size=3, strides=2)
    
    # Block 3: 50x32 -> 50x32 (ks=3)
    x = resnet1d_block(x, filters=32, kernel_size=3, strides=1)
    
    # Block 4: 50x32 -> 25x64 (ks=3)
    x = resnet1d_block(x, filters=64, kernel_size=3, strides=2)
    
    # Dual Pooling (GAP + GMP) - Rất nhẹ với chuỗi chỉ còn độ dài 25
    gap = GlobalAveragePooling1D()(x)
    gmp = GlobalMaxPooling1D()(x)
    x = Concatenate()([gap, gmp])
    
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
```

## Tóm tắt Kế hoạch V25 (Action Items):
1. Khởi tạo môi trường v25, dùng file code kiến trúc `build_resnet1d_v25` phía trên.
2. Thiết lập hàm Data Augmentation tạo nhiễu ngẫu nhiên.
3. Huấn luyện với `class_weight` tương thích tỷ lệ (nhấn mạnh lớp Fall).
4. Viết script export mô hình có Full INT8 Quantization (TFLite Representative Dataset) để test trực tiếp khả năng load trên firmware ESP32-S3.
