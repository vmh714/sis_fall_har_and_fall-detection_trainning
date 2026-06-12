import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CURRENT_DIR))
from ml_pipeline import DataPreprocessor, OutputReporter

tf.keras.backend.set_floatx('float32')


class FallDetectionTrainerV27(DataPreprocessor, OutputReporter):
    """CNN-LSTM baseline (kiến trúc v1) huấn luyện lại trên tập 5 nhãn"""
    def __init__(self):
        data_dir  = CURRENT_DIR / 'tool_for_new_dataset' / 'SisFall_dataset_Windowed'
        cache_dir = CURRENT_DIR / 'train_cache_v18_v19_idle_trans'
        out_dir   = CURRENT_DIR / 'train_v27_cnn_lstm_kq'
        class_names = ['Walk', 'Run', 'Idle', 'Trans', 'Fall']

        DataPreprocessor.__init__(self, data_dir, cache_dir, class_names)
        OutputReporter.__init__(self, out_dir, class_names)

        self.version = 'v27'


def build_cnn_lstm(input_shape=(200, 6), n_classes=5):
    """
    Kiến trúc CNN-LSTM giữ nguyên từ v1, chỉ mở rộng đầu ra lên 5 nhãn.
    Conv1D x2 → MaxPool → LSTM x2 → Dense head
    """
    inputs = Input(shape=input_shape)

    # --- CNN block: trích xuất đặc trưng cục bộ ---
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)      # 200 → 100
    x = Dropout(0.3)(x)

    # --- LSTM block: mã hóa ngữ cảnh thời gian ---
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(64)(x)
    x = Dropout(0.4)(x)

    # --- Classification head ---
    x = Dense(32, activation='relu')(x)
    outputs = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model


def main():
    trainer = FallDetectionTrainerV27()

    # 1. Nạp dữ liệu (dùng lại cache 5-nhãn của v25)
    X_train, y_train, X_val, y_val, X_test, y_test = trainer.load_or_create_dataset()
    X_train, X_val, X_test = trainer.apply_preprocessing(X_train, X_val, X_test)

    # 2. Class weights — đẩy Fall lên x3 giống v25
    weight_modifiers = {'Fall': 3.0, 'Trans': 1.0, 'Idle': 1.0}
    class_weights = trainer.get_balanced_class_weights(y_train, weight_modifiers)

    # 3. One-hot (cần cho label_smoothing loss)
    y_train_oh = to_categorical(y_train, num_classes=len(trainer.class_names))
    y_val_oh   = to_categorical(y_val,   num_classes=len(trainer.class_names))

    # 4. Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"Khởi tạo CNN-LSTM {trainer.version}...")
    model = build_cnn_lstm(input_shape, n_classes=len(trainer.class_names))
    model.summary()

    # 5. Callbacks
    checkpoint_path = trainer.out_dir / f'best_model_{trainer.version}.keras'
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=str(checkpoint_path), monitor='val_loss', save_best_only=True, verbose=1),
    ]

    # 6. Training
    print(f'\n[*] Bắt đầu huấn luyện {trainer.version}...')
    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=100,
        batch_size=256,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    # 7. Evaluate & report
    trainer.plot_training_history(history, version=trainer.version)

    best_model = tf.keras.models.load_model(str(checkpoint_path))
    trainer.evaluate_and_report(best_model, X_test, y_test, version=trainer.version, fall_threshold=0.25)


if __name__ == '__main__':
    main()
