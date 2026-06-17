"""
LOSO Personalization Pipeline — Bài báo #1
==========================================
Leave-One-Subject-Out + on-device personalization dưới ngân sách bộ nhớ.

Tái dùng:
  - DataPreprocessor.parse_filename_info  (label logic 5 lớp, khớp v25/v26)
  - Preprocessing fixed-scale (clip ±8g /8.0 accel, /2000.0 gyro)
  - Kiến trúc ResNet-1D v26 (SeparableConv, ESP-NN friendly)

Xem README_LOSO_methodology.md cho thiết kế khoa học & 5 chốt chặn đúng đắn.

Chạy:
  python loso/loso_pipeline.py --data-dir <NGUON_5_LOP> --subjects SA22 SE12 --epochs 20
  python loso/loso_pipeline.py --data-dir <NGUON_5_LOP> --full --epochs 60 --seeds 5
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import f1_score, recall_score

# --- import loader/preprocessing có sẵn của project ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from ml_pipeline_v26 import DataPreprocessor  # noqa: E402

CLASS_NAMES = ['Walk', 'Run', 'Idle', 'Trans', 'Fall']
FALL_IDX = CLASS_NAMES.index('Fall')
TRANS_IDX = CLASS_NAMES.index('Trans')
IDLE_IDX = CLASS_NAMES.index('Idle')
# Các lớp được phép dùng cho calibration (KHÔNG có Fall — chốt chặn #2)
ENROLL_CLASSES = [CLASS_NAMES.index(c) for c in ['Walk', 'Run', 'Idle', 'Trans']]


# =====================================================================
# 1. LOADER — giữ subject_id + trial_id cho từng window (không augment, không split)
# =====================================================================
def load_all_windows(data_dir):
    """Trả về X, y, subjects, trials cho TOÀN BỘ window (mọi subject)."""
    dp = DataPreprocessor(data_dir, cache_dir=str(Path(data_dir).parent / '_loso_cache'),
                          class_names=CLASS_NAMES)
    files = list(Path(data_dir).rglob('*.csv'))
    X, y, subjects, trials = [], [], [], []
    for fp in files:
        stem = fp.stem                       # vd: D01_SA01_R01_W000
        label_str = dp.parse_filename_info(stem)
        if label_str is None:
            continue
        parts = stem.split('_')
        subject = parts[1]                   # SA01 / SE03
        trial = parts[2] if len(parts) > 2 else 'R00'  # R01...
        df = pd.read_csv(fp)
        if len(df) != 200:
            continue
        X.append(df.to_numpy())
        y.append(CLASS_NAMES.index(label_str))
        subjects.append(subject)
        trials.append(trial)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    subjects = np.asarray(subjects)
    trials = np.asarray(trials)
    print(f"[loader] {len(X)} windows | {len(set(subjects))} subjects | "
          f"phân bố lớp: {np.bincount(y, minlength=len(CLASS_NAMES))}")
    return X, y, subjects, trials


def preprocess(X):
    """Fixed-scale giống v26 — KHÔNG z-score (chốt chặn #5). Trả về bản copy."""
    X = X.copy()
    X[:, :, 0:3] = np.clip(X[:, :, 0:3], -8.0, 8.0) / 8.0
    X[:, :, 3:6] = X[:, :, 3:6] / 2000.0
    return X


# =====================================================================
# 2. MODEL — ResNet-1D v26 (copy từ train_v26.py để import sạch)
# =====================================================================
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, ReLU, Add,  # noqa: E402
                                      GlobalAveragePooling1D, GlobalMaxPooling1D,
                                      Dense, Multiply, Concatenate, SeparableConv1D)
from tensorflow.keras.models import Model  # noqa: E402


def _se_block(x, filters):
    ratio = max(2, filters // 8)
    se = GlobalAveragePooling1D()(x)
    se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)
    return Multiply()([x, se])


def _block(x, filters, strides=1):
    y = SeparableConv1D(filters, 3, strides=strides, padding='same', use_bias=False)(x)
    y = BatchNormalization()(y); y = ReLU(6.0)(y)
    y = SeparableConv1D(filters, 3, strides=1, padding='same', use_bias=False)(y)
    y = BatchNormalization()(y)
    y = _se_block(y, filters)
    sc = x
    if strides != 1 or x.shape[-1] != filters:
        sc = Conv1D(filters, 1, strides=strides, padding='same', use_bias=False)(x)
        sc = BatchNormalization()(sc)
    y = Add()([y, sc]); y = ReLU(6.0)(y)
    return y


def build_model(input_shape=(200, 6), num_classes=5):
    inp = Input(input_shape)
    x = Conv1D(16, 3, strides=2, padding='same', use_bias=False)(inp)
    x = BatchNormalization()(x); x = ReLU(6.0)(x)
    x = _block(x, 16, 1)
    x = _block(x, 32, 2)
    x = _block(x, 32, 1)
    x = _block(x, 64, 2)
    x = _block(x, 96, 1)
    x = Concatenate()([GlobalAveragePooling1D()(x), GlobalMaxPooling1D()(x)])
    out = Dense(num_classes, activation='softmax')(x)
    return Model(inp, out)


def compile_model(model, lr=1e-3):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0),
                  metrics=['accuracy'])
    return model


# =====================================================================
# 3. ADAPTATION STRATEGIES — set trainable flags (trục co-design, mục 3 của README)
# =====================================================================
def set_trainable(model, strategy):
    """Đóng băng toàn bộ rồi mở khoá phần tương ứng với chiến lược."""
    for l in model.layers:
        l.trainable = False

    if strategy == 'full':
        for l in model.layers:
            l.trainable = True
    elif strategy == 'last_dense':
        model.layers[-1].trainable = True
    elif strategy == 'norm_tuning':            # TinyTL-style: chỉ BN affine (γ/β)
        for l in model.layers:
            if isinstance(l, BatchNormalization):
                l.trainable = True
        model.layers[-1].trainable = True       # + classifier
    elif strategy == 'se_only':                # chỉ Dense trong SE block + classifier
        for l in model.layers:
            if isinstance(l, Dense):
                l.trainable = True
    elif strategy == 'last_block':             # ~Block 5 cuối + classifier (heuristic theo tên)
        for l in model.layers[-15:]:
            l.trainable = True
    else:
        raise ValueError(f"strategy chưa hỗ trợ: {strategy}")
    # NOTE: 'sparse_topk' (kiểu On-Device-256KB) là đóng góp method — xem phần mở rộng cuối file.
    return model


def trainable_bytes(model):
    """Proxy chi phí bộ nhớ: tổng tham số trainable × 4 byte (fp32)."""
    n = int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))
    return n, n * 4 / 1024.0  # (params, KB)


# =====================================================================
# 4. SPLIT TRONG-SUBJECT theo TRIAL (chốt chặn #1) + bỏ Fall khỏi calibration (#2)
# =====================================================================
def split_calibration_test(y_s, trials_s, K, seed):
    """
    Chia index của 1 subject thành (calib_idx, test_idx).
    - test = các trial 'giữ lại để test'; calib = trial khác → KHÔNG rò rỉ theo trial.
    - calib chỉ lấy ENROLL_CLASSES (không Fall), K window/lớp, stratified.
    """
    rng = np.random.RandomState(seed)
    uniq_trials = np.unique(trials_s)
    rng.shuffle(uniq_trials)
    # nửa trial đầu -> nguồn calibration, nửa sau -> test (đảm bảo disjoint theo trial)
    n_calib_trials = max(1, len(uniq_trials) // 2)
    calib_trials = set(uniq_trials[:n_calib_trials])

    calib_pool = np.array([i for i in range(len(y_s))
                           if trials_s[i] in calib_trials and y_s[i] in ENROLL_CLASSES])
    test_idx = np.array([i for i in range(len(y_s)) if trials_s[i] not in calib_trials])

    # lấy K window/lớp (stratified) từ calib_pool
    calib_idx = []
    for c in ENROLL_CLASSES:
        pool_c = calib_pool[y_s[calib_pool] == c]
        if len(pool_c) == 0:
            continue
        take = min(K, len(pool_c))
        calib_idx.extend(rng.choice(pool_c, size=take, replace=False))
    return np.array(calib_idx, dtype=int), test_idx


# =====================================================================
# 5. METRICS
# =====================================================================
def evaluate(model, X, y):
    proba = model.predict(X, batch_size=256, verbose=0)
    pred = np.argmax(proba, axis=1)
    return {
        'f1_trans': f1_score(y, pred, labels=[TRANS_IDX], average='macro', zero_division=0),
        'f1_idle': f1_score(y, pred, labels=[IDLE_IDX], average='macro', zero_division=0),
        'macro_f1': f1_score(y, pred, average='macro', zero_division=0),
        'fall_recall': recall_score(y, pred, labels=[FALL_IDX], average='macro', zero_division=0),
    }


# =====================================================================
# 6. VÒNG LOSO
# =====================================================================
def run_loso(args):
    X, y, subjects, trials = load_all_windows(args.data_dir)
    all_subjects = sorted(set(subjects))
    target_subjects = all_subjects if args.full else args.subjects
    strategies = args.strategies

    rows = []
    for S in target_subjects:
        mask_S = subjects == S
        Xs, ys, ts = preprocess(X[mask_S]), y[mask_S], trials[mask_S]
        # base model = train trên TẤT CẢ subject khác (LOSO thật)
        mask_other = subjects != S
        Xo, yo = preprocess(X[mask_other]), y[mask_other]

        print(f"\n===== Held-out subject {S} | test windows={mask_S.sum()} =====")
        base = compile_model(build_model())
        yo_oh = tf.keras.utils.to_categorical(yo, len(CLASS_NAMES))
        base.fit(Xo, yo_oh, epochs=args.epochs, batch_size=256, verbose=0,
                 callbacks=[tf.keras.callbacks.EarlyStopping(
                     monitor='loss', patience=8, restore_best_weights=True)])

        for seed in range(args.seeds):
            calib_idx, test_idx = split_calibration_test(ys, ts, args.K, seed)
            if len(calib_idx) == 0 or len(test_idx) == 0:
                continue
            X_test, y_test = Xs[test_idx], ys[test_idx]

            # --- baseline trước adapt (zero-shot LOSO) ---
            m0 = evaluate(base, X_test, y_test)
            rows.append({'subject': S, 'group': S[:2], 'seed': seed,
                         'strategy': 'none', 'K': 0, 'trainable_kb': 0.0, **m0})

            # --- mỗi chiến lược personalization ---
            X_cal = Xs[calib_idx]
            y_cal_oh = tf.keras.utils.to_categorical(ys[calib_idx], len(CLASS_NAMES))
            for strat in strategies:
                model = tf.keras.models.clone_model(base)
                model.set_weights(base.get_weights())
                set_trainable(model, strat)
                compile_model(model, lr=args.adapt_lr)
                _, kb = trainable_bytes(model)
                model.fit(X_cal, y_cal_oh, epochs=args.adapt_epochs,
                          batch_size=32, verbose=0)
                m = evaluate(model, X_test, y_test)
                rows.append({'subject': S, 'group': S[:2], 'seed': seed,
                             'strategy': strat, 'K': args.K, 'trainable_kb': round(kb, 2), **m})
                print(f"  [{strat:11s}] ΔF1-Trans={m['f1_trans']-m0['f1_trans']:+.3f} "
                      f"FallRecall {m0['fall_recall']:.3f}->{m['fall_recall']:.3f} "
                      f"(+{kb:.1f}KB)")

    df = pd.DataFrame(rows)
    out = Path(__file__).parent / 'loso_results.csv'
    df.to_csv(out, index=False)
    print(f"\n[done] Lưu {len(df)} dòng -> {out}")
    _summary(df)


def _summary(df):
    print("\n=== Tóm tắt ΔF1-Trans theo chiến lược (mean qua subject×seed) ===")
    base = df[df.strategy == 'none'].groupby('subject')['f1_trans'].mean()
    for strat in sorted(df.strategy.unique()):
        if strat == 'none':
            continue
        sub = df[df.strategy == strat]
        delta = sub['f1_trans'].mean() - base.reindex(sub['subject']).mean()
        fr_drop = df[df.strategy == 'none']['fall_recall'].mean() - sub['fall_recall'].mean()
        kb = sub['trainable_kb'].mean()
        print(f"  {strat:12s} ΔF1-Trans={delta:+.3f}  FallRecallΔ={-fr_drop:+.3f}  "
              f"train≈{kb:.1f}KB")


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True,
                    help='Nguồn windowed 5 LỚP (KHÔNG phải folder local Walk/Run/Fall)')
    ap.add_argument('--subjects', nargs='+', default=['SA22', 'SA23', 'SE12'],
                    help='Subset subject cho pilot')
    ap.add_argument('--full', action='store_true', help='Chạy đủ 38 subject')
    ap.add_argument('--strategies', nargs='+',
                    default=['last_dense', 'norm_tuning', 'se_only', 'full'])
    ap.add_argument('--K', type=int, default=20, help='Số window/lớp cho calibration')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=40, help='Epoch train base LOSO')
    ap.add_argument('--adapt-epochs', type=int, default=15)
    ap.add_argument('--adapt-lr', type=float, default=5e-4)
    args = ap.parse_args()
    run_loso(args)


if __name__ == '__main__':
    main()

# =====================================================================
# MỞ RỘNG (đóng góp method — làm sau khi pilot xác nhận giả thuyết):
#   sparse_topk: thay vì đóng băng theo layer, chọn top-k% weight có |gradient| lớn nhất
#   trên tập calibration để update (tinh thần "On-Device Training under 256KB").
#   → cho phép phân bổ ngân sách bộ nhớ mịn hơn, là điểm novelty để nâng từ
#     workshop lên full paper. Cần tự viết mask gradient (GradientTape) thay vì layer.trainable.
# =====================================================================
