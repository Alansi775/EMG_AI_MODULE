# ============================
# 2_training.py (ConTraNet-Enhanced)
# ============================
# Drop-in replacement with:
# - Multi-scale CNN + Channel Attention + 2x Transformer blocks
# - Attention pooling
# - Stronger augmentation & optimization
# - Saves model as HDF5 (.h5) and metadata.pkl (compatible with your inference)
#
# Usage:
#   python 2_training.py [data_folder]
#   python 2_training.py data
#
# Outputs:
#   models/trained_model.h5
#   models/metadata.pkl
# ============================

import os, sys, pickle, random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, constraints
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------
# Reproducibility
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------
# Augmentations
# ---------------------------
def jitter(x, sigma=0.02):
    # Additive noise per channel scaled by channel std
    return x + np.random.normal(0, sigma * (np.std(x, axis=0, keepdims=True) + 1e-7), size=x.shape)

def scaling(x, sigma=0.1):
    # Channel-wise multiplicative scale
    factor = np.random.normal(1.0, sigma, size=(1, x.shape[1]))
    return x * factor

def time_warp(x, max_warp=0.07):
    """Random stretch/compress along time; resample back to same length."""
    T, C = x.shape
    warp = np.random.uniform(1 - max_warp, 1 + max_warp)
    new_len = max(8, int(T * warp))
    xp = np.linspace(0, T - 1, T)
    xp_new = np.linspace(0, T - 1, new_len)
    out = np.zeros((new_len, C), dtype=x.dtype)
    for ch in range(C):
        out[:, ch] = np.interp(xp_new, xp, x[:, ch])
    xp_back = np.linspace(0, new_len - 1, T)
    xw = np.zeros_like(x)
    for ch in range(C):
        xw[:, ch] = np.interp(xp_back, np.arange(new_len), out[:, ch])
    return xw

def freq_mask(x, max_mask_ratio=0.15):
    """Randomly mask some frequency bins (per channel) in rFFT domain."""
    T, C = x.shape
    Xf = np.fft.rfft(x, axis=0)
    # build mask (same shape as Xf)
    mask = np.ones_like(Xf, dtype=np.float32)
    bins = Xf.shape[0]
    if bins > 4 and np.random.rand() < 0.5:
        # choose a random contiguous band to suppress
        band = max(1, int(bins * max_mask_ratio))
        start = np.random.randint(1, bins - band)  # avoid DC bin
        mask[start:start+band, :] = 0.0
    Xf = Xf * mask
    xr = np.fft.irfft(Xf, n=T, axis=0)
    return xr.astype(np.float32)

def channel_dropout(x, p=0.1):
    """Randomly drop entire channels (simulate electrode failure)."""
    if np.random.rand() < p:
        T, C = x.shape
        drop_n = max(1, int(0.15 * C))
        idx = np.random.choice(C, drop_n, replace=False)
        x = x.copy()
        x[:, idx] = 0.0
    return x

def augment(window):
    # Stochastic pipeline (tuned for small EMG sets)
    if random.random() < 0.6: window = jitter(window, sigma=0.02)
    if random.random() < 0.5: window = scaling(window, sigma=0.1)
    if random.random() < 0.35: window = time_warp(window, max_warp=0.07)
    if random.random() < 0.35: window = freq_mask(window, max_mask_ratio=0.12)
    if random.random() < 0.25: window = channel_dropout(window, p=1.0)  # apply once with prob
    return window

# ---------------------------
# Segmentation
# ---------------------------
def segment_emg(emg, winlen=128, overlap=64):
    step = max(1, winlen - overlap)
    windows = [emg[i:i+winlen] for i in range(0, emg.shape[0] - winlen + 1, step)]
    return np.array(windows) if windows else np.empty((0, winlen, emg.shape[1]))

# ---------------------------
# Positional encoding (Lambda-friendly for load_model)
# ---------------------------
def add_pos_encoding(x, d_model):
    pos = tf.range(tf.shape(x)[1], dtype=tf.float32)             # (T,)
    i = tf.range(d_model, dtype=tf.float32)                      # (D,)
    angle_rates = 1.0 / tf.pow(10000.0, (2.0 * (i // 2)) / tf.cast(d_model, tf.float32))
    angles = tf.expand_dims(pos, 1) * tf.expand_dims(angle_rates, 0)  # (T,D)
    sines = tf.sin(angles[:, 0::2])
    coses = tf.cos(angles[:, 1::2])
    pe = tf.concat([sines, coses], axis=-1)                      # (T,D)
    return x + tf.expand_dims(pe, 0)                             # (1,T,D) broadcast

# ---------------------------
# Building blocks
# ---------------------------
def squeeze_excite(x, ratio=8):
    """Channel attention (Squeeze-and-Excitation)."""
    ch = x.shape[-1]
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(max(8, ch // ratio), activation='relu')(se)
    se = layers.Dense(ch, activation='sigmoid')(se)
    return layers.Multiply()([x, tf.expand_dims(se, 1)])

def multi_scale_cnn(inp):
    """Parallel temporal receptive fields."""
    c3 = layers.Conv1D(64, 3, padding='same', activation='elu')(inp)
    c5 = layers.Conv1D(64, 5, padding='same', activation='elu')(inp)
    c7 = layers.Conv1D(64, 7, padding='same', activation='elu')(inp)
    x = layers.Concatenate()([c3, c5, c7])                # (T, 192)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(0.30)(x)
    x = layers.AveragePooling1D(4)(x)
    x = layers.Conv1D(128, 3, padding='same', activation='elu',
                      kernel_constraint=constraints.MaxNorm(0.25))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)
    x = squeeze_excite(x, ratio=8)
    return x  # (T/4, 128)

def transformer_stack(x, d_model=128, n_heads=4, ff_dim=256, dropout=0.5, blocks=2):
    """Stacked Transformer encoder blocks with residuals."""
    # Project to model dimension + add positional encoding
    x = layers.Dense(d_model)(x)
    x = layers.Lambda(lambda t: add_pos_encoding(t, d_model))(x)

    key_dim = max(8, d_model // n_heads)
    for _ in range(blocks):
        attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=key_dim, dropout=dropout)(x, x)
        x = layers.Add()([x, attn])
        x = layers.LayerNormalization()(x)

        ffn = layers.Dense(ff_dim, activation='elu')(x)
        ffn = layers.Dropout(dropout)(ffn)
        ffn = layers.Dense(d_model)(ffn)
        x = layers.Add()([x, ffn])
        x = layers.LayerNormalization()(x)
    return x  # (T', d_model)

def attention_pooling(x):
    """Temporal attention pooling (learn where to look in time)."""
    w = layers.Dense(1, activation='tanh')(x)      # (T,1)
    w = tf.nn.softmax(w, axis=1)                   # (T,1)
    x = tf.reduce_sum(x * w, axis=1)               # (B, d_model)
    return x

# ---------------------------
# Model
# ---------------------------
def build_contranet_enhanced(input_shape, n_classes,
                             d_model=128, n_heads=4, ff_dim=256, dropout=0.5, blocks=2):
    inp = layers.Input(shape=input_shape)              # (T, C)

    x = multi_scale_cnn(inp)                           # (T', 128)
    x = transformer_stack(x, d_model, n_heads, ff_dim, dropout, blocks)  # (T', d_model)
    x = attention_pooling(x)                           # (B, d_model)

    x = layers.Dense(160, activation='relu')(x)
    x = layers.Dropout(0.50)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)

    model = models.Model(inp, out)
    # Backward-compatible loss (Intel TF builds sometimes lack label_smoothing)
    def make_loss():
        try:
            return tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.1)
        except TypeError:
            return tf.keras.losses.SparseCategoricalCrossentropy()

    loss = make_loss()

    opt  = optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    return model

# ---------------------------
# Dataset loading
# ---------------------------
def load_dataset(folder, winlen=128, overlap=64, apply_aug=True):
    cwd = os.getcwd()
    os.chdir(folder)
    gestures = dict(enumerate(sorted([d for d in os.listdir() if os.path.isdir(d)])))

    X, y = [], []
    for gi, gname in gestures.items():
        os.chdir(gname)
        files = sorted([t for t in os.listdir() if os.path.isfile(t)])
        for f in files:
            emg = np.loadtxt(f, delimiter=',')
            if emg.ndim == 1:
                emg = emg[:, None]
            segs = segment_emg(emg, winlen, overlap)
            if segs.size == 0:
                continue
            for s in segs:
                if apply_aug and random.random() < 0.5:
                    X.append(augment(s))
                else:
                    X.append(s)
            y.extend([gi] * segs.shape[0])
        os.chdir('..')
    os.chdir(cwd)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, gestures

# ---------------------------
# Normalization (per-channel)
# ---------------------------
def compute_stats(X):
    # per-channel stats over train-only
    mean = X.mean(axis=(0, 1), keepdims=True)       # (1,1,C)
    std  = X.std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1e-6
    return mean.astype(np.float32), std.astype(np.float32)

def norm(X, mean, std):
    return (X - mean) / std

# ---------------------------
# Main
# ---------------------------
def main():
    data_folder = 'data'
    model_dir   = 'models'
    winlen, overlap = 128, 64

    if len(sys.argv) > 1:
        data_folder = sys.argv[1]

    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, 'trained_model.h5')
    meta_file  = os.path.join(model_dir, 'metadata.pkl')
    ckpt_file  = os.path.join(model_dir, 'ckpt_best.weights.h5')  # weights-only to avoid JSON serialization mid-fit

    print(f'Loading dataset from: {data_folder}')
    X, y, gestures = load_dataset(data_folder, winlen, overlap, apply_aug=True)
    if X.size == 0:
        print("No data found. Check your folder structure and CSV contents.")
        sys.exit(1)

    n_classes = len(gestures)
    input_shape = (X.shape[1], X.shape[2])
    print(f"Dataset windows: {X.shape}, Classes: {n_classes} -> {gestures}")
    print(f"Input shape (T,C): {input_shape}")

    # Train/Val split (stratified)
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    # Per-channel normalization (fit on TRAIN only)
    mean, std = compute_stats(Xtr)
    Xtr = norm(Xtr, mean, std).astype(np.float32)
    Xval = norm(Xval, mean, std).astype(np.float32)

    # Build model
    model = build_contranet_enhanced(input_shape, n_classes,
                                     d_model=128, n_heads=4, ff_dim=256, dropout=0.5, blocks=2)

    # Class weights to handle imbalance (optional)
    classes = np.unique(ytr)
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=ytr)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

    # LR schedule: Cosine decay restarts
    steps_per_epoch = max(1, len(Xtr) // 64)
    lr_schedule = optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=5 * steps_per_epoch,
        t_mul=2.0,
        m_mul=0.8,
        alpha=1e-5
    )

    # Swap optimizer LR to schedule
    model.optimizer.learning_rate = lr_schedule

    cbs = [
        callbacks.EarlyStopping(patience=25, restore_best_weights=True, monitor='val_accuracy'),
        callbacks.ReduceLROnPlateau(patience=8, factor=0.5, monitor='val_loss', verbose=1, min_lr=1e-6),
        # Save weights only during training (avoids JSON serialization of Lambda)
        callbacks.ModelCheckpoint(filepath=ckpt_file, save_best_only=True,
                                  save_weights_only=True, monitor='val_accuracy', verbose=1)
    ]

    history = model.fit(
        Xtr, ytr,
        epochs=120,
        batch_size=64,                   # a bit larger helps attention training
        validation_data=(Xval, yval),
        callbacks=cbs,
        class_weight=class_weight,
        verbose=1
    )

    # Evaluate
    ypred = np.argmax(model.predict(Xval, verbose=0), axis=1)
    acc = accuracy_score(yval, ypred)
    cm  = confusion_matrix(yval, ypred)
    print("Validation accuracy:", acc)
    print("Confusion matrix:\n", cm)

    # Save final full model (.h5). Ensure best weights are loaded (EarlyStopping restore_best_weights=True).
    print(f"Saving model to {model_file}")
    model.save(model_file)  # HDF5 format; compatible with your 3_inference.py (custom_objects={'add_pos_encoding': add_pos_encoding})

    # Save metadata for inference
    meta = dict(
        gestures=gestures,              # {int: name}
        input_shape=input_shape,        # (T, C)
        winlen=winlen,
        overlap=overlap,
        n_classes=n_classes,
        mean=mean.squeeze(),            # (C,) for your inference convenience
        std=std.squeeze(),              # (C,)
        model_type='ConTraNet_Enhanced',
        label_encoder=None
    )
    with open(meta_file, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Saved metadata to {meta_file}")

if __name__ == '__main__':
    main()
