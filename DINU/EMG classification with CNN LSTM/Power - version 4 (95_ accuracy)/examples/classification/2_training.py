#Best training model so far

import sys, os, pickle, random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


# ================== Data augmentation functions ==================
def jitter(x, sigma=0.01):
    return x + np.random.normal(loc=0., scale=sigma * np.std(x), size=x.shape)

def scaling(x, sigma=0.1):
    factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[1],))
    return x * factor

def time_warp(x, max_warp=0.05):
    """Random stretch/compress along time axis."""
    orig_len = x.shape[0]
    warp = np.random.uniform(1 - max_warp, 1 + max_warp)
    new_len = int(orig_len * warp)
    x_new = np.interp(
        np.linspace(0, orig_len, new_len),
        np.arange(orig_len),
        x[:,0]
    )
    x_warped = np.zeros_like(x)
    for ch in range(x.shape[1]):
        x_warped[:, ch] = np.interp(
            np.linspace(0, new_len, orig_len),
            np.arange(new_len),
            np.interp(np.linspace(0, orig_len, new_len),
                      np.arange(orig_len), x[:, ch])
        )
    return x_warped

def augment(window):
    if random.random() < 0.5:
        window = jitter(window)
    if random.random() < 0.5:
        window = scaling(window)
    if random.random() < 0.3:
        window = time_warp(window)
    return window


# ================== EMG segmentation ==================
def segment_emg(emg, winlen=40, overlap=20):
    """
    Segment EMG signals into overlapping windows.
    emg: numpy array (samples, n_channels)
    Returns: (windows, winlen, n_channels)
    """
    windows = []
    step = winlen - overlap
    for start in range(0, emg.shape[0] - winlen, step):
        end = start + winlen
        windows.append(emg[start:end, :])
    return np.array(windows)


# ================== CNN-LSTM Model ==================
def build_cnn_lstm(input_shape, n_classes):
    model = models.Sequential([
        layers.Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Conv1D(128, kernel_size=5, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Conv1D(256, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ================== Main ==================
def main():
    # ======= Parameters ========
    data_folder = 'data'
    model_dir = 'models'
    model_file = os.path.join(model_dir, 'trained_model.keras')
    metadata_file = os.path.join(model_dir, 'metadata.pkl')

    winlen = 150
    overlap = 75

    if len(sys.argv) > 1:
        data_folder = sys.argv[1]
    if len(sys.argv) > 2:
        model_file = sys.argv[2] + '.keras'
        metadata_file = sys.argv[2] + '_metadata.pkl'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # ======= Load dataset ========
    os.chdir(data_folder)
    gestures = dict(enumerate(sorted([c for c in os.listdir() if os.path.isdir(c)])))

    X, y = [], []
    for (g, gesture) in gestures.items():
        os.chdir(gesture)
        trials = sorted([t for t in os.listdir() if os.path.isfile(t)])
        for trial in trials:
            emg = np.loadtxt(trial, delimiter=',')
            windows = segment_emg(emg, winlen=winlen, overlap=overlap)
            for w in windows:
                # Augment some windows (50% chance)
                if random.random() < 0.5:
                    w = augment(w)
                X.append(w)
            y.extend([g] * windows.shape[0])
        os.chdir('..')
    os.chdir('..')

    X = np.array(X)   # shape: (samples, winlen, n_channels)
    y = np.array(y)

    n_classes = len(gestures)
    input_shape = (X.shape[1], X.shape[2])

    print(f"Dataset shape: {X.shape}, Labels: {y.shape}")
    print(f"Number of classes: {n_classes}, Input shape: {input_shape}")

    # ======= Train/val split ========
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                      stratify=y, random_state=42)

    # ======= Build & train model ========
    model = build_cnn_lstm(input_shape, n_classes)

    cbs = [
        callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1)
    ]

    history = model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=32,
                        validation_data=(X_val, y_val),
                        callbacks=cbs,
                        verbose=1)

        # ======= Evaluate ========
    y_pred = np.argmax(model.predict(X_val), axis=1)
    acc = accuracy_score(y_val, y_pred)
    print("Validation accuracy:", acc)
    print("Confusion matrix:\n", confusion_matrix(y_val, y_pred))

        # ======= Save model ========
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print(f"Saving model to {model_file}")
    model.save(model_file)

    # Save metadata
    with open(metadata_file, 'wb') as f:
        pickle.dump({
            'gestures': gestures,
            'input_shape': input_shape,
            'winlen': winlen,
            'overlap': overlap,
            'n_classes': n_classes
        }, f)

if __name__ == '__main__':
    main()

