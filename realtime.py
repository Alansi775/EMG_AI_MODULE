# realtime.py
# Real-time gesture recognition using trained CNN+LSTM model

import asyncio
import myo
from myo import ClassifierMode, EMGMode, IMUMode
import torch
import torch.nn as nn
import numpy as np
from scipy import signal
from collections import deque
import time

# ── Config ──
FS           = 200
WIN_SAMPLES  = 40
N_CHANNELS   = 8
N_CLASSES    = 6
CONFIDENCE_THRESHOLD = 0.70   # minimum confidence to show prediction
SMOOTHING_WINDOW     = 5      # number of predictions to average

GESTURE_NAMES = {
    0: 'rest',
    1: 'fist',
    2: 'open_hand',
    3: 'wave_in',
    4: 'wave_out',
    5: 'pinch',
}

GESTURE_EMOJI = {
    0: '😐',
    1: '✊',
    2: '🖐',
    3: '👈',
    4: '👉',
    5: '🤌',
}


# ── Model (same architecture as training) ──
class EMG_CNN_LSTM(nn.Module):
    def __init__(self, n_channels=8, n_classes=6):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 64,  kernel_size=3, padding=1),
            nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.3),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.3),
        )
        self.lstm = nn.LSTM(
            input_size=256, hidden_size=128,
            num_layers=2, batch_first=True,
            dropout=0.3, bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(x)


# ── Load model ──
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model  = EMG_CNN_LSTM(n_classes=N_CLASSES).to(DEVICE)
model.load_state_dict(torch.load('best_model_v4.pt', map_location=DEVICE))
model.eval()
print(f"✅ Model loaded — Device: {DEVICE}")


# ── Filter (same as training) ──
nyq    = FS / 2
b,  a  = signal.butter(4, [20/nyq, 90/nyq], btype='band')
bn, an = signal.iirnotch(50, Q=30, fs=FS)


# ── Shared state ──
class State:
    # ring buffer — keeps last WIN_SAMPLES
    emg_buffer       = deque(maxlen=WIN_SAMPLES)
    raw_for_norm     = deque(maxlen=400)   # 2 seconds for normalization stats
    pred_history     = deque(maxlen=SMOOTHING_WINDOW)
    last_gesture     = -1
    last_print_time  = 0
    prediction_count = 0

STATE = State()


def predict():
    if len(STATE.emg_buffer) < WIN_SAMPLES:
        return None, 0.0

    # ── Preprocess ──
    window = np.array(STATE.emg_buffer, dtype=np.float32)  # (40, 8)

    # Filter
    window = signal.filtfilt(b,  a,  window, axis=0)
    window = signal.filtfilt(bn, an, window, axis=0)

    # Normalize using recent 2-second stats
    if len(STATE.raw_for_norm) >= 40:
        ref = np.array(STATE.raw_for_norm, dtype=np.float32)
        mean = ref.mean(axis=0)
        std  = np.where(ref.std(axis=0) < 1e-8, 1e-8, ref.std(axis=0))
        window = (window - mean) / std
    else:
        mean = window.mean(axis=0)
        std  = np.where(window.std(axis=0) < 1e-8, 1e-8, window.std(axis=0))
        window = (window - mean) / std

    # ── Inference ──
    x = torch.tensor(window.T, dtype=torch.float32)   # (8, 40)
    x = x.unsqueeze(0).to(DEVICE)                     # (1, 8, 40)

    with torch.no_grad():
        logits     = model(x)
        probs      = torch.softmax(logits, dim=1)[0]
        confidence = probs.max().item()
        pred_label = probs.argmax().item()

    return pred_label, confidence


class RealtimeClassifier(myo.MyoClient):

    async def on_emg_data(self, emg: myo.EMGData):
        for sample in [emg.sample1, emg.sample2]:
            STATE.emg_buffer.append(list(sample))
            STATE.raw_for_norm.append(list(sample))

        STATE.prediction_count += 1

        # predict every 10 samples (50ms)
        if STATE.prediction_count % 10 != 0:
            return

        pred_label, confidence = predict()
        if pred_label is None:
            return

        # Smoothing — majority vote over last N predictions
        STATE.pred_history.append(pred_label)
        if len(STATE.pred_history) < SMOOTHING_WINDOW:
            return

        # Most common prediction in history
        from collections import Counter
        smoothed_label = Counter(STATE.pred_history).most_common(1)[0][0]

        # Only print if changed or every 0.5 seconds
        now = time.time()
        changed = smoothed_label != STATE.last_gesture
        timeout = (now - STATE.last_print_time) > 0.5

        if (changed or timeout) and confidence >= CONFIDENCE_THRESHOLD:
            emoji = GESTURE_EMOJI[smoothed_label]
            name  = GESTURE_NAMES[smoothed_label]
            conf_bar = '█' * int(confidence * 20)
            print(f"\r  {emoji}  {name:<12}  "
                  f"conf: {confidence:.0%}  [{conf_bar:<20}]   ",
                  end='', flush=True)
            STATE.last_gesture    = smoothed_label
            STATE.last_print_time = now

        elif confidence < CONFIDENCE_THRESHOLD:
            print(f"\r  ❓  low confidence ({confidence:.0%})                        ",
                  end='', flush=True)

    async def on_imu_data(self, _): pass
    async def on_classifier_event(self, _): pass
    async def on_aggregated_data(self, _): pass
    async def on_emg_data_aggregated(self, _): pass
    async def on_fv_data(self, _): pass
    async def on_motion_event(self, _): pass


async def main():
    print("\n" + "═" * 52)
    print("  REAL-TIME EMG GESTURE RECOGNITION")
    print("═" * 52)
    print(f"\n  Gestures: {list(GESTURE_NAMES.values())}")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
    print(f"  Smoothing: {SMOOTHING_WINDOW} predictions\n")
    print("  🔍  Scanning for Myo Armband...")

    client = await RealtimeClassifier.with_device()
    print(f"  ✅  Connected: {client.device.name}\n")
    print("  Put on the Myo and try the gestures!")
    print("  Press Ctrl+C to stop.\n")
    print("─" * 52)

    await client.setup(
        classifier_mode=ClassifierMode.DISABLED,
        emg_mode=EMGMode.SEND_EMG,
        imu_mode=IMUMode.SEND_DATA,
    )
    await client.start()

    try:
        await asyncio.get_event_loop().run_in_executor(None, input)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        print("\n\n  Stopping...")
        await client.stop()
        await client.disconnect()
        print("  Done.")


if __name__ == "__main__":
    asyncio.run(main())