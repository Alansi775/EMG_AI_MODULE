# debug_predict.py
# يطبع التوزيع الكامل للاحتمالات لكل تنبؤ — لتشخيص المشكلة

import asyncio
import myo
from myo import ClassifierMode, EMGMode, IMUMode
import torch
import torch.nn as nn
import numpy as np
from scipy import signal
from collections import deque
import time

FS          = 200
WIN_SAMPLES = 150

GESTURE_NAMES = {
    0: 'rest', 1: 'fist', 2: 'open_hand',
    3: 'wave_in', 4: 'wave_out', 5: 'pinch',
}


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
            nn.Linear(128, 6)
        )
    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(x)


DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model  = EMG_CNN_LSTM().to(DEVICE)
model.load_state_dict(torch.load('best_model_v4.pt', map_location=DEVICE))
model.eval()
print(f" Model loaded")

nyq    = FS / 2
b,  a  = signal.butter(4, [20/nyq, 90/nyq], btype='band')
bn, an = signal.iirnotch(50, Q=30, fs=FS)


class State:
    emg_buffer   = deque(maxlen=WIN_SAMPLES)
    raw_for_norm = deque(maxlen=400)
    counter      = 0

STATE = State()


def predict_full():
    if len(STATE.emg_buffer) < WIN_SAMPLES:
        return None

    window = np.array(STATE.emg_buffer, dtype=np.float32)
    window = signal.filtfilt(b,  a,  window, axis=0)
    window = signal.filtfilt(bn, an, window, axis=0)

    ref  = np.array(STATE.raw_for_norm, dtype=np.float32)
    mean = ref.mean(axis=0)
    std  = np.where(ref.std(axis=0) < 1e-8, 1e-8, ref.std(axis=0))
    window = (window - mean) / std

    window = window.T.copy()
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()

    return probs


class DebugClassifier(myo.MyoClient):
    async def on_emg_data(self, emg: myo.EMGData):
        for sample in [emg.sample1, emg.sample2]:
            STATE.emg_buffer.append(list(sample))
            STATE.raw_for_norm.append(list(sample))

        STATE.counter += 1
        if STATE.counter % 40 != 0:   # كل 200ms تقريباً
            return

        probs = predict_full()
        if probs is None:
            return

        # اطبع كل الاحتمالات
        line = "  ".join(
            f"{GESTURE_NAMES[i]}:{probs[i]*100:5.1f}%" for i in range(6)
        )
        winner = GESTURE_NAMES[probs.argmax()]
        print(f"\r  → {winner:<10} | {line}   ", end='', flush=True)

    async def on_imu_data(self, _): pass
    async def on_classifier_event(self, _): pass
    async def on_aggregated_data(self, _): pass
    async def on_emg_data_aggregated(self, _): pass
    async def on_fv_data(self, _): pass
    async def on_motion_event(self, _): pass


async def main():
    print("🔍 Scanning...")
    client = await DebugClassifier.with_device()
    print(f" Connected: {client.device.name}\n")
    print("جرب كل حركة وراقب الأرقام كاملة\n")

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
        await client.stop()
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())