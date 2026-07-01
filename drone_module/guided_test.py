# guided_test.py — uses fixed normalization from training

import asyncio
import myo
from myo import ClassifierMode, EMGMode, IMUMode
import torch
import torch.nn as nn
import numpy as np
from scipy import signal
from collections import deque, Counter
import time
import json

FS          = 200
WIN_SAMPLES = 150

GESTURE_NAMES = {
    0: 'rest', 1: 'fist', 2: 'open_hand',
    3: 'wave_in', 4: 'wave_out', 5: 'pinch',
}

TEST_SEQUENCE = [
    'rest', 'wave_in', 'rest', 'wave_out', 'rest',
    'fist', 'rest', 'open_hand', 'rest', 'pinch',
    'rest', 'wave_out', 'rest', 'wave_in', 'rest',
    'pinch', 'rest', 'fist', 'rest', 'open_hand',
]

HOLD_SECONDS       = 5
COUNTDOWN_SECONDS  = 3


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

# ── تطبيع ثابت من التدريب — نفس الأرقام بالضبط ──
NORM_MEAN = np.load('norm_mean.npy')
NORM_STD  = np.load('norm_std.npy')
print(f" Model + fixed normalization loaded")

nyq    = FS / 2
b,  a  = signal.butter(4, [20/nyq, 90/nyq], btype='band')
bn, an = signal.iirnotch(50, Q=30, fs=FS)


class State:
    emg_buffer       = deque(maxlen=WIN_SAMPLES)
    active           = True
    current_truth    = None
    predictions_log  = []
    is_recording     = False

STATE = State()


def predict():
    if len(STATE.emg_buffer) < WIN_SAMPLES:
        return None, 0.0

    window = np.array(STATE.emg_buffer, dtype=np.float32)
    window = signal.filtfilt(b,  a,  window, axis=0)
    window = signal.filtfilt(bn, an, window, axis=0)

    # ── نفس التطبيع الثابت المستخدم في التدريب ──
    window = (window - NORM_MEAN) / NORM_STD

    window = window.T.copy()
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs      = torch.softmax(model(x), dim=1)[0]
        confidence = probs.max().item()
        pred_label = probs.argmax().item()

    return pred_label, confidence


class TestClassifier(myo.MyoClient):

    async def on_emg_data(self, emg: myo.EMGData):
        for sample in [emg.sample1, emg.sample2]:
            STATE.emg_buffer.append(list(sample))

        if not STATE.is_recording:
            return

        pred_label, confidence = predict()
        if pred_label is None:
            return

        STATE.predictions_log.append({
            'truth': STATE.current_truth,
            'pred': GESTURE_NAMES[pred_label],
            'confidence': confidence,
        })

    async def on_imu_data(self, _): pass
    async def on_classifier_event(self, _): pass
    async def on_aggregated_data(self, _): pass
    async def on_emg_data_aggregated(self, _): pass
    async def on_fv_data(self, _): pass
    async def on_motion_event(self, _): pass


async def countdown(seconds, message):
    for i in range(seconds, 0, -1):
        print(f"\r  ⏳  {message}  —  {i}s   ", end='', flush=True)
        await asyncio.sleep(1)
    print(f"\r    GO!                              ")


async def run_test():
    print("\n" + "═" * 56)
    print("  GUIDED REAL-TIME ACCURACY TEST  (fixed normalization)")
    print("═" * 56)
    print(f"\n  Sequence: {len(TEST_SEQUENCE)} gestures")
    print(f"  Hold time: {HOLD_SECONDS}s each\n")
    print("  Get ready — starting in 5 seconds...")
    await asyncio.sleep(5)

    all_results = []

    for idx, gesture in enumerate(TEST_SEQUENCE, 1):
        print("\n" + "─" * 56)
        print(f"  [{idx}/{len(TEST_SEQUENCE)}]")

        await countdown(COUNTDOWN_SECONDS, f"Prepare for  {gesture.upper()}")
        print(f"    DO  {gesture.upper()}  NOW  —  hold it steady!\n")

        STATE.current_truth   = gesture
        STATE.predictions_log = []
        STATE.is_recording    = True

        start = time.time()
        last_shown = None
        while time.time() - start < HOLD_SECONDS:
            await asyncio.sleep(0.1)
            if STATE.predictions_log:
                latest = STATE.predictions_log[-1]
                if latest['pred'] != last_shown:
                    correct = "" if latest['pred'] == gesture else "❌"
                    print(f"      {correct}  model says: {latest['pred']:<10} "
                          f"(conf: {latest['confidence']:.0%})")
                    last_shown = latest['pred']

        STATE.is_recording = False

        preds = [p['pred'] for p in STATE.predictions_log]
        if preds:
            correct_count = sum(1 for p in preds if p == gesture)
            acc = correct_count / len(preds) * 100
            most_common = Counter(preds).most_common(3)
            print(f"\n      📊  Accuracy this round: {acc:.0f}%  "
                  f"({correct_count}/{len(preds)} predictions)")
            print(f"      📊  Distribution: {most_common}")

        all_results.append({'gesture': gesture, 'predictions': preds})

    print("\n" + "═" * 56)
    print("  FINAL SUMMARY")
    print("═" * 56)

    gesture_stats = {}
    for r in all_results:
        g = r['gesture']
        if g not in gesture_stats:
            gesture_stats[g] = {'correct': 0, 'total': 0, 'confusions': []}
        for p in r['predictions']:
            gesture_stats[g]['total'] += 1
            if p == g:
                gesture_stats[g]['correct'] += 1
            else:
                gesture_stats[g]['confusions'].append(p)

    print(f"\n  {'Gesture':<12} {'Accuracy':<12} {'Most Confused With'}")
    print("  " + "─" * 50)
    for g, stats in gesture_stats.items():
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] else 0
        confusion = Counter(stats['confusions']).most_common(1)
        confusion_str = f"{confusion[0][0]} ({confusion[0][1]}x)" if confusion else "—"
        print(f"  {g:<12} {acc:>5.0f}%       {confusion_str}")

    with open('test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  💾  Full results saved to test_results.json")
    print("═" * 56)

    STATE.active = False


async def main():
    print("🔍 Scanning for Myo Armband...")
    client = await TestClassifier.with_device()
    print(f" Connected: {client.device.name}")

    await client.setup(
        classifier_mode=ClassifierMode.DISABLED,
        emg_mode=EMGMode.SEND_EMG,
        imu_mode=IMUMode.SEND_DATA,
    )
    await client.start()

    await run_test()

    await client.stop()
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())