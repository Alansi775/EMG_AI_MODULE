# realtime.py — uses fixed normalization from training (matches exactly)

import asyncio
import myo
from myo import ClassifierMode, EMGMode, IMUMode
import torch
import torch.nn as nn
import numpy as np
from scipy import signal
from collections import deque, Counter
import time

FS           = 200
WIN_SAMPLES  = 150
N_CHANNELS   = 8
N_CLASSES    = 6
CONFIDENCE_THRESHOLD = 0.70
RAW_VOTE_HISTORY      = 5
CONFIRM_VOTES_NEEDED  = 4
STABILITY_ROUNDS      = 2

GESTURE_NAMES = {
    0: 'rest', 1: 'fist', 2: 'open_hand',
    3: 'wave_in', 4: 'wave_out', 5: 'pinch',
}
GESTURE_EMOJI = {
    0: '', 1: '', 2: '',
    3: '', 4: '', 5: '',
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
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(x)


DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model  = EMG_CNN_LSTM(n_classes=N_CLASSES).to(DEVICE)
model.load_state_dict(torch.load('best_model_v4.pt', map_location=DEVICE))
model.eval()
print(f" Model loaded — Device: {DEVICE}")

# ── تحميل نفس إحصائيات التطبيع من التدريب — الإصلاح الأساسي ──
NORM_MEAN = np.load('norm_mean.npy')
NORM_STD  = np.load('norm_std.npy')
print(f" Normalization stats loaded (fixed, matches training)")

nyq    = FS / 2
b,  a  = signal.butter(4, [20/nyq, 90/nyq], btype='band')
bn, an = signal.iirnotch(50, Q=30, fs=FS)


class State:
    emg_buffer        = deque(maxlen=WIN_SAMPLES)
    pred_history       = deque(maxlen=RAW_VOTE_HISTORY)

    confirmed_gesture  = 0
    candidate_gesture  = None
    candidate_count    = 0

    last_print_time    = 0
    prediction_count   = 0

STATE = State()


def predict():
    if len(STATE.emg_buffer) < WIN_SAMPLES:
        return None, 0.0

    window = np.array(STATE.emg_buffer, dtype=np.float32)
    window = signal.filtfilt(b,  a,  window, axis=0)
    window = signal.filtfilt(bn, an, window, axis=0)

    # ── تطبيع ثابت — نفس الأرقام بالضبط اللي استُخدمت في التدريب ──
    window = (window - NORM_MEAN) / NORM_STD

    window = window.T.copy()
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits     = model(x)
        probs      = torch.softmax(logits, dim=1)[0]
        confidence = probs.max().item()
        pred_label = probs.argmax().item()

    return pred_label, confidence


def send_to_actuator(gesture_label):
    name = GESTURE_NAMES[gesture_label]
    print(f"\n   ACTUATOR COMMAND → {name.upper()}\n")
    # هنا تحط كود التحكم بالموتور/الدرون لاحقاً


class RealtimeClassifier(myo.MyoClient):

    async def on_emg_data(self, emg: myo.EMGData):
        for sample in [emg.sample1, emg.sample2]:
            STATE.emg_buffer.append(list(sample))

        STATE.prediction_count += 1
        if STATE.prediction_count % 10 != 0:
            return

        pred_label, confidence = predict()
        if pred_label is None or confidence < CONFIDENCE_THRESHOLD:
            return

        STATE.pred_history.append(pred_label)
        if len(STATE.pred_history) < STATE.pred_history.maxlen:
            return

        vote_counts = Counter(STATE.pred_history)
        top_label, top_count = vote_counts.most_common(1)[0]

        if top_count < CONFIRM_VOTES_NEEDED:
            return

        if top_label == STATE.confirmed_gesture:
            STATE.candidate_gesture = None
            STATE.candidate_count   = 0
        elif top_label == STATE.candidate_gesture:
            STATE.candidate_count += 1
        else:
            STATE.candidate_gesture = top_label
            STATE.candidate_count   = 1

        if STATE.candidate_count >= STABILITY_ROUNDS:
            STATE.confirmed_gesture = STATE.candidate_gesture
            STATE.candidate_gesture = None
            STATE.candidate_count   = 0
            send_to_actuator(STATE.confirmed_gesture)

        now = time.time()
        if (now - STATE.last_print_time) > 0.3:
            emoji = GESTURE_EMOJI[STATE.confirmed_gesture]
            name  = GESTURE_NAMES[STATE.confirmed_gesture]
            cand  = (f" (trying: {GESTURE_NAMES[STATE.candidate_gesture]} "
                     f"{STATE.candidate_count}/{STABILITY_ROUNDS})"
                     if STATE.candidate_gesture is not None else "")
            print(f"\r  {emoji}  STATE: {name:<12} conf:{confidence:.0%}{cand}        ",
                  end='', flush=True)
            STATE.last_print_time = now

    async def on_imu_data(self, _): pass
    async def on_classifier_event(self, _): pass
    async def on_aggregated_data(self, _): pass
    async def on_emg_data_aggregated(self, _): pass
    async def on_fv_data(self, _): pass
    async def on_motion_event(self, _): pass


async def main():
    print("\n" + "═" * 52)
    print("  REAL-TIME EMG — FIXED NORMALIZATION")
    print("═" * 52)
    print(f"\n  Window           : {WIN_SAMPLES} samples (~{WIN_SAMPLES/FS*1000:.0f}ms)")
    print(f"  Confidence min   : {CONFIDENCE_THRESHOLD:.0%}")
    print(f"  Raw vote history : {RAW_VOTE_HISTORY}  (need {CONFIRM_VOTES_NEEDED}+ agreeing)")
    print(f"  Stability rounds : {STABILITY_ROUNDS}  (before actuator commits)\n")
    print("  🔍  Scanning for Myo Armband...")

    client = await RealtimeClassifier.with_device()
    print(f"    Connected: {client.device.name}\n")
    print("  Try the gestures! Press Ctrl+C to stop.\n")
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