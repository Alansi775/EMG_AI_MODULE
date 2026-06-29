# guided_collector_v3.py
# Correct flow: record gesture → STOP → rest countdown → record rest → STOP → gesture countdown → record

import asyncio
import myo
from myo import ClassifierMode, EMGMode, IMUMode
import pandas as pd
import time
import json
import os
from datetime import datetime

# ── Configuration ──
FS_EMG = 200

GESTURES = [
    ('fist',       1, 'Close your hand into a tight FIST and hold it'),
    ('open_hand',  2, 'Open your hand fully — spread all fingers wide'),
    ('wave_in',    3, 'Bend your wrist INWARD toward your body and hold'),
    ('wave_out',   4, 'Bend your wrist OUTWARD away from your body and hold'),
    ('pinch',      5, 'Pinch index finger and thumb together — hold it'),
]

REPETITIONS        = 6   # rounds per gesture
GESTURE_HOLD_SEC   = 5   # seconds to RECORD each gesture
REST_HOLD_SEC      = 5   # seconds to RECORD rest after each gesture
REST_COUNTDOWN_SEC = 8   # seconds to relax BEFORE recording rest (not recorded)
NEXT_COUNTDOWN_SEC = 4   # seconds to prepare BEFORE recording next gesture (not recorded)

SESSION_DIR = f"sessions/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(SESSION_DIR, exist_ok=True)


# ── Shared state ──
class State:
    emg_buffer    = []
    label_buffer  = []
    time_buffer   = []
    current_label = -1   # -1 = NOT recording
    active        = True
    sample_count  = 0

STATE = State()


# ── Myo Collector ──
class EMGCollector(myo.MyoClient):

    async def on_emg_data(self, emg: myo.EMGData):
        if not STATE.active or STATE.current_label < 0:
            return
        ts = time.time()
        for sample in [emg.sample1, emg.sample2]:
            STATE.emg_buffer.append(list(sample))
            STATE.label_buffer.append(STATE.current_label)
            STATE.time_buffer.append(ts)
            STATE.sample_count += 1

    async def on_imu_data(self, _): pass
    async def on_classifier_event(self, _): pass
    async def on_aggregated_data(self, _): pass
    async def on_emg_data_aggregated(self, _): pass
    async def on_fv_data(self, _): pass
    async def on_motion_event(self, _): pass


# ── UI helpers ──
def divider():
    print('\n' + '═' * 54)

def header(text):
    divider()
    print(f"  {text}")
    divider()

async def silent_countdown(seconds, message):
    """Count down without recording — just display"""
    STATE.current_label = -1  # make sure nothing is recorded
    print()
    for i in range(seconds, 0, -1):
        print(f"\r  ⏳  {message}  —  {i}s   ", end='', flush=True)
        await asyncio.sleep(1)
    print(f"\r  ✅  {message}  —  GO!                    ")
    await asyncio.sleep(0.1)

async def record_bar(seconds, label_name, label_id):
    """Show progress bar and record data"""
    STATE.current_label = label_id
    steps     = seconds * 10
    bar_width = 32
    samples_before = STATE.sample_count

    for step in range(steps + 1):
        filled  = int(bar_width * step / steps) if steps > 0 else bar_width
        bar     = '█' * filled + '░' * (bar_width - filled)
        elapsed = step * 0.1
        captured = STATE.sample_count - samples_before
        print(f"\r  🔴  [{bar}] {elapsed:.1f}s  |  {captured} samples  |  {label_name}   ",
              end='', flush=True)
        if step < steps:
            await asyncio.sleep(0.1)

    STATE.current_label = -1  # STOP recording immediately
    captured = STATE.sample_count - samples_before
    print(f"\n  ✔   Recorded {captured} samples  ({captured / FS_EMG:.1f}s)  —  '{label_name}'")
    return captured


# ── Guided Session ──
async def run_session():

    header("EMG GESTURE COLLECTION  v3  —  Myo Armband")
    print(f"\n  Session     :  {SESSION_DIR}")
    print(f"  Gestures    :  {len(GESTURES)} gestures  +  rest")
    print(f"  Rounds      :  {REPETITIONS}")
    print(f"  Per gesture :  {GESTURE_HOLD_SEC}s recorded  |  {REST_HOLD_SEC}s rest recorded")
    print(f"  Countdown   :  {REST_COUNTDOWN_SEC}s rest-prep  |  {NEXT_COUNTDOWN_SEC}s gesture-prep  (not recorded)")
    divider()

    print("\n  Put on the Myo on your RIGHT forearm.")
    print("  Logo facing UP.  Sensor tight against your skin.")
    print("  Rest your arm on the table, palm facing DOWN.\n")
    print("  Starting in 5 seconds — get comfortable...")
    await asyncio.sleep(5)

    for rep in range(1, REPETITIONS + 1):
        header(f"ROUND  {rep}  /  {REPETITIONS}")

        for gesture_name, gesture_label, instruction in GESTURES:

            # ── Countdown to gesture (NOT recorded) ──
            print(f"\n  📋  {instruction}")
            await silent_countdown(
                NEXT_COUNTDOWN_SEC,
                f"Prepare for  {gesture_name.upper().replace('_', ' ')}"
            )

            # ── Record gesture ──
            print(f"  👊  HOLD that gesture now — recording starts!")
            await record_bar(GESTURE_HOLD_SEC, gesture_name.replace('_', ' '), gesture_label)

            # ── Rest countdown (NOT recorded — give time to actually relax) ──
            await silent_countdown(
                REST_COUNTDOWN_SEC,
                "Relax your hand — place it flat on the table, palm down"
            )

            # ── Record rest ──
            print(f"  🖐   Keep your hand STILL and relaxed — recording rest now")
            await record_bar(REST_HOLD_SEC, 'rest', 0)

        print(f"\n  ✅  Round {rep} complete!")
        if rep < REPETITIONS:
            print("  💆  Shake out your hand gently — 4 second break...")
            await asyncio.sleep(4)

    # ── Final rest ──
    header("FINAL REST")
    print("  Keep your hand relaxed on the table...")
    await record_bar(REST_HOLD_SEC, 'rest', 0)

    STATE.active = False
    header("ALL DONE  🎉  Excellent work!")


# ── Save ──
def save():
    if not STATE.emg_buffer:
        print("  No data collected.")
        return None

    cols = [f'emg_{i}' for i in range(8)]
    df   = pd.DataFrame(STATE.emg_buffer, columns=cols)
    df['timestamp'] = STATE.time_buffer
    df['label']     = STATE.label_buffer

    g_names = {0: 'rest'}
    g_names.update({label: name for name, label, _ in GESTURES})
    df['gesture'] = df['label'].map(g_names)

    path = f"{SESSION_DIR}/emg_data.csv"
    df.to_csv(path, index=False)

    meta = {
        'session'      : SESSION_DIR,
        'date'         : datetime.now().isoformat(),
        'total_samples': len(df),
        'duration_sec' : round(len(df) / FS_EMG, 1),
        'fs_emg'       : FS_EMG,
        'repetitions'  : REPETITIONS,
        'per_label'    : df['label'].value_counts().sort_index().to_dict(),
    }
    with open(f"{SESSION_DIR}/meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    header("SESSION SAVED")
    print(f"  File     :  {path}")
    print(f"  Samples  :  {len(df):,}   ({meta['duration_sec']}s total)\n")
    print("  Breakdown:")

    for label, count in sorted(meta['per_label'].items()):
        name = g_names.get(int(label), f'label_{label}')
        secs = round(count / FS_EMG, 1)
        pct  = count / len(df) * 100
        bar  = '█' * min(int(pct / 2), 30)
        print(f"    {name:14s} [{label}]  {count:5d} smp  {secs:5.1f}s  {pct:4.1f}%  {bar}")
    divider()
    return df


# ── Entry point ──
async def main():
    print("\n  🔍  Scanning for Myo Armband...")
    client = await EMGCollector.with_device()
    print(f"  ✅  Connected:  {client.device.name}  ({client.device.address})")

    await client.setup(
        classifier_mode=ClassifierMode.DISABLED,
        emg_mode=EMGMode.SEND_EMG,
        imu_mode=IMUMode.SEND_DATA,
    )
    await client.start()

    await run_session()

    await client.stop()
    await client.disconnect()

    save()


if __name__ == "__main__":
    asyncio.run(main())