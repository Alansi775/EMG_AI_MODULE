import pickle, time, sys
import numpy as np
import myo
from myo_ecn.listeners import Buffer, ConnectionChecker
from tensorflow.keras.models import load_model
from collections import deque
import serial

# ================== Arduino Setup ==================
ARDUINO_PORT = 'COM3'  # Change if needed
BAUDRATE = 9600

try:
    arduino = serial.Serial(ARDUINO_PORT, BAUDRATE, timeout=1)
    print(f"✅ Connected to Arduino on {ARDUINO_PORT}")
except Exception as e:
    print(f"⚠️ Could not connect to Arduino: {e}")
    arduino = None

# ================== Servo configuration ==================
servo_config = {
    'thumb':  {'pin': 3,  'relax': 180, 'close': 100},
    'index':  {'pin': 5,  'relax': 180, 'close': 70},
    'middle': {'pin': 6,  'relax': 180, 'close': 55},
    'ring':   {'pin': 9,  'relax': 180, 'close': 80},
    'little': {'pin': 12, 'relax': 0,   'close': 100},
}

# ================== Gesture → Servo mapping ==================
gesture_to_angles = {
    'fist':   {f: cfg['close'] for f, cfg in servo_config.items()},
    'open':   {f: cfg['relax'] for f, cfg in servo_config.items()},
    'thumb':  {'thumb': servo_config['thumb']['close']},
    'index':  {'index': servo_config['index']['close']},
    'middle': {'middle': servo_config['middle']['close']},
    'ring':   {'ring': servo_config['ring']['close']},
    'little': {'little': servo_config['little']['close']},
    'relax':  {f: cfg['relax'] for f, cfg in servo_config.items()},
}

# ================== Normalization function ==================
def normalize_emg(emg, mean, std):
    return (emg - mean) / (std + 1e-8)

# ================== Main ==================
def main():
    # ===== Myo setup =====
    myo.init(sdk_path=r'C:\Users\ThinkCentre\Desktop\Power\myo_sdk')
    hub = myo.Hub()
    if not ConnectionChecker().ok:
        print("Myo not connected. Please check Myo Connect.")
        return

    # ===== Load model & metadata =====
    MODEL_FILE = 'models/trained_model.keras'
    METADATA_FILE = 'models/metadata.pkl'

    if len(sys.argv) > 1:
        MODEL_FILE = sys.argv[1]
    if len(sys.argv) > 2:
        METADATA_FILE = sys.argv[2]

    print(f"Loading model from {MODEL_FILE}")
    model = load_model(MODEL_FILE)

    with open(METADATA_FILE, 'rb') as f:
        metadata = pickle.load(f)

    gestures = metadata['gestures']
    label_encoder = metadata.get('label_encoder', None)
    winlen = metadata['winlen']

    mean = metadata.get('mean', None)
    std = metadata.get('std', None)

    # ===== Setup buffer =====
    emg_buffer = Buffer(winlen)

    # ===== Majority voting =====
    vote_window = 7
    pred_history = deque(maxlen=vote_window)
    confidence_threshold = 0.6

    print("You may start performing gestures. Press Ctrl-C to stop.")

    with hub.run_in_background(emg_buffer.on_event):
        while hub.running:
            time.sleep(0.025)

            if len(emg_buffer.emg_data_queue) < winlen:
                continue

            emg = emg_buffer.get_emg_data()
            emg = np.array([x[1] for x in emg])

            if mean is not None and std is not None:
                emg = normalize_emg(emg, mean, std)

            X = emg.reshape(1, emg.shape[0], emg.shape[1])
            preds = model.predict(X, verbose=0)
            pred_prob = np.max(preds, axis=1)[0]
            pred_class = np.argmax(preds, axis=1)[0]

            if pred_prob >= confidence_threshold:
                pred_history.append((pred_class, pred_prob))
            else:
                continue

            # Weighted majority voting
            class_scores = {}
            for cls, prob in pred_history:
                class_scores[cls] = class_scores.get(cls, 0) + prob
            final_pred = max(class_scores, key=class_scores.get)

            if label_encoder:
                gesture_name = label_encoder.inverse_transform([final_pred])[0]
            else:
                gesture_name = gestures[final_pred]

            # Terminal visualization
            print(f"\rRecognized gesture: {gesture_name} (conf: {pred_prob:.2f})", end='')

            # Send angles to Arduino
            if arduino:
                angles = gesture_to_angles.get(gesture_name, {f: cfg['relax'] for f, cfg in servo_config.items()})
                cmd = f"T{angles.get('thumb', servo_config['thumb']['relax'])}," \
                      f"I{angles.get('index', servo_config['index']['relax'])}," \
                      f"M{angles.get('middle', servo_config['middle']['relax'])}," \
                      f"R{angles.get('ring', servo_config['ring']['relax'])}," \
                      f"L{angles.get('little', servo_config['little']['relax'])}\n"
                arduino.write(cmd.encode())

if __name__ == '__main__':
    main()
