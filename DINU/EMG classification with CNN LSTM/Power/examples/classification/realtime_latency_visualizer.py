import os, pickle, time, threading, queue
import numpy as np
import myo
from myo_ecn.listeners import Buffer, ConnectionChecker
from tensorflow.keras.models import load_model
import tensorflow as tf
from collections import deque, defaultdict
import serial
import tkinter as tk
from tkinter import ttk

# ---------------- Arduino Setup ----------------
ARDUINO_PORT = 'COM3'
BAUDRATE = 9600
arduino = None
arduino_status = "⚠️ Not Connected"
try:
    arduino = serial.Serial(ARDUINO_PORT, BAUDRATE, timeout=0.05)
    arduino_status = "✅ Connected"
except Exception:
    pass

# ---------------- Servo Config ----------------
servo_config = {
    'thumb':  {'pin': 3,  'relax': 180, 'close': 100},
    'index':  {'pin': 5,  'relax': 180, 'close': 70},
    'middle': {'pin': 6,  'relax': 180, 'close': 55},
    'ring':   {'pin': 9,  'relax': 180, 'close': 80},
    'little': {'pin': 12, 'relax': 0,   'close': 100},
    'wrist':  {'pin': 11, 'relax': 90,  'left': 180, 'right': 0},
}

gesture_to_angles = {
    'fist':   {f: cfg['close'] for f, cfg in servo_config.items() if f != 'wrist'},
    'open':   {f: cfg['relax'] for f, cfg in servo_config.items() if f != 'wrist'},
    'thumb':  {'thumb': servo_config['thumb']['close']},
    'index':  {'index': servo_config['index']['close']},
    'middle': {'middle': servo_config['middle']['close']},
    'ring':   {'ring': servo_config['ring']['close']},
    'little': {'little': servo_config['little']['close']},
    'ok': {
        'thumb': int((servo_config['thumb']['relax'] + servo_config['thumb']['close']) / 3),
        'index': int((servo_config['index']['relax'] + servo_config['index']['close']) / 2.5)
    },
    'relax':  {f: cfg['relax'] for f, cfg in servo_config.items() if f != 'wrist'},
    'wrist_left':  {},
    'wrist_right': {},
}

gesture_emojis = {
    'fist': '✊', 'open': '🖐️', 'thumb': '👍', 'index': '☝️', 'middle': '🖕',
    'ring': '💍', 'little': '🤙', 'ok': '👌', 'relax': '😌',
    'wrist_left': '↩️', 'wrist_right': '↪️',
}

# ---------------- Normalization ----------------
def normalize_emg(emg, mean, std): 
    return (emg - mean) / (std + 1e-8)

# ---------------- Positional encoding ----------------
def add_pos_encoding(x, d_model=128):
    pos = tf.range(tf.shape(x)[1], dtype=tf.float32)
    i = tf.range(d_model, dtype=tf.float32)
    angle_rates = 1 / tf.pow(10000.0, (2 * (i//2)) / tf.cast(d_model, tf.float32))
    angles = tf.expand_dims(pos, 1) * tf.expand_dims(angle_rates, 0)
    sines = tf.sin(angles[:, 0::2]); coses = tf.cos(angles[:, 1::2])
    pe = tf.concat([sines, coses], axis=-1)
    return x + tf.expand_dims(pe, 0)

# ---------------- Latency accounting ----------------
cmd_counter = 0
send_times = {}
id_to_gesture = {}
arduino_ready = threading.Event()
arduino_ready.clear()

class Stats:
    def __init__(self, window=10):
        self.count = 0
        self.sum = 0.0
        self.win = deque(maxlen=window)
        self.last_value = 0.0
    def update(self, v):
        self.count += 1
        self.sum += v
        self.win.append(v)
        self.last_value = v
    @property
    def total_avg(self):
        return (self.sum / self.count) if self.count else 0.0

gesture_stats = defaultdict(Stats)

# ---------------- Serial reader ----------------
def serial_reader(incoming_q, status_cb):
    if not arduino: return
    start = time.monotonic()
    while True:
        try:
            b = arduino.readline()
            if not b:
                if time.monotonic() - start > 3.0:
                    try: arduino.write(b"PING\n")
                    except Exception: pass
                    start = time.monotonic()
                continue
            line = b.decode(errors='ignore').strip()
            if line == "READY":
                status_cb("Arduino READY")
                arduino_ready.set()
                break
        except Exception:
            break

    while True:
        try:
            line = arduino.readline()
            if not line:
                continue
            text = line.decode(errors='ignore').strip()
            if text.startswith("DONE,"):
                try:
                    cid = int(text.split(",")[1])
                    incoming_q.put(("DONE", cid, time.monotonic()))
                except Exception:
                    continue
        except Exception:
            break

# ---------------- Command sending ----------------
def make_command(gesture_name):
    wrist_pos = servo_config['wrist']['relax']
    if gesture_name == 'wrist_left':
        wrist_pos = servo_config['wrist']['left']
    elif gesture_name == 'wrist_right':
        wrist_pos = servo_config['wrist']['right']

    angles = gesture_to_angles.get(gesture_name,
        {f: cfg['relax'] for f, cfg in servo_config.items() if f != 'wrist'})
    T = angles.get('thumb',  servo_config['thumb']['relax'])
    I = angles.get('index',  servo_config['index']['relax'])
    M = angles.get('middle', servo_config['middle']['relax'])
    R = angles.get('ring',   servo_config['ring']['relax'])
    L = angles.get('little', servo_config['little']['relax'])
    return T, I, M, R, L, wrist_pos

def send_gesture_command(gesture_name):
    global cmd_counter
    if not arduino or not arduino_ready.is_set():
        return None
    cmd_counter += 1
    cid = cmd_counter
    T,I,M,R,L,W = make_command(gesture_name)
    msg = f"C{cid};T{T},I{I},M{M},R{R},L{L},W{W}\n"
    try:
        arduino.write(msg.encode())
        send_times[cid] = time.monotonic()
        id_to_gesture[cid] = gesture_name
        arduino_ready.clear()
        return cid
    except Exception:
        return None

# ---------------- Inference thread ----------------
def inference_loop(gui_cb, status_cb, stop_ev):
    myo.init(sdk_path=r'C:\\Users\\ThinkCentre\\Desktop\\Power\\myo_sdk')
    myo_status = "✅ Connected" if ConnectionChecker().ok else "⚠️ Not Connected"
    status_cb(f"Myo: {myo_status} | Arduino: {arduino_status}")

    MODEL_DIR = "models"
    model = load_model(os.path.join(MODEL_DIR, "trained_model.h5"),
                       custom_objects={'add_pos_encoding': add_pos_encoding})
    with open(os.path.join(MODEL_DIR, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    label_encoder = metadata.get("label_encoder")
    gestures = metadata['gestures']
    mean, std = metadata.get('mean'), metadata.get('std')
    winlen = metadata['winlen']

    confidence_threshold = 0.9
    vote_window = 7
    pred_history = deque(maxlen=vote_window)

    buf = Buffer(winlen)
    hub = myo.Hub()
    last_sent_gesture = None

    status_cb("Starting Myo stream…")
    with hub.run_in_background(buf.on_event):
        while not stop_ev.is_set():
            time.sleep(0.01)
            if len(buf.emg_data_queue) < winlen:
                continue
            emg = np.array([x[1] for x in buf.get_emg_data()])
            if mean is not None and std is not None:
                emg = normalize_emg(emg, mean, std)
            X = emg.reshape(1, emg.shape[0], emg.shape[1])
            preds = model.predict(X, verbose=0)
            pred_prob = float(np.max(preds))
            if pred_prob < confidence_threshold:
                continue
            pred_class = int(np.argmax(preds))
            pred_history.append((pred_class, pred_prob))
            scores = {}
            for c, p in pred_history:
                scores[c] = scores.get(c, 0.0) + p
            final_cls = max(scores, key=scores.get)
            gesture_name = label_encoder.inverse_transform([final_cls])[0] if label_encoder else gestures[final_cls]
            if arduino_ready.is_set() and gesture_name != last_sent_gesture:
                if send_gesture_command(gesture_name) is not None:
                    gui_cb(gesture_name, pred_prob)
                    last_sent_gesture = gesture_name

# ---------------- GUI ----------------
def start_gui():
    root = tk.Tk()
    root.title("🦾 Real-time Latency Visualizer — Myo → Servo")
    root.attributes("-fullscreen", True)
    root.configure(bg="#121212")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TLabel", background="#121212", foreground="#E6F3FF", font=("Segoe UI", 12))
    style.configure("Big.TLabel", background="#121212", foreground="#FFFFFF", font=("Segoe UI", 20, "bold"))
    style.configure("Hdr.TLabel", background="#121212", foreground="#66D9EF", font=("Segoe UI", 14, "bold"))

    ttk.Label(root, text="🦾 EMG → Servo Latency (Numerical)", style="Hdr.TLabel").pack(pady=(8, 2))
    status_lbl = ttk.Label(root, text="Initializing…"); status_lbl.pack()

    # Current gesture / latency
    top_frame = ttk.Frame(root); top_frame.pack(pady=6)
    current_gesture = tk.StringVar(value="—")
    last_latency = tk.StringVar(value="0.000 s")
    ttk.Label(top_frame, text="Current gesture:", style="Hdr.TLabel").grid(row=0, column=0, padx=10)
    ttk.Label(top_frame, textvariable=current_gesture, style="Big.TLabel").grid(row=0, column=1, padx=10)
    ttk.Label(top_frame, text="Last latency:", style="Hdr.TLabel").grid(row=0, column=2, padx=20)
    ttk.Label(top_frame, textvariable=last_latency, style="Big.TLabel").grid(row=0, column=3, padx=10)

    # Table
    cols = ("gesture", "trials", "total_avg", "last_latency")
    tree = ttk.Treeview(root, columns=cols, show="headings", height=18)
    for c, w in zip(cols, (18, 10, 16, 16)):
        tree.heading(c, text=c.upper())
        tree.column(c, width=w*10, anchor="center")
    tree.pack(fill="x", padx=20, pady=8)
    for g in sorted(gesture_to_angles.keys()):
        tree.insert("", "end", iid=g, values=(g, 0, "0.000", "0.000"))

    # Logging setup
    os.makedirs("Latency", exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join("Latency", f"Latency_{timestamp}.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write("=== EMG → SERVO LATENCY SESSION LOG ===\n")
    log_file.write(f"Session start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Arduino Port: {ARDUINO_PORT} @ {BAUDRATE} baud\n")
    log_file.write(f"Model: models/trained_model.h5\n")
    log_file.write("-------------------------------------------\n\n")
    log_file.flush()

    def write_log(gesture, lat, s):
        log_file.write(
            f"[{time.strftime('%H:%M:%S')}] Gesture: {gesture:<12} | "
            f"Latency: {lat:.3f} s | Avg: {s.total_avg:.3f} s | Trial #: {s.count}\n"
        )
        log_file.flush()

    # Stop
    stop_flag = threading.Event()
    def stop():
        stop_flag.set()
        log_file.write("\n=== SESSION CLOSED ===\n"); log_file.close()
        root.destroy()
    ttk.Button(root, text="🟥 Stop & Exit", command=stop).pack(pady=10)

    # Threads
    incoming_q = queue.Queue()
    def status_cb(msg): status_lbl.config(text=msg)
    threading.Thread(target=serial_reader, args=(incoming_q, status_cb), daemon=True).start()

    def gui_cb(gesture_name, conf):
        emoji = gesture_emojis.get(gesture_name, "❓")
        current_gesture.set(f"{emoji} {gesture_name}")

    threading.Thread(target=inference_loop,
                     args=(gui_cb, status_cb, stop_flag),
                     daemon=True).start()

    # Poll latency
    def poll_incoming():
        try:
            while True:
                kind, cid, t_recv = incoming_q.get_nowait()
                if kind == "DONE":
                    t0 = send_times.pop(cid, None)
                    g = id_to_gesture.pop(cid, "unknown")
                    if t0 is not None:
                        lat = t_recv - t0
                        if g not in gesture_stats:
                            gesture_stats[g] = Stats()
                            if not tree.exists(g):
                                tree.insert("", "end", iid=g, values=(g, 0, "0.000", "0.000"))
                        s = gesture_stats[g]
                        s.update(lat)
                        last_latency.set(f"{lat:.3f} s")
                        tree.item(g, values=(g, s.count, f"{s.total_avg:.3f}", f"{s.last_value:.3f}"))
                        write_log(g, lat, s)
                    arduino_ready.set()
        except queue.Empty:
            pass
        if not stop_flag.is_set():
            root.after(30, poll_incoming)
    poll_incoming()
    root.protocol("WM_DELETE_WINDOW", stop)
    root.mainloop()

# --------------- Entry ---------------
if __name__ == "__main__":
    start_gui()
