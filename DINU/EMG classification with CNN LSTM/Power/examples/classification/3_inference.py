# ===========================================
# 3_inference.py (Dark UI + Fast Servo Response + RMS Graph + Fullscreen)
# ===========================================

import os, pickle, time, threading
import numpy as np
import myo
from myo_ecn.listeners import Buffer, ConnectionChecker
from tensorflow.keras.models import load_model
import tensorflow as tf
from collections import deque
import serial
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ---------------- Arduino Setup ----------------
ARDUINO_PORT = 'COM3'
BAUDRATE = 9600
try:
    arduino = serial.Serial(ARDUINO_PORT, BAUDRATE, timeout=1)
    arduino_status = " Connected"
except Exception:
    arduino = None
    arduino_status = "⚠️ Not Connected"

# ---------------- Servo Config ----------------
servo_config = {
    'thumb':  {'pin': 3,  'relax': 180, 'close': 100},
    'index':  {'pin': 5,  'relax': 180, 'close': 70},
    'middle': {'pin': 6,  'relax': 180, 'close': 55},
    'ring':   {'pin': 9,  'relax': 180, 'close': 80},
    'little': {'pin': 12, 'relax': 0,   'close': 100},
    'wrist':  {'pin': 11, 'relax': 90, 'left': 180, 'right': 0},
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
}

gesture_emojis = {
    'fist': '', 'open': '️', 'thumb': '👍', 'index': '☝️', 'middle': '🖕',
    'ring': '💍', 'little': '🤙', 'ok': '👌',
    'wrist_left': '↩️', 'wrist_right': '↪️',
    'relax': '😌',
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
    sines = tf.sin(angles[:, 0::2])
    coses = tf.cos(angles[:, 1::2])
    pe = tf.concat([sines, coses], axis=-1)
    return x + tf.expand_dims(pe, 0)

# ---------------- Inference Thread ----------------
def inference_thread_with_pause(update_text, emg_buffer, model, label_encoder, metadata,
                                stop_flag, pause_flag, update_history, update_stats):
    gestures = metadata['gestures']
    mean, std = metadata.get('mean'), metadata.get('std')
    confidence_threshold = 0.9
    vote_window = 7
    pred_history = deque(maxlen=vote_window)

    wrist_position = servo_config['wrist']['relax']
    last_cmd = None

    hub = myo.Hub()
    update_text("⚙️ Starting Myo EMG stream...")

    with hub.run_in_background(emg_buffer.on_event):
        while not stop_flag.is_set():
            if pause_flag.is_set():
                time.sleep(0.05)
                continue

            time.sleep(0.01)  # Faster inference cycle
            if len(emg_buffer.emg_data_queue) < metadata['winlen']:
                continue

            emg = np.array([x[1] for x in emg_buffer.get_emg_data()])
            if mean is not None and std is not None:
                emg = normalize_emg(emg, mean, std)

            X = emg.reshape(1, emg.shape[0], emg.shape[1])
            preds = model.predict(X, verbose=0)
            pred_prob = float(np.max(preds))
            pred_class = int(np.argmax(preds))

            if pred_prob < confidence_threshold:
                continue

            pred_history.append((pred_class, pred_prob))
            class_scores = {}
            for cls, prob in pred_history:
                class_scores[cls] = class_scores.get(cls, 0.0) + prob
            final_pred = max(class_scores, key=class_scores.get)
            gesture_name = label_encoder.inverse_transform([final_pred])[0] if label_encoder else gestures[final_pred]
            emoji = gesture_emojis.get(gesture_name, "❓")

            confidence_pct = int(pred_prob * 100)
            bar_blocks = max(0, min((confidence_pct - 80)//2, 10))
            conf_bar = "⬛" * bar_blocks + "⬜" * (10 - bar_blocks)
            update_text(f"{emoji} {gesture_name}  |  {conf_bar}  {confidence_pct}%")

            update_history(gesture_name, confidence_pct)
            update_stats(emg)

            # ---------------- Arduino Control (Instant Response) ----------------
            if arduino:
                if gesture_name == 'wrist_left':
                    wrist_position = servo_config['wrist']['left']
                elif gesture_name == 'wrist_right':
                    wrist_position = servo_config['wrist']['right']
                elif gesture_name in ('open', 'relax', 'ok'):
                    wrist_position = servo_config['wrist']['relax']

                angles = gesture_to_angles.get(
                    gesture_name,
                    {f: cfg['relax'] for f, cfg in servo_config.items() if f != 'wrist'}
                )

                cmd = (
                    f"T{angles.get('thumb',180)},"
                    f"I{angles.get('index',180)},"
                    f"M{angles.get('middle',180)},"
                    f"R{angles.get('ring',180)},"
                    f"L{angles.get('little',0)},"
                    f"W{wrist_position}\n"
                )

                if cmd != last_cmd:
                    try:
                        arduino.write(cmd.encode())
                    except Exception:
                        pass
                    last_cmd = cmd

# ---------------- GUI ----------------
def start_gui():
    myo.init(sdk_path=r'C:\\Users\\ThinkCentre\\Desktop\\Power\\myo_sdk')
    myo_status = " Connected" if ConnectionChecker().ok else "⚠️ Not Connected"

    MODEL_DIR = "models"
    MODEL_FILE = os.path.join(MODEL_DIR, "trained_model.h5")
    META_FILE = os.path.join(MODEL_DIR, "metadata.pkl")

    print(f" Loading model from: {MODEL_FILE}")
    model = load_model(MODEL_FILE, custom_objects={'add_pos_encoding': add_pos_encoding})
    with open(META_FILE, "rb") as f:
        metadata = pickle.load(f)
    label_encoder = metadata.get("label_encoder")
    emg_buffer = Buffer(metadata["winlen"])

    # -------- GUI Setup (Dark Theme) --------
    root = tk.Tk()
    root.title("🦾 EMG Gesture Visualization + Control")
    root.attributes("-fullscreen", True)
    root.configure(bg="#121212")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TFrame", background="#121212")
    style.configure("TLabel", background="#121212", foreground="#00FFFF", font=("Segoe UI", 12))
    style.configure("TButton", background="#1F1F1F", foreground="#00FF99",
                    font=("Segoe UI", 11, "bold"), padding=6, relief="flat")
    style.map("TButton", background=[("active", "#00FF99")], foreground=[("active", "black")])

    # --- MAIN TITLE ---
    ttk.Label(root, text="🦾 EMG Gesture Recognition & Control System",
              font=("Segoe UI", 18, "bold"), foreground="#00FFFF",
              background="#121212").pack(pady=(8, 0))

    # --- Status Frame ---
    status_frame = ttk.Frame(root); status_frame.pack(pady=(2, 4))
    ttk.Label(status_frame, text=f"Myo: {myo_status}").grid(row=0, column=0, padx=15)
    ttk.Label(status_frame, text=f"Arduino: {arduino_status}").grid(row=0, column=1, padx=15)

    # --- Status Label ---
    lbl = ttk.Label(root, text="Press ▶️ Start to begin", font=("Segoe UI", 14, "bold"),
                    foreground="#FFFFFF", background="#121212")
    lbl.pack(pady=10)

    # --- Plot Setup ---
    main_frame = tk.Frame(root, bg="#121212")
    main_frame.pack(fill="both", expand=True, pady=(0, 5))

    fig = Figure(figsize=(10, 5.3), facecolor="#121212")
    axes = [fig.add_subplot(8, 1, i + 1, facecolor="#0A0A0A") for i in range(8)]
    lines = []
    for i, ax in enumerate(axes):
        line, = ax.plot(np.zeros(512), color="#00FF66", linewidth=1.2)
        lines.append(line)
        ax.set_ylim(-128, 128)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ("bottom", "top", "left", "right"):
            ax.spines[sp].set_color("#222")
        ax.set_ylabel(f"CH{i+1}", rotation=0, labelpad=30,
                      fontsize=9, weight="bold", color="#888")

    canvas = FigureCanvasTkAgg(fig, master=main_frame)
    canvas.get_tk_widget().config(bg="#121212", highlightthickness=0)
    canvas.get_tk_widget().pack(side="left", fill="both", expand=True, padx=(70, 5))

    # --- Gesture History ---
    history_frame = tk.Frame(main_frame, bg="#121212")
    history_frame.pack(side="right", fill="y", padx=(5, 30))
    ttk.Label(history_frame, text="Gesture History",
              font=("Segoe UI", 12, "bold"), foreground="#00FFFF",
              background="#121212").pack(pady=(0, 3))

    history_box = tk.Listbox(history_frame, bg="#0A0A0A", fg="#00FFAA",
                             font=("Consolas", 10), borderwidth=0,
                             highlightthickness=0, width=45, height=30)
    history_box.pack(fill="y", expand=True)

    # --- Buttons ---
    btn_frame = ttk.Frame(root); btn_frame.pack(pady=10)
    btn_start = ttk.Button(btn_frame, text="▶️ Start")
    btn_pause = ttk.Button(btn_frame, text="⏸️ Pause")
    btn_resume = ttk.Button(btn_frame, text="🟢 Continue")
    btn_stop = ttk.Button(btn_frame, text="🟥 Stop & Exit")
    btn_start.grid(row=0, column=0, padx=10)
    btn_pause.grid(row=0, column=1, padx=10)
    btn_resume.grid(row=0, column=2, padx=10)
    btn_stop.grid(row=0, column=3, padx=10)

    # --- RMS Graph ---
    fig_stats = Figure(figsize=(12, 2.5), facecolor="#121212")
    ax_stats = fig_stats.add_subplot(111, facecolor="#0A0A0A")
    bars = ax_stats.bar(range(8), np.zeros(8), color="#00CCFF", edgecolor="#00FFFF", linewidth=0.8)
    ax_stats.set_ylim(0, 1)
    ax_stats.set_xticks(range(8))
    ax_stats.set_xticklabels([f"CH{i+1}" for i in range(8)], color="#00FFFF", fontsize=9, weight="bold")
    ax_stats.tick_params(axis="y", colors="#00FFFF", labelsize=9, width=0.8)
    for sp in ("bottom", "top", "left", "right"): ax_stats.spines[sp].set_color("#222")
    stats_canvas = FigureCanvasTkAgg(fig_stats, master=root)
    stats_canvas.get_tk_widget().pack(pady=10)

    ttk.Label(root, text="Created by Dinu Putere 2025",
              font=("Segoe UI", 10, "italic"), foreground="#555",
              background="#121212").pack(side="bottom", pady=5)

    # --- State ---
    stop_flag, pause_flag = threading.Event(), threading.Event()
    buf = np.zeros((8, 512))

    def update_text(msg): lbl.config(text=msg, foreground="#00FFAA")
    def update_history(g, c):
        history_box.insert(0, f"{time.strftime('%H:%M:%S')} → {g:<10} ({c:>3}%)")
        if history_box.size() > 60: history_box.delete(60, 'end')
    def update_stats(emg):
        rms = np.sqrt((emg**2).mean(axis=0))
        for i, b in enumerate(bars): b.set_height(float(abs(rms[i])))
        stats_canvas.draw_idle()
    def update_plot():
        nonlocal buf
        if len(emg_buffer.emg_data_queue) > 0:
            emg_chunk = np.array([x[1] for x in emg_buffer.get_emg_data()]).T
            n_new = emg_chunk.shape[1]
            buf = np.hstack((buf[:, n_new:], emg_chunk)) if n_new < 512 else emg_chunk[:, -512:]
            for ch in range(8): lines[ch].set_ydata(buf[ch])
            canvas.draw_idle()
        if not stop_flag.is_set(): root.after(50, update_plot)

    def start_all():
        stop_flag.clear(); pause_flag.clear()
        threading.Thread(target=inference_thread_with_pause,
                         args=(update_text, emg_buffer, model, label_encoder, metadata,
                               stop_flag, pause_flag, update_history, update_stats),
                         daemon=True).start()
        update_plot()

    def stop_all(): stop_flag.set(); root.destroy()
    def pause_all(): pause_flag.set(); lbl.config(text="⏸️ Paused Inference", foreground="#FFCC00")
    def continue_all(): pause_flag.clear(); lbl.config(text="▶️ Resumed Inference", foreground="#00FFAA")

    btn_start.configure(command=start_all)
    btn_pause.configure(command=pause_all)
    btn_resume.configure(command=continue_all)
    btn_stop.configure(command=stop_all)

    root.protocol("WM_DELETE_WINDOW", stop_all)
    root.mainloop()

# ---------------- Entry ----------------
if __name__ == "__main__":
    start_gui()
