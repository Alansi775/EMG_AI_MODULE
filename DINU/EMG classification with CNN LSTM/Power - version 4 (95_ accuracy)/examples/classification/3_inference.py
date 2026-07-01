import pickle, time, sys, threading
import numpy as np
import myo
from myo_ecn.listeners import Buffer, ConnectionChecker
from tensorflow.keras.models import load_model
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
except Exception as e:
    arduino = None
    arduino_status = "⚠️ Not Connected"

# ---------------- Servo Config ----------------
servo_config = {
    'thumb':  {'pin': 3,  'relax': 180, 'close': 100},
    'index':  {'pin': 5,  'relax': 180, 'close': 70},
    'middle': {'pin': 6,  'relax': 180, 'close': 55},
    'ring':   {'pin': 9,  'relax': 180, 'close': 80},
    'little': {'pin': 12, 'relax': 0,   'close': 100},
    'wrist':  {'pin': 11, 'relax': 90, 'left': 180, 'right': 0, 'up': 135, 'down': 45},
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
    'fist': '',
    'open': '️',
    'thumb': '👍',
    'index': '☝️',
    'middle': '🖕',
    'ring': '💍',
    'little': '🤙',
    'ok': '👌',
    'wrist_left': '↩️',
    'wrist_right': '↪️',
    'wrist_up': '⬆️',
    'wrist_down': '⬇️',
    'relax': '😌',
}

def normalize_emg(emg, mean, std):
    return (emg - mean) / (std + 1e-8)


# ---------------- Inference Thread with Pause ----------------
def inference_thread_with_pause(update_text, emg_buffer, model, label_encoder, metadata, stop_flag, pause_flag):
    gestures = metadata['gestures']
    mean, std = metadata.get('mean'), metadata.get('std')
    confidence_threshold = 0.9
    vote_window = 7
    pred_history = deque(maxlen=vote_window)
    wrist_position = servo_config['wrist']['relax']

    hub = myo.Hub()
    update_text("⚙️ Starting Myo EMG stream...")

    with hub.run_in_background(emg_buffer.on_event):
        while not stop_flag.is_set():
            if pause_flag.is_set():
                time.sleep(0.1)
                continue

            time.sleep(0.025)
            if len(emg_buffer.emg_data_queue) < metadata['winlen']:
                continue

            emg = np.array([x[1] for x in emg_buffer.get_emg_data()])
            if mean is not None and std is not None:
                emg = normalize_emg(emg, mean, std)

            X = emg.reshape(1, emg.shape[0], emg.shape[1])
            preds = model.predict(X, verbose=0)
            pred_prob = np.max(preds)
            pred_class = np.argmax(preds)

            if pred_prob < confidence_threshold:
                continue

            pred_history.append((pred_class, pred_prob))
            class_scores = {}
            for cls, prob in pred_history:
                class_scores[cls] = class_scores.get(cls, 0) + prob
            final_pred = max(class_scores, key=class_scores.get)
            gesture_name = label_encoder.inverse_transform([final_pred])[0] if label_encoder else gestures[final_pred]
            emoji = gesture_emojis.get(gesture_name, "❓")

            confidence_pct = int(pred_prob * 100)
            blocks = int((confidence_pct - 80) / 2)
            blocks = max(0, min(blocks, 10))
            conf_bar = "⬛" * blocks + "⬜" * (10 - blocks)
            if confidence_pct >= 99:
                conf_bar = "⬛" * 10

            update_text(f"{emoji} {gesture_name}  |  {conf_bar}  {confidence_pct}%")

            # Arduino output
            if arduino:
                if gesture_name == 'wrist_left':
                    wrist_position = servo_config['wrist']['left']
                elif gesture_name == 'wrist_right':
                    wrist_position = servo_config['wrist']['right']
                elif gesture_name == 'wrist_up':
                    wrist_position = servo_config['wrist']['up']
                elif gesture_name == 'wrist_down':
                    wrist_position = servo_config['wrist']['down']
                else:
                    wrist_position = servo_config['wrist']['relax']

                angles = gesture_to_angles.get(
                    gesture_name,
                    {f: cfg['relax'] for f, cfg in servo_config.items() if f != 'wrist'}
                )

                cmd = f"T{angles.get('thumb', servo_config['thumb']['relax'])}," \
                      f"I{angles.get('index', servo_config['index']['relax'])}," \
                      f"M{angles.get('middle', servo_config['middle']['relax'])}," \
                      f"R{angles.get('ring', servo_config['ring']['relax'])}," \
                      f"L{angles.get('little', servo_config['little']['relax'])}," \
                      f"W{wrist_position}\n"
                arduino.write(cmd.encode())


# ---------------- GUI ----------------
def start_gui():
    myo.init(sdk_path=r'C:\Users\ThinkCentre\Desktop\Power\myo_sdk')
    myo_status = " Connected" if ConnectionChecker().ok else "⚠️ Not Connected"

    MODEL_FILE, META_FILE = 'models/trained_model.keras', 'models/metadata.pkl'
    model = load_model(MODEL_FILE)
    with open(META_FILE, 'rb') as f:
        metadata = pickle.load(f)
    label_encoder = metadata.get('label_encoder')
    emg_buffer = Buffer(metadata['winlen'])

    root = tk.Tk()
    root.title("🦾 EMG Gesture Visualization + Control")
    root.geometry("1150x720")

    # ---------------- Connection Status ----------------
    status_frame = ttk.Frame(root)
    status_frame.pack(pady=5)
    ttk.Label(status_frame, text=f"Myo: {myo_status}", font=("Segoe UI", 12)).grid(row=0, column=0, padx=15)
    ttk.Label(status_frame, text=f"Arduino: {arduino_status}", font=("Segoe UI", 12)).grid(row=0, column=1, padx=15)

    lbl = ttk.Label(root, text="Press ▶️ Start to begin", font=("Segoe UI", 14, "bold"))
    lbl.pack(pady=10)

    # ---------------- Plot Setup ----------------
    fig = Figure(figsize=(10, 5))
    axes = [fig.add_subplot(8, 1, i + 1) for i in range(8)]
    lines = []
    for i, ax in enumerate(axes):
        line, = ax.plot(np.zeros(512))
        lines.append(line)
        ax.set_ylim(-128, 128)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(f"CH{i+1}", rotation=0, labelpad=30, fontsize=9, weight='bold', color='black')

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(pady=10)

    # Overlay “Paused” label
    paused_label = tk.Label(root, text="⏸️ PAUSED", font=("Segoe UI", 40, "bold"),
                            fg="red", bg="white", bd=2, relief="solid")
    paused_label.place(x=450, y=320)
    paused_label.lower()  # Initially hidden

    # Rolling buffer for plot
    plot_buffer = np.zeros((8, 512))
    stop_flag = threading.Event()
    pause_flag = threading.Event()
    inference_thread_ref = None

    # ---------------- Functions ----------------
    def update_text(msg):
        lbl.config(text=msg)
        root.update_idletasks()

    def update_plot():
        nonlocal plot_buffer
        if len(emg_buffer.emg_data_queue) > 0:
            emg_chunk = np.array([x[1] for x in emg_buffer.get_emg_data()]).T
            n_new = emg_chunk.shape[1]
            if n_new < 512:
                plot_buffer = np.hstack((plot_buffer[:, n_new:], emg_chunk))
            else:
                plot_buffer = emg_chunk[:, -512:]
            for ch in range(8):
                lines[ch].set_ydata(plot_buffer[ch])
            canvas.draw()
        if not stop_flag.is_set():
            root.after(50, update_plot)

    # ---------------- Button Actions ----------------
    def start_all():
        nonlocal inference_thread_ref
        stop_flag.clear()
        pause_flag.clear()
        paused_label.lower()
        lbl.config(text="🧠 Starting EMG stream...")
        inference_thread_ref = threading.Thread(
            target=inference_thread_with_pause,
            args=(update_text, emg_buffer, model, label_encoder, metadata, stop_flag, pause_flag),
            daemon=True
        )
        inference_thread_ref.start()
        update_plot()

    def pause_all():
        pause_flag.set()
        paused_label.lift()
        lbl.config(text="⏸️ Paused inference")

    def continue_all():
        pause_flag.clear()
        paused_label.lower()
        lbl.config(text="▶️ Resumed inference")

    def stop_all():
        stop_flag.set()
        lbl.config(text="🛑 Stopped")
        root.destroy()

    # ---------------- Buttons ----------------
    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=10)

    ttk.Button(btn_frame, text="▶️ Start", command=start_all).grid(row=0, column=0, padx=10)
    ttk.Button(btn_frame, text="⏸️ Pause", command=pause_all).grid(row=0, column=1, padx=10)
    ttk.Button(btn_frame, text="🟢 Continue", command=continue_all).grid(row=0, column=2, padx=10)
    ttk.Button(btn_frame, text="🟥 Stop & Exit", command=stop_all).grid(row=0, column=3, padx=10)

    root.protocol("WM_DELETE_WINDOW", stop_all)
    root.mainloop()


if __name__ == '__main__':
    start_gui()
