# ===========================================
# 4_gesture_recorder.py
# ===========================================
# Works like 3_inference.py but records 5 s of stable gesture data
# Asks confirmation and saves CSVs into data/<gesture_name>/
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
from tkinter import ttk, messagebox
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
    'wrist':  {'pin': 11, 'relax': 90, 'left': 180, 'right': 0, 'up': 135, 'down': 45},
}

gesture_emojis = {
    'fist': '', 'open': '️', 'thumb': '👍', 'index': '☝️',
    'middle': '🖕', 'ring': '💍', 'little': '🤙', 'ok': '👌',
    'wrist_left': '↩️', 'wrist_right': '↪️', 'wrist_up': '⬆️',
    'wrist_down': '⬇️', 'relax': '😌',
}

# ---------------- Normalization ----------------
def normalize_emg(emg, mean, std):
    return (emg - mean) / (std + 1e-8)

# ---------------- Positional Encoding ----------------
def add_pos_encoding(x, d_model=128):
    pos = tf.range(tf.shape(x)[1], dtype=tf.float32)
    i = tf.range(d_model, dtype=tf.float32)
    angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    angles = tf.expand_dims(pos, 1) * tf.expand_dims(angle_rates, 0)
    sines = tf.sin(angles[:, 0::2])
    coses = tf.cos(angles[:, 1::2])
    pe = tf.concat([sines, coses], axis=-1)
    return x + tf.expand_dims(pe, 0)

# ---------------- CSV Helpers ----------------
def next_csv_index(folder):
    os.makedirs(folder, exist_ok=True)
    max_idx = 0
    for f in os.listdir(folder):
        if f.endswith(".csv"):
            try: max_idx = max(max_idx, int(os.path.splitext(f)[0]))
            except: pass
    return max_idx + 1

def save_emg_csv(matrix, folder):
    idx = next_csv_index(folder)
    path = os.path.join(folder, f"{idx}.csv")
    np.savetxt(path, matrix, fmt="%.2f", delimiter=", ")
    return path

# ---------------- Inference + Recording Thread ----------------
def recorder_thread(update_text, ask_user, emg_buffer, model, label_encoder, metadata, stop_flag, pause_flag):
    gestures = metadata['gestures']
    mean, std = metadata.get('mean'), metadata.get('std')
    fs, winlen = 200, metadata['winlen']
    record_secs = 5.0
    confidence_threshold = 0.85
    vote_window = 7

    pred_history = deque(maxlen=vote_window)
    raw_buffer = deque(maxlen=int(fs * record_secs))
    current_gesture = None
    stable_start = None
    recording = False

    hub = myo.Hub()
    update_text("⚙️ Starting Myo EMG stream...")

    with hub.run_in_background(emg_buffer.on_event):
        while not stop_flag.is_set():
            if pause_flag.is_set():
                time.sleep(0.1)
                continue
            time.sleep(0.025)
            if len(emg_buffer.emg_data_queue) < winlen:
                continue

            # --- Fetch EMG ---
            emg = np.array([x[1] for x in emg_buffer.get_emg_data()])
            if mean is not None and std is not None:
                emg = normalize_emg(emg, mean, std)

            # keep full raw 5 s buffer
            for v in emg:
                raw_buffer.append(v)

            # --- Predict ---
            X = emg.reshape(1, emg.shape[0], emg.shape[1])
            preds = model.predict(X, verbose=0)
            pred_prob, pred_class = float(np.max(preds)), int(np.argmax(preds))
            if pred_prob < confidence_threshold:
                stable_start = None
                continue

            pred_history.append((pred_class, pred_prob))
            scores = {}
            for c, p in pred_history:
                scores[c] = scores.get(c, 0) + p
            final_pred = max(scores, key=scores.get)
            gesture_name = (
                label_encoder.inverse_transform([final_pred])[0]
                if label_encoder else gestures[final_pred]
            )
            emoji = gesture_emojis.get(gesture_name, "❓")

            conf_pct = int(pred_prob * 100)
            bar = "⬛" * max(0, min(10, (conf_pct - 80)//2))
            update_text(f"{emoji} {gesture_name} | {bar} {conf_pct}%")

            # --- Stability detection ---
            now = time.monotonic()
            if gesture_name != current_gesture:
                current_gesture = gesture_name
                stable_start = now
            elif stable_start and (now - stable_start) >= record_secs and not recording:
                recording = True
                captured = np.array(raw_buffer)
                ask_user(gesture_name, captured)
                recording = False
                stable_start = None

# ---------------- GUI ----------------
def start_gui():
    myo.init(sdk_path=r'C:\\Users\\ThinkCentre\\Desktop\\Power\\myo_sdk')
    myo_status = " Connected" if ConnectionChecker().ok else "⚠️ Not Connected"

    MODEL_DIR = "models"
    MODEL_FILE = next((os.path.join(MODEL_DIR,f)
                       for f in ["trained_model.h5","trained_model.keras"]
                       if os.path.exists(os.path.join(MODEL_DIR,f))), None)
    if not MODEL_FILE: raise FileNotFoundError("No trained_model found in /models")

    META_FILE = os.path.join(MODEL_DIR, "metadata.pkl")
    print(f" Loading model from: {MODEL_FILE}")
    model = load_model(MODEL_FILE, custom_objects={'add_pos_encoding': add_pos_encoding})
    with open(META_FILE, "rb") as f: metadata = pickle.load(f)
    label_encoder = metadata.get("label_encoder")
    emg_buffer = Buffer(metadata["winlen"])

    gesture_names = list(metadata['gestures'].values()) if isinstance(metadata['gestures'], dict) else metadata['gestures']

    root = tk.Tk()
    root.title("🧠 EMG Recorder with Gesture Confirmation")
    root.geometry("1150x720")

    # Status
    status_frame = ttk.Frame(root); status_frame.pack(pady=5)
    ttk.Label(status_frame, text=f"Myo: {myo_status}", font=("Segoe UI", 12)).grid(row=0,column=0,padx=15)
    ttk.Label(status_frame, text=f"Arduino: {arduino_status}", font=("Segoe UI", 12)).grid(row=0,column=1,padx=15)
    lbl = ttk.Label(root, text="Press ▶️ Start to begin", font=("Segoe UI", 14,"bold")); lbl.pack(pady=10)

    # Plot
    fig = Figure(figsize=(10,5))
    axes = [fig.add_subplot(8,1,i+1) for i in range(8)]
    lines=[]
    for i,ax in enumerate(axes):
        (l,)=ax.plot(np.zeros(512)); lines.append(l)
        ax.set_ylim(-128,128); ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"CH{i+1}", rotation=0, labelpad=30, fontsize=9)
    canvas = FigureCanvasTkAgg(fig,master=root)
    canvas.get_tk_widget().pack(pady=10)

    paused_label=tk.Label(root,text="⏸️ PAUSED",font=("Segoe UI",40,"bold"),
                          fg="red",bg="white",bd=2,relief="solid")
    paused_label.place(x=450,y=320); paused_label.lower()

    plot_buffer=np.zeros((8,512))
    stop_flag=threading.Event(); pause_flag=threading.Event()
    thread_ref=None

    # --- Ask user confirmation ---
    def ask_user(gesture_name, captured):
        def save_to(name):
            folder=os.path.join("data",name)
            path=save_emg_csv(captured,folder)
            messagebox.showinfo("Saved",f" Saved 5 s EMG to {path}")
        def manual_select():
            win=tk.Toplevel(root); win.title("Select Correct Gesture")
            ttk.Label(win,text="Select correct gesture:",font=("Segoe UI",12)).pack(pady=5)
            combo=ttk.Combobox(win,values=gesture_names,state="readonly"); combo.pack(pady=5)
            def confirm():
                val=combo.get()
                if val: save_to(val)
                win.destroy()
            ttk.Button(win,text="Save",command=confirm).pack(pady=5)
        if messagebox.askyesno("Confirm",f"Is this the '{gesture_name}' gesture?"):
            save_to(gesture_name)
        else:
            manual_select()

    # --- Plot updater ---
    def update_text(msg):
        lbl.config(text=msg); root.update_idletasks()
    def update_plot():
        nonlocal plot_buffer
        if len(emg_buffer.emg_data_queue)>0:
            chunk=np.array([x[1] for x in emg_buffer.get_emg_data()]).T
            n_new=chunk.shape[1]
            plot_buffer=np.hstack((plot_buffer[:,n_new:],chunk)) if n_new<512 else chunk[:,-512:]
            for ch in range(8): lines[ch].set_ydata(plot_buffer[ch])
            canvas.draw()
        if not stop_flag.is_set(): root.after(50,update_plot)

    # --- Buttons ---
    def start_all():
        nonlocal thread_ref
        if thread_ref and thread_ref.is_alive(): return
        stop_flag.clear(); pause_flag.clear(); paused_label.lower()
        update_text("🧠 Starting EMG recording...")
        thread_ref=threading.Thread(target=recorder_thread,
            args=(update_text,ask_user,emg_buffer,model,label_encoder,metadata,stop_flag,pause_flag),
            daemon=True)
        thread_ref.start(); update_plot()
    def pause_all(): pause_flag.set(); paused_label.lift(); update_text("⏸️ Paused")
    def cont_all(): pause_flag.clear(); paused_label.lower(); update_text("▶️ Resumed")
    def stop_all(): stop_flag.set(); update_text("🛑 Stopped"); root.destroy()

    btn_frame=ttk.Frame(root); btn_frame.pack(pady=10)
    ttk.Button(btn_frame,text="▶️ Start",command=start_all).grid(row=0,column=0,padx=10)
    ttk.Button(btn_frame,text="⏸️ Pause",command=pause_all).grid(row=0,column=1,padx=10)
    ttk.Button(btn_frame,text="🟢 Continue",command=cont_all).grid(row=0,column=2,padx=10)
    ttk.Button(btn_frame,text="🟥 Stop & Exit",command=stop_all).grid(row=0,column=3,padx=10)

    root.protocol("WM_DELETE_WINDOW", stop_all)
    root.mainloop()

# ---------------- Entry ----------------
if __name__ == "__main__":
    start_gui()
