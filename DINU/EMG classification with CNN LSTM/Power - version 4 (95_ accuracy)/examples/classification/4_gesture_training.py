# 4_gesture_training.py
import os, sys, time, threading, pickle
import numpy as np
import myo
from collections import deque
from myo_ecn.listeners import Buffer, ConnectionChecker
from tensorflow.keras.models import load_model
import serial
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ================== Arduino Setup ==================
ARDUINO_PORT = 'COM3'
BAUDRATE = 9600
try:
    arduino = serial.Serial(ARDUINO_PORT, BAUDRATE, timeout=1)
    arduino_status = "✅ Connected"
except Exception as e:
    arduino = None
    arduino_status = "⚠️ Not Connected"

# ================== Servo configuration ==================
servo_config = {
    'thumb':  {'pin': 3,  'relax': 180, 'close': 100},
    'index':  {'pin': 5,  'relax': 180, 'close': 70},
    'middle': {'pin': 6,  'relax': 180, 'close': 55},
    'ring':   {'pin': 9,  'relax': 180, 'close': 80},
    'little': {'pin': 12, 'relax': 0,   'close': 100},
    'wrist':  {'pin': 11, 'relax': 90, 'left': 180, 'right': 0, 'up': 135, 'down': 45},
}

# ================== Gesture → Servo mapping ==================
gesture_to_angles = {
    'fist':   {f: cfg['close'] for f, cfg in servo_config.items() if f != 'wrist'},
    'open':   {f: cfg['relax'] for f, cfg in servo_config.items() if f != 'wrist'},
    'thumb':  {'thumb': servo_config['thumb']['close']},
    'index':  {'index': servo_config['index']['close']},
    'middle': {'middle': servo_config['middle']['close']},
    'ring':   {'ring': servo_config['ring']['close']},
    'little': {'little': servo_config['little']['close']},
    'ok':     {
        'thumb': int((servo_config['thumb']['relax'] + servo_config['thumb']['close']) / 3),
        'index': int((servo_config['index']['relax'] + servo_config['index']['close']) / 2.5)
    },
    'relax':  {f: cfg['relax'] for f, cfg in servo_config.items() if f != 'wrist'},
}

# ================== Emojis ==================
gesture_emojis = {
    'fist': '✊',
    'open': '🖐️',
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

# ================== Helpers ==================
def normalize_emg(emg, mean, std):
    return (emg - mean) / (std + 1e-8)

def next_csv_index(folder):
    os.makedirs(folder, exist_ok=True)
    max_idx = 0
    for name in os.listdir(folder):
        if name.lower().endswith('.csv'):
            base = os.path.splitext(name)[0]
            try:
                idx = int(base)
                if idx > max_idx:
                    max_idx = idx
            except:
                pass
    return max_idx + 1

def save_emg_csv(matrix_2d, folder):
    """matrix_2d: (N_samples, 8) floats. Save as 'NNN.csv' with ', ' and 1 decimal."""
    idx = next_csv_index(folder)
    path = os.path.join(folder, f"{idx}.csv")
    with open(path, 'w') as f:
        for row in matrix_2d:
            f.write(", ".join(f"{x:.1f}" for x in row.tolist()) + "\n")
    return path

# ================== Recording / Inference Thread ==================
def recording_thread(update_text, update_question, emg_buffer, model, label_encoder,
                     metadata, stop_flag, pause_flag):
    gestures = metadata['gestures']              # dict: {idx: name} OR list-like
    mean, std = metadata.get('mean'), metadata.get('std')
    winlen = metadata['winlen']
    fs = 200                                     # Myo EMG nominal rate
    record_seconds = 5.0
    min_confidence = 0.90
    vote_window = 7

    # Rolling raw EMG (last 5s) with de-duplication by timestamp
    raw_emg_deque = deque(maxlen=int(fs * record_seconds))
    last_ts_seen = -1

    # Stability detection
    current_gesture = None
    stable_start_t = None
    cool_down_until = 0.0     # to avoid spamming confirmations
    pending_confirmation = False
    captured_matrix = None    # numpy (N,8) to save once confirmed

    # Wrist
    wrist_position = servo_config['wrist']['relax']

    # Prediction vote history
    from collections import deque as dq
    pred_history = dq(maxlen=vote_window)

    # Build reverse mapping from label id to name
    def label_to_name(lbl):
        if label_encoder is not None:
            return label_encoder.inverse_transform([lbl])[0]
        else:
            # gestures might be a dict {0:'open',...} or list
            return gestures[lbl] if not isinstance(gestures, dict) else gestures[lbl]

    # Start background Myo hub
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

            # --------- Pull fresh EMG (timestamps + samples) and fill raw buffer (no dup) ---------
            emg_list = emg_buffer.get_emg_data()  # list of (ts, vec8)
            new_rows = []
            for ts, vec in emg_list:
                if ts > last_ts_seen:
                    new_rows.append(vec)
                    last_ts_seen = ts
            if new_rows:
                for v in new_rows:
                    raw_emg_deque.append(np.array(v, dtype=float))

            # Need enough samples to classify
            window_matrix = np.array([x[1] for x in emg_list])  # for classifier, as your inference does
            if mean is not None and std is not None:
                window_matrix = normalize_emg(window_matrix, mean, std)

            X = window_matrix.reshape(1, window_matrix.shape[0], window_matrix.shape[1])
            preds = model.predict(X, verbose=0)
            pred_prob = float(np.max(preds))
            pred_class = int(np.argmax(preds))

            if pred_prob >= min_confidence:
                pred_history.append((pred_class, pred_prob))
            else:
                # drop low conf impacts on stability by resetting timer
                current_gesture = None
                stable_start_t = None
                update_question("")  # clear any previous
                # normal HUD text
                emoji = "❓"
                update_text(f"{emoji} Low confidence  |  {int(pred_prob*100)}%")
                continue

            # Weighted vote
            score = {}
            for c, p in pred_history:
                score[c] = score.get(c, 0.0) + p
            final_pred = max(score, key=score.get)
            gesture_name = label_to_name(final_pred)
            emoji = gesture_emojis.get(gesture_name, "❓")

            # HUD confidence bar (80–100% -> 0..10 blocks)
            confidence_pct = int(pred_prob * 100)
            blocks = int((confidence_pct - 80) / 2)
            blocks = max(0, min(blocks, 10))
            conf_bar = "⬛" * blocks + "⬜" * (10 - blocks)
            if confidence_pct >= 99:
                conf_bar = "⬛" * 10
            update_text(f"{emoji} {gesture_name}  |  {conf_bar}  {confidence_pct}%")

            # --------- Stability detection over 5 seconds ---------
            now = time.monotonic()
            if now < cool_down_until:
                # in cooldown after last confirmation
                pass
            else:
                if current_gesture != gesture_name:
                    current_gesture = gesture_name
                    stable_start_t = now if pred_prob >= min_confidence else None
                    update_question("")  # reset question text
                else:
                    if stable_start_t is not None and (now - stable_start_t) >= record_seconds and not pending_confirmation:
                        # We have at least 5s continuous same gesture, ask user to confirm & save
                        # capture last 5s raw EMG as matrix (N,8)
                        if len(raw_emg_deque) >= int(fs * record_seconds):
                            captured = np.array(raw_emg_deque, dtype=float)
                            captured_matrix = captured.copy()
                            pending_confirmation = True
                            # ask in GUI below plots
                            update_question(f"Is it '{gesture_name}'?  (Click YES to save, NO to discard)")
                        else:
                            # not enough raw yet (unlikely), retry next loop
                            pass

            # --------- Arduino output (same as inference) ---------
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
                cmd = (
                    f"T{angles.get('thumb', servo_config['thumb']['relax'])},"
                    f"I{angles.get('index', servo_config['index']['relax'])},"
                    f"M{angles.get('middle', servo_config['middle']['relax'])},"
                    f"R{angles.get('ring', servo_config['ring']['relax'])},"
                    f"L{angles.get('little', servo_config['little']['relax'])},"
                    f"W{wrist_position}\n"
                )
                try:
                    arduino.write(cmd.encode())
                except:
                    pass

            # --------- Handle pending confirmation save/skip via shared state ---------
            # The GUI thread will call confirm_yes() / confirm_no() which mutate these closures:
            def do_save():
                nonlocal pending_confirmation, cool_down_until, captured_matrix, current_gesture, stable_start_t
                if not pending_confirmation or captured_matrix is None:
                    return False
                # Save to data/<gesture_name>/<NNN>.csv
                folder = os.path.join('data', gesture_name)
                try:
                    path = save_emg_csv(captured_matrix, folder)
                    # cooldown 2s
                    cool_down_until = time.monotonic() + 2.0
                    pending_confirmation = False
                    captured_matrix = None
                    current_gesture = None
                    stable_start_t = None
                    update_question(f"✅ Saved to {path}")
                    return True
                except Exception as e:
                    update_question(f"⚠️ Save error: {e}")
                    pending_confirmation = False
                    captured_matrix = None
                    return False

            def do_discard():
                nonlocal pending_confirmation, cool_down_until, captured_matrix, current_gesture, stable_start_t
                if not pending_confirmation:
                    return False
                # Discard and cool down a bit to avoid instant re-ask
                cool_down_until = time.monotonic() + 1.0
                pending_confirmation = False
                captured_matrix = None
                current_gesture = None
                stable_start_t = None
                update_question("🗑️ Discarded")
                return True

            # Expose handlers to GUI through attributes on function object (hacky but simple)
            recording_thread.do_save = do_save
            recording_thread.do_discard = do_discard
            recording_thread.pending_confirmation = lambda: pending_confirmation
            recording_thread.current_candidate = lambda: current_gesture

# ================== GUI ==================
def start_gui():
    # ---- Init Myo / Model ----
    myo.init(sdk_path=r'C:\Users\ThinkCentre\Desktop\Power\myo_sdk')
    myo_status = "✅ Connected" if ConnectionChecker().ok else "⚠️ Not Connected"

    MODEL_FILE, META_FILE = 'models/trained_model.keras', 'models/metadata.pkl'
    model = load_model(MODEL_FILE)
    with open(META_FILE, 'rb') as f:
        metadata = pickle.load(f)
    label_encoder = metadata.get('label_encoder')
    emg_buffer = Buffer(metadata['winlen'])

    # ---- GUI ----
    root = tk.Tk()
    root.title("🦾 EMG Self-Labeling Recorder (matches inference)")
    root.geometry("1200x780")

    # Status bar
    status_frame = ttk.Frame(root)
    status_frame.pack(pady=5)
    ttk.Label(status_frame, text=f"Myo: {myo_status}", font=("Segoe UI", 12)).grid(row=0, column=0, padx=15)
    ttk.Label(status_frame, text=f"Arduino: {arduino_status}", font=("Segoe UI", 12)).grid(row=0, column=1, padx=15)

    # Gesture / HUD
    lbl = ttk.Label(root, text="Press ▶️ Start to begin", font=("Segoe UI", 14, "bold"))
    lbl.pack(pady=10)

    # Plot
    fig = Figure(figsize=(10.5, 5.2))
    axes = [fig.add_subplot(8, 1, i + 1) for i in range(8)]
    lines = []
    for i, ax in enumerate(axes):
        line, = ax.plot(np.zeros(512))
        lines.append(line)
        ax.set_ylim(-128, 128)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"CH{i+1}", rotation=0, labelpad=30, fontsize=9, weight='bold', color='black')
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(pady=10)

    # Confirmation row (below plots, no flashing)
    question_var = tk.StringVar(value="")
    q_frame = ttk.Frame(root)
    q_frame.pack(pady=6)
    q_label = ttk.Label(q_frame, textvariable=question_var, font=("Segoe UI", 12))
    q_label.grid(row=0, column=0, padx=10)
    yes_btn = ttk.Button(q_frame, text="✅ Yes")
    no_btn  = ttk.Button(q_frame, text="❌ No")
    yes_btn.grid(row=0, column=1, padx=6)
    no_btn.grid(row=0, column=2, padx=6)

    # Overlay “Paused”
    paused_label = tk.Label(root, text="⏸️ PAUSED", font=("Segoe UI", 36, "bold"),
                            fg="red", bg="white", bd=2, relief="solid")
    paused_label.place(x=480, y=350)
    paused_label.lower()

    # Rolling buffer for plotting (kept same as inference)
    plot_buffer = np.zeros((8, 512))
    stop_flag = threading.Event()
    pause_flag = threading.Event()
    worker_ref = None

    # --- Updaters ---
    def update_text(msg):
        lbl.config(text=msg)
        root.update_idletasks()

    def update_question(msg):
        question_var.set(msg)
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

    # --- Button handlers ---
    def start_all():
        nonlocal worker_ref
        if worker_ref and worker_ref.is_alive():
            return
        stop_flag.clear()
        pause_flag.clear()
        paused_label.lower()
        update_text("🧠 Starting EMG stream...")
        worker_ref = threading.Thread(
            target=recording_thread,
            args=(update_text, update_question, emg_buffer, model, label_encoder,
                  metadata, stop_flag, pause_flag),
            daemon=True
        )
        worker_ref.start()
        update_plot()

    def pause_all():
        pause_flag.set()
        paused_label.lift()
        update_text("⏸️ Paused")

    def continue_all():
        pause_flag.clear()
        paused_label.lower()
        update_text("▶️ Resumed")

    def stop_all():
        stop_flag.set()
        update_text("🛑 Stopped")
        root.destroy()

    def confirm_yes():
        # call the save handler exposed by the thread
        try:
            if hasattr(recording_thread, 'pending_confirmation') and recording_thread.pending_confirmation():
                recording_thread.do_save()
        except Exception as e:
            update_question(f"⚠️ Save error: {e}")

    def confirm_no():
        try:
            if hasattr(recording_thread, 'pending_confirmation') and recording_thread.pending_confirmation():
                recording_thread.do_discard()
        except Exception as e:
            update_question(f"⚠️ Discard error: {e}")

    yes_btn.config(command=confirm_yes)
    no_btn.config(command=confirm_no)

    # Buttons row
    btn_frame = ttk.Frame(root); btn_frame.pack(pady=10)
    ttk.Button(btn_frame, text="▶️ Start", command=start_all).grid(row=0, column=0, padx=10)
    ttk.Button(btn_frame, text="⏸️ Pause", command=pause_all).grid(row=0, column=1, padx=10)
    ttk.Button(btn_frame, text="🟢 Continue", command=continue_all).grid(row=0, column=2, padx=10)
    ttk.Button(btn_frame, text="🟥 Stop & Exit", command=stop_all).grid(row=0, column=3, padx=10)

    root.protocol("WM_DELETE_WINDOW", stop_all)
    root.mainloop()

# ================== Entrypoint ==================
if __name__ == '__main__':
    start_gui()
