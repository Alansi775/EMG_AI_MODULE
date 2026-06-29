# ===========================================
# servo_activation_metrics.py (metrics logging, append mode)
# ===========================================
import os, time, threading, serial, pickle, queue
import numpy as np
import myo
from myo_ecn.listeners import Buffer, ConnectionChecker
from tensorflow.keras.models import load_model
import tensorflow as tf
from collections import deque
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ---------- Arduino setup ----------
ARDUINO_PORT = 'COM3'
BAUDRATE = 115200
try:
    arduino = serial.Serial(ARDUINO_PORT, BAUDRATE, timeout=0, write_timeout=0)
    arduino_status = "✅ Connected"
except Exception:
    arduino = None
    arduino_status = "⚠️ Not Connected"

# ---------- Servo configuration ----------
servo_names = ["thumb", "index", "middle", "ring", "little", "wrist"]
servo_config = {
    "thumb":  {"relax": 180, "close": 100},
    "index":  {"relax": 180, "close": 70},
    "middle": {"relax": 180, "close": 55},
    "ring":   {"relax": 180, "close": 80},
    "little": {"relax": 0,   "close": 100},
    "wrist":  {"relax": 90,  "left": 180, "right": 0},
}
gesture_to_angles = {
    # All fingers closed
    "fist":   {f: cfg["close"] for f, cfg in servo_config.items() if f != "wrist"},

    # All fingers open
    "open":   {f: cfg["relax"] for f, cfg in servo_config.items() if f != "wrist"},

    # Only thumb open (others closed)
    "thumb": {
        "thumb": servo_config["thumb"]["relax"],
        "index": servo_config["index"]["close"],
        "middle": servo_config["middle"]["close"],
        "ring": servo_config["ring"]["close"],
        "little": servo_config["little"]["close"]
    },

    # Only index open
    "index": {
        "thumb": servo_config["thumb"]["close"],
        "index": servo_config["index"]["relax"],
        "middle": servo_config["middle"]["close"],
        "ring": servo_config["ring"]["close"],
        "little": servo_config["little"]["close"]
    },

    # Only middle open
    "middle": {
        "thumb": servo_config["thumb"]["close"],
        "index": servo_config["index"]["close"],
        "middle": servo_config["middle"]["relax"],
        "ring": servo_config["ring"]["close"],
        "little": servo_config["little"]["close"]
    },

    # Only ring open
    "ring": {
        "thumb": servo_config["thumb"]["close"],
        "index": servo_config["index"]["close"],
        "middle": servo_config["middle"]["close"],
        "ring": servo_config["ring"]["relax"],
        "little": servo_config["little"]["close"]
    },

    # Only little open
    "little": {
        "thumb": servo_config["thumb"]["close"],
        "index": servo_config["index"]["close"],
        "middle": servo_config["middle"]["close"],
        "ring": servo_config["ring"]["close"],
        "little": servo_config["little"]["relax"]
    },

    # "OK" gesture (thumb and index partially closed)
    "ok": {
        "thumb": int((servo_config["thumb"]["relax"] + servo_config["thumb"]["close"]) / 3),
        "index": int((servo_config["index"]["relax"] + servo_config["index"]["close"]) / 2.5),
    },

    # Neutral relaxed pose
    "relax": {f: cfg["relax"] for f, cfg in servo_config.items() if f != "wrist"},

    # Wrist control
    "wrist_left":  {"wrist": servo_config["wrist"]["left"]},
    "wrist_right": {"wrist": servo_config["wrist"]["right"]},
}


# ---------- Helper functions ----------
def add_pos_encoding(x, d_model=128):
    pos = tf.range(tf.shape(x)[1], dtype=tf.float32)
    i = tf.range(d_model, dtype=tf.float32)
    angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    angles = tf.expand_dims(pos, 1) * tf.expand_dims(angle_rates, 0)
    sines = tf.sin(angles[:, 0::2])
    coses = tf.cos(angles[:, 1::2])
    pe = tf.concat([sines, coses], axis=-1)
    return x + tf.expand_dims(pe, 0)

def normalize_emg(emg, mean, std):
    return (emg - mean) / (std + 1e-8)

# ---------- Serial listener thread ----------
serial_feedback = queue.Queue()
def serial_listener(arduino, stop_flag):
    while not stop_flag.is_set():
        try:
            line = arduino.readline().decode('ascii', errors='ignore').strip()
            if line.startswith("DONE"):
                serial_feedback.put((time.perf_counter(), line))
        except Exception:
            pass

# ---------- Realtime thread ----------
def realtime_thread(ui_set_status, ui_set_gesture, emg_buffer, model, label_encoder,
                    metadata, stop_flag, plot_state, metrics_log):
    mean, std = metadata.get("mean"), metadata.get("std")
    gestures = metadata.get("gestures", [])
    confidence_threshold = 0.90
    winlen = int(metadata.get("winlen", 128))

    sample_dt = 0.01  # 100 Hz update rate
    t_window = 5.0
    maxlen = int(t_window / sample_dt) + 32

    t_buf = deque(maxlen=maxlen)
    emg_buf = [deque(maxlen=512) for _ in range(8)]
    servo_buf = {s: deque(maxlen=maxlen) for s in servo_names}

    current_servo_state = {s: 0.0 for s in servo_names}
    current_servo_state["wrist"] = 0.0

    command_id = 0
    start_time = time.perf_counter()
    next_tick = start_time

    hub = myo.Hub()
    ui_set_status("⚙️ Streaming EMG data…  |  Gesture: —")

    with hub.run_in_background(emg_buffer.on_event):
        while not stop_flag.is_set():
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(0.0005)
                continue
            next_tick += sample_dt

            # ----- EMG -----
            if len(emg_buffer.emg_data_queue) == 0:
                continue
            emg_block = np.array([x[1] for x in emg_buffer.get_emg_data()])
            emg_for_model = normalize_emg(emg_block, mean, std) if (mean is not None and std is not None) else emg_block
            for i in range(8):
                emg_buf[i].extend(emg_block[:, i])

            # ----- Gesture prediction -----
            gesture = "relax"
            t_pred = None
            if model is not None and len(emg_for_model) >= winlen:
                X = emg_for_model[-winlen:, :].reshape(1, winlen, 8)
                preds = model.predict(X, verbose=0)
                if float(np.max(preds)) >= confidence_threshold:
                    pred = int(np.argmax(preds))
                    gesture = (
                        label_encoder.inverse_transform([pred])[0]
                        if label_encoder else (gestures[pred] if gestures else "relax")
                    )
                    t_pred = time.perf_counter()
            ui_set_gesture(gesture)

            # ----- Servo command -----
            if arduino:
                angles = gesture_to_angles.get(
                    gesture, {f: servo_config[f]["relax"] for f in servo_names}
                )
                target_state = {}
                for s in servo_names:
                    if s == "wrist":
                        w = angles.get("wrist", servo_config["wrist"]["relax"])
                        if w == servo_config["wrist"]["relax"]:
                            target_state[s] = 0.0
                        elif w == servo_config["wrist"]["right"]:
                            target_state[s] = 1.0
                        else:
                            target_state[s] = -1.0
                    else:
                        cfg = servo_config[s]
                        a = angles.get(s, cfg["relax"])
                        target_state[s] = 1.0 if abs(a - cfg["close"]) <= abs(a - cfg["relax"]) else 0.0

                if target_state != current_servo_state:
                    command_id += 1
                    cmd = (f"C{command_id};"
                           f"T{angles.get('thumb',180)},I{angles.get('index',180)},"
                           f"M{angles.get('middle',180)},R{angles.get('ring',180)},"
                           f"L{angles.get('little',0)},W{angles.get('wrist',90)}\n")
                    t_servo = time.perf_counter()
                    try:
                        arduino.write(cmd.encode('ascii', errors='ignore'))
                        arduino.flush()
                    except Exception:
                        pass
                    current_servo_state.update(target_state)

                    # Record metrics
                    metrics_log.append({
                        "gesture": gesture,
                        "t_emg": now - start_time,
                        "t_pred": (t_pred - start_time) if t_pred else None,
                        "t_servo": t_servo - start_time,
                        "lat_pred": (t_pred - now) if t_pred else None,
                        "lat_servo": t_servo - (t_pred if t_pred else now)
                    })

            # ----- Timeline -----
            t_now = now - start_time
            t_buf.append(t_now)

            # Update EMG and servo plots
            for i, line in enumerate(plot_state["emg_lines"]):
                y = np.fromiter(emg_buf[i], dtype=float)
                if y.size > 1:
                    x = np.linspace(max(0, t_now - t_window), t_now, y.size)
                    line.set_data(x, y)
            for s, line in zip(servo_names, plot_state["servo_lines"]):
                servo_buf[s].append(current_servo_state[s])
                line.set_data(t_buf, servo_buf[s])
            for ax in plot_state["all_axes"]:
                ax.set_xlim(max(0, t_now - t_window), t_now)
            plot_state["schedule_draw"]()

# ---------- GUI ----------
def start_gui():
    myo.init(sdk_path=r"C:\\Users\\ThinkCentre\\Desktop\\Power\\myo_sdk")
    myo_status = "✅ Connected" if ConnectionChecker().ok else "⚠️ Not Connected"

    # Load model
    model, label_encoder = None, None
    metadata = {"winlen": 128, "gestures": []}
    try:
        model = load_model("models/trained_model.h5", custom_objects={"add_pos_encoding": add_pos_encoding})
        with open("models/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        label_encoder = metadata.get("label_encoder")
    except Exception:
        pass

    emg_buffer = Buffer(buffer_len=512)

    root = tk.Tk()
    root.title("🦾 EMG + Servo Realtime (Metrics Logging)")
    root.attributes("-fullscreen", True)
    root.configure(bg="#121212")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TLabel", background="#121212", foreground="#00FFFF", font=("Segoe UI", 12))
    style.configure("TButton", background="#1F1F1F", foreground="#00FF99",
                    font=("Segoe UI", 11, "bold"), padding=6, relief="flat")
    style.map("TButton", background=[("active", "#00FF99")], foreground=[("active", "black")])

    top = ttk.Frame(root); top.pack(fill="x", padx=16, pady=(10, 6))
    ttk.Label(top, text="🦾 EMG + Servo Realtime", font=("Segoe UI", 18, "bold")).pack(side="left")
    status_lbl = ttk.Label(top, text=f"Myo: {myo_status}    Arduino: {arduino_status}")
    status_lbl.pack(side="left", padx=16)

    btns = ttk.Frame(top); btns.pack(side="right")
    btn_start = ttk.Button(btns, text="▶️ Start")
    btn_stop  = ttk.Button(btns, text="🟥 Stop & Save")
    btn_start.grid(row=0, column=0, padx=8)
    btn_stop.grid(row=0, column=1, padx=8)

    lbl = ttk.Label(root, text="Press ▶️ Start", font=("Segoe UI", 13, "bold"),
                    foreground="#FFFFFF", background="#121212")
    lbl.pack(pady=4)

    # Figure (8 EMG + 6 servo)
    fig = Figure(figsize=(12, 9), facecolor="#121212")
    axes = [fig.add_subplot(14, 1, i + 1, facecolor="#0A0A0A") for i in range(14)]
    emg_lines, servo_lines = [], []
    for i, ax in enumerate(axes[:8]):
        line, = ax.plot([], [], color="#00FF66", linewidth=1.0)
        emg_lines.append(line)
        ax.set_ylabel(f"CH{i+1}", color="#00FFFF", fontsize=8)
        ax.tick_params(axis="x", colors="#00FFFF", labelsize=6)
        ax.tick_params(axis="y", colors="#00FFFF", labelsize=6)
        ax.set_ylim(-100, 100)
    for j, name in enumerate(servo_names):
        ax = axes[8 + j]
        line, = ax.plot([], [], color="#FF66FF", linewidth=1.3, drawstyle="steps-post")
        servo_lines.append(line)
        ax.set_ylabel(name, color="#FF66FF", fontsize=8)
        ax.set_ylim(-1.1, 1.1)
        ax.tick_params(axis="x", colors="#00FFFF", labelsize=6)
        ax.tick_params(axis="y", colors="#FF66FF", labelsize=6)
    axes[-1].set_xlabel("Time (s)", color="#00FFFF", fontsize=8)
    fig.tight_layout(pad=0.3)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=16, pady=(6, 12))

    stop_flag = threading.Event()
    metrics_log = []

    def ui_set_status(text):
        root.after(0, lambda: lbl.config(text=text, foreground="#00FFAA"))
    def ui_set_gesture(gesture):
        root.after(0, lambda: lbl.config(
            text=f"⚙️ Streaming EMG data…  |  Gesture: {gesture.upper()}",
            foreground="#00FFAA"))
    def schedule_draw():
        root.after(0, canvas.draw_idle)

    plot_state = {"emg_lines": emg_lines, "servo_lines": servo_lines,
                  "all_axes": axes, "schedule_draw": schedule_draw}

    def start_all():
        stop_flag.clear()
        if arduino:
            threading.Thread(target=serial_listener, args=(arduino, stop_flag), daemon=True).start()
        threading.Thread(target=realtime_thread,
                         args=(ui_set_status, ui_set_gesture, emg_buffer,
                               model, label_encoder, metadata,
                               stop_flag, plot_state, metrics_log),
                         daemon=True).start()
        lbl.config(text="Starting…", foreground="#00FFAA")

    def stop_all():
        stop_flag.set()
        file_exists = os.path.exists("metrics_log.txt")
        # Append metrics instead of overwriting
        with open("metrics_log.txt", "a") as f:
            if not file_exists:
                f.write(f"{'Gesture':<15}{'t_emg (s)':<15}{'t_pred (s)':<15}{'t_servo (s)':<15}"
                        f"{'lat_pred (s)':<15}{'lat_servo (s)':<15}\n")
                f.write("="*85 + "\n")
            for m in metrics_log:
                f.write(f"{m['gesture']:<15}"
                        f"{(m['t_emg']):<15.6f}"
                        f"{(m['t_pred'] if m['t_pred'] else 0):<15.6f}"
                        f"{(m['t_servo']):<15.6f}"
                        f"{(m['lat_pred'] if m['lat_pred'] else 0):<15.6f}"
                        f"{(m['lat_servo']):<15.6f}\n")
        lbl.config(text="Stopped. Metrics appended to metrics_log.txt", foreground="#FF5555")
        root.after(350, root.destroy)

    btn_start.configure(command=start_all)
    btn_stop.configure(command=stop_all)
    root.protocol("WM_DELETE_WINDOW", stop_all)
    root.mainloop()

if __name__ == "__main__":
    start_gui()
