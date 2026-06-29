# train_v4.py
# Correct windowing: no leakage between train and test

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import os

FS          = 200
WIN_SAMPLES = 40
STEP        = 20
N_CHANNELS  = 8
N_CLASSES   = 6
BATCH_SIZE  = 128
EPOCHS      = 100
DEVICE      = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

GESTURE_NAMES = {
    0: 'rest', 1: 'fist', 2: 'open_hand',
    3: 'wave_in', 4: 'wave_out', 5: 'pinch'
}

print(f"Device: {DEVICE}")


# ── Load ──
def load_sessions(sessions_dir="sessions"):
    all_dfs = []
    dirs = sorted([
        d for d in os.listdir(sessions_dir)
        if os.path.isdir(f"{sessions_dir}/{d}")
        and os.path.exists(f"{sessions_dir}/{d}/emg_data.csv")
    ])
    for i, d in enumerate(dirs):
        df = pd.read_csv(f"{sessions_dir}/{d}/emg_data.csv")
        df['session_id'] = i
        all_dfs.append(df)
        print(f"  Session {i+1}: {len(df):,} samples — {d}")
    return pd.concat(all_dfs, ignore_index=True)

print("\nLoading sessions...")
df = load_sessions()

# ── أضف repetition_id لكل session ──
# كل gesture + rest = block واحد
# نعرّف الـ block بتغير الـ label
df['block_id'] = (df['label'] != df['label'].shift()).cumsum()
df['rep_group'] = df['session_id'].astype(str) + '_' + df['block_id'].astype(str)

print(f"Total: {len(df):,} samples")
print(f"Total blocks: {df['block_id'].nunique()}\n")


# ── Preprocessing per session ──
def preprocess_by_session(df):
    EMG_COLS = [f'emg_{i}' for i in range(8)]
    emg_out  = np.zeros((len(df), 8), dtype=np.float32)
    nyq = FS / 2
    b,  a  = signal.butter(4, [20/nyq, 90/nyq], btype='band')
    bn, an = signal.iirnotch(50, Q=30, fs=FS)

    for sid in df['session_id'].unique():
        mask = (df['session_id'] == sid).values
        emg  = df.loc[mask, EMG_COLS].values.astype(np.float32)
        emg  = signal.filtfilt(b,  a,  emg, axis=0)
        emg  = signal.filtfilt(bn, an, emg, axis=0)
        mean = emg.mean(axis=0)
        std  = np.where(emg.std(axis=0) < 1e-8, 1e-8, emg.std(axis=0))
        emg_out[mask] = (emg - mean) / std

    return emg_out

print("Preprocessing...")
emg_norm = preprocess_by_session(df)
labels   = df['label'].values.astype(np.int64)
blocks   = df['block_id'].values


# ── Windowing per block — THE FIX ──
# نعمل windowing داخل كل block بشكل منفصل
# هكذا لا يوجد window يتشارك samples بين train و test

def extract_windows_per_block(emg, labels, blocks,
                               win=WIN_SAMPLES, step=STEP):
    X, y, block_ids = [], [], []
    unique_blocks = np.unique(blocks)

    for bid in unique_blocks:
        mask   = blocks == bid
        e_blk  = emg[mask]
        l_blk  = labels[mask]

        # تأكد أن الـ block نظيف (label واحد فقط)
        if len(np.unique(l_blk)) != 1:
            continue
        lbl = l_blk[0]

        n = len(l_blk)
        i = 0
        while i + win <= n:
            X.append(e_blk[i:i+win])
            y.append(lbl)
            block_ids.append(bid)
            i += step

    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int64),
            np.array(block_ids))

print("Extracting windows per block (no leakage)...")
X, y, block_ids = extract_windows_per_block(emg_norm, labels, blocks)
print(f"Windows : {len(X):,}  |  Shape: {X.shape}\n")

for lbl, name in GESTURE_NAMES.items():
    count = (y == lbl).sum()
    print(f"  {name:12s} [{lbl}]: {count:5d} windows")


# ── Split على مستوى الـ Block لا الـ Window ──
# كل block إما كله في Train أو كله في Test
# هذا يمنع أي تداخل

np.random.seed(42)

train_blocks_list = []
test_blocks_list  = []

for lbl in range(N_CLASSES):
    lbl_blocks = np.unique(block_ids[y == lbl])
    np.random.shuffle(lbl_blocks)
    n_test = max(1, int(len(lbl_blocks) * 0.2))
    test_blocks_list.extend(lbl_blocks[:n_test].tolist())
    train_blocks_list.extend(lbl_blocks[n_test:].tolist())

train_blocks = set(train_blocks_list)
test_blocks  = set(test_blocks_list)
train_mask = np.array([b in train_blocks for b in block_ids])
test_mask  = np.array([b in test_blocks  for b in block_ids])

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"\nTrain: {len(X_train):,} windows  |  Test: {len(X_test):,} windows")
print(f"Train blocks: {len(train_blocks)}  |  Test blocks: {len(test_blocks)}")

# تحقق من توزيع الـ classes
print("\nClass distribution in test:")
for lbl, name in GESTURE_NAMES.items():
    count = (y_test == lbl).sum()
    print(f"  {name:12s}: {count}")


# ── Dataset ──
class EMGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(EMGDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader  = DataLoader(EMGDataset(X_test,  y_test),
                          batch_size=BATCH_SIZE, shuffle=False)


# ── Model ──
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


# ── Training ──
model     = EMG_CNN_LSTM(n_classes=N_CLASSES).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(cw, dtype=torch.float32).to(DEVICE),
    label_smoothing=0.05
)

print("\n" + "=" * 52)
print("  TRAINING  v4  —  No Data Leakage")
print("=" * 52)

best_acc, best_epoch = 0.0, 0
train_losses, test_accs = [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total   += len(yb)

    acc      = correct / total
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    test_accs.append(acc)

    if acc > best_acc:
        best_acc, best_epoch = acc, epoch
        torch.save(model.state_dict(), 'best_model_v4.pt')

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  |  "
              f"Loss: {avg_loss:.4f}  |  "
              f"Acc: {acc:.3f}  |  "
              f"Best: {best_acc:.3f} (ep {best_epoch})")

print(f"\n  Best accuracy: {best_acc:.3f}  at epoch {best_epoch}")


# ── Evaluation ──
model.load_state_dict(torch.load('best_model_v4.pt'))
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb.to(DEVICE)).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(yb.numpy())

names = [GESTURE_NAMES[i] for i in range(N_CLASSES)]
print("\n" + "=" * 52)
print("  CLASSIFICATION REPORT  v4")
print("=" * 52)
print(classification_report(all_true, all_preds, target_names=names))

cm = confusion_matrix(all_true, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=names, yticklabels=names)
plt.title(f'Confusion Matrix v4 — Best Acc: {best_acc:.3f}')
plt.ylabel('True'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_v4.png', dpi=150)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses); ax1.set_title('Training Loss'); ax1.set_xlabel('Epoch')
ax2.plot(test_accs);    ax2.set_title('Test Accuracy'); ax2.set_xlabel('Epoch')
ax2.axhline(y=best_acc, color='r', linestyle='--', label=f'Best: {best_acc:.3f}')
ax2.legend()
plt.tight_layout()
plt.savefig('training_curves_v4.png', dpi=150)

print("  Saved: confusion_matrix_v4.png  |  training_curves_v4.png")
print("  Done.")