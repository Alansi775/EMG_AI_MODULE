---
language: en
tags:
  - emg
  - gesture-recognition
  - cnn-lstm
  - myo-armband
  - prosthetics
  - real-time
license: mit
---

# EMG Gesture Recognition — Myo Armband

Real-time hand gesture classification from EMG signals using CNN + LSTM.
Built for prosthetic arm control and drone navigation.

## Results (Real-Time on Myo Armband)

| Gesture    | Accuracy |
|------------|----------|
| rest       | 100%     |
| wave_in    | 100%     |
| fist       | 98%      |
| pinch      | 97%      |
| open_hand  | 92%      |
| wave_out   | 85%      |
| **Average**| **95%**  |

## Model Architecture
Input: 150 samples × 8 EMG channels (750ms @ 200Hz)
↓
CNN Block:
Conv1D(64) → BatchNorm → ReLU
Conv1D(128) → BatchNorm → ReLU → MaxPool
Conv1D(256) → BatchNorm → ReLU → MaxPool
↓
Bidirectional LSTM (128 units × 2 layers)
↓
Dense(128) → Dense(6)
↓
Output: 6 gesture classes

## Hardware

- **Sensor**: Myo Armband — 8 EMG channels, 200Hz
- **Platform**: Apple Silicon (MPS) / CPU
- **Latency**: < 100ms end-to-end

## Dataset

- 4 recording sessions
- 6 gestures × 6 rounds each
- ~250,000 EMG samples total
- Per-subject global normalization

## Key Challenges Solved

1. **Data Leakage** — block-level train/test split (no overlapping windows)
2. **Normalization Mismatch** — fixed global stats saved and reused at inference
3. **Class Imbalance** — weighted CrossEntropy loss
4. **Real-time Stability** — hysteresis voting system for actuator control

## Files

| File | Description |
|------|-------------|
| `models/best_model_v4.pt` | Trained PyTorch model weights |
| `models/norm_mean.npy` | Normalization mean (required at inference) |
| `models/norm_std.npy` | Normalization std (required at inference) |
| `results/confusion_matrix_v4.png` | Confusion matrix |
| `results/training_curves_v4.png` | Training loss & accuracy curves |
| `code/train.py` | Full training pipeline |
| `code/realtime.py` | Real-time inference with Myo |
| `code/guided_test.py` | Guided accuracy evaluation |


## Demo

Watch the model running in real-time on the Myo Armband:

https://youtube.com/shorts/uGS7Rv67E7w

## Usage

```python
import torch
import numpy as np

# Load model
model = EMG_CNN_LSTM(n_channels=8, n_classes=6)
model.load_state_dict(torch.load('models/best_model_v4.pt'))
model.eval()

# Load normalization stats
norm_mean = np.load('models/norm_mean.npy')
norm_std  = np.load('models/norm_std.npy')

# Normalize input (150 samples × 8 channels)
window = (window - norm_mean) / norm_std

# Predict
x = torch.tensor(window.T.copy(), dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)
    pred  = probs.argmax().item()
```

## Author

Mohammed Alansi
AI & Robotics Research — EMG-Based Prosthetic Control
