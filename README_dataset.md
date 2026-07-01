---
language: en
tags:
  - emg
  - gesture-recognition
  - myo-armband
  - biosignals
  - time-series
license: mit
task_categories:
  - time-series-classification
---

# EMG Gesture Dataset — Myo Armband

Real EMG recordings from Myo Armband for hand gesture recognition.
Collected to train the [EMG-Gesture-Recognition](https://huggingface.co/malansi/EMG-Gesture-Recognition) model.

## Dataset Summary

| Property | Value |
|----------|-------|
| Sensor | Myo Armband |
| Channels | 8 EMG channels |
| Sampling Rate | 200 Hz |
| Sessions | 4 |
| Gestures | 6 |
| Rounds per gesture | 6 |
| Total samples | ~250,000 |

## Gestures

| Label | Gesture | Description |
|-------|---------|-------------|
| 0 | rest | Hand relaxed, flat on table |
| 1 | fist | Hand closed into tight fist |
| 2 | open_hand | Hand fully open, fingers spread |
| 3 | wave_in | Wrist bent inward toward body |
| 4 | wave_out | Wrist bent outward, fingers closed |
| 5 | pinch | Index finger and thumb pinched |

## Recording Protocol
For each gesture:

Countdown (4s)        ← not recorded
Hold gesture (5s)     ← recorded
Relax countdown (8s)  ← not recorded
Rest (5s)             ← recorded
Repeat × 6 rounds


## File Structure
data/
session_20260629_135035/
emg_data.csv   ← EMG + labels
meta.json      ← session metadata
session_20260629_141407/
...
session_20260629_161304/
...
session_20260629_174158/
...

## CSV Columns

| Column | Description |
|--------|-------------|
| emg_0 … emg_7 | Raw EMG signal (8 channels) |
| timestamp | Unix timestamp |
| label | Gesture label (0–5) |
| gesture | Gesture name |


## Demo

Watch the model trained on this dataset running in real-time:

https://youtube.com/shorts/uGS7Rv67E7w

## Notes

- Session 1 contains a minor labeling error in the last repetition of pinch
- All sessions used for training the final model
- Normalization stats: see model repo

## Related

- Model: [malansi/EMG-Gesture-Recognition](https://huggingface.co/malansi/EMG-Gesture-Recognition)

## Author

Mohammed Alansi
AI & Robotics Research — EMG-Based Prosthetic Control
