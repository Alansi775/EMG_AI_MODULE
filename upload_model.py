from huggingface_hub import HfApi, upload_file
from datetime import datetime

HF_USERNAME = "malansi"
REPO_NAME   = "EMG-Gesture-Recognition"
REPO_ID     = f"{HF_USERNAME}/{REPO_NAME}"

api = HfApi()

files = [
    ("best_model_v4.pt",        "models/best_model_v4.pt"),
    ("norm_mean.npy",           "models/norm_mean.npy"),
    ("norm_std.npy",            "models/norm_std.npy"),
    ("confusion_matrix_v4.png", "results/confusion_matrix_v4.png"),
    ("training_curves_v4.png",  "results/training_curves_v4.png"),
    ("train.py",                "code/train.py"),
    ("realtime.py",             "code/realtime.py"),
    ("guided_test.py",          "code/guided_test.py"),
]

print(f"Uploading to {REPO_ID}...\n")
for local, remote in files:
    try:
        upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=f"Update {local} — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        )
        print(f"  ✅  {local}")
    except Exception as e:
        print(f"  ❌  {local}: {e}")

print(f"\n✅ Done → https://huggingface.co/{REPO_ID}")
