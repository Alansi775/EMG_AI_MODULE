from huggingface_hub import upload_file, upload_folder
from datetime import datetime
import os

REPO_ID = "malansi/EMG-Gesture-Dataset"

# ارفع كل الجلسات
print("Uploading sessions...\n")

sessions_dir = "sessions"
for session in sorted(os.listdir(sessions_dir)):
    session_path = f"{sessions_dir}/{session}"
    if not os.path.isdir(session_path):
        continue
    
    for fname in ["emg_data.csv", "meta.json"]:
        fpath = f"{session_path}/{fname}"
        if not os.path.exists(fpath):
            continue
        try:
            upload_file(
                path_or_fileobj=fpath,
                path_in_repo=f"data/{session}/{fname}",
                repo_id=REPO_ID,
                repo_type="dataset",
                commit_message=f"Add {session}/{fname}",
            )
            print(f"  ✅  {session}/{fname}")
        except Exception as e:
            print(f"  ❌  {session}/{fname}: {e}")

print(f"\n✅ Done → https://huggingface.co/{REPO_ID}")
