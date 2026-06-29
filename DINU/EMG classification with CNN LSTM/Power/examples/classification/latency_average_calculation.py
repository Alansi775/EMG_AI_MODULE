import pandas as pd
import re

# Path to your uploaded file
file_path = "metrics_log.txt"

rows = []
with open(file_path, "r") as f:
    for line in f:
        if re.match(r"^\s*[A-Za-z_]+\s", line) and not line.startswith("Gesture"):
            parts = line.split()
            if len(parts) >= 6:
                gesture = parts[0]
                try:
                    t_emg, t_pred, t_servo, lat_pred, lat_servo = map(float, parts[1:6])
                    rows.append({
                        "Gesture": gesture,
                        "lat_pred": lat_pred,
                        "lat_servo": lat_servo
                    })
                except ValueError:
                    continue

# Convert to DataFrame
df = pd.DataFrame(rows)

# Compute aggregated statistics
agg_df = df.groupby("Gesture").agg(
    avg_pred=("lat_pred", "mean"),
    std_pred=("lat_pred", "std"),
    avg_servo=("lat_servo", "mean"),
    std_servo=("lat_servo", "std"),
    samples=("lat_pred", "count")
).reset_index()

# Round for readability
agg_df = agg_df.round({
    "avg_pred": 4,
    "std_pred": 4,
    "avg_servo": 4,
    "std_servo": 4
})

# Sort alphabetically
agg_df = agg_df.sort_values("Gesture")

# Display formatted output
print("\nAverage, Std Deviation, and Sample Count per Gesture:")
print("=" * 80)
print(agg_df.to_string(index=False))
