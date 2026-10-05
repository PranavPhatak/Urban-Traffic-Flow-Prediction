from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED_DIR = Path("../dataset/processed")
SENSORS = ["GD0501_B", "GD0501_C", "GD0501_D"]
HORIZON = 1          
ROLLING_STD_WINDOW = 24

EXCLUDE_COVID_PERIOD = True
COVID_EXCLUDE_START = "2020-03-01"
COVID_EXCLUDE_END = "2021-12-31"

LOG_TRANSFORM_FLOW = True

df = pd.read_csv(PROCESSED_DIR / "GD0501_clean.csv", parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

if EXCLUDE_COVID_PERIOD:
    before = len(df)
    covid_mask = (df["datetime"] >= COVID_EXCLUDE_START) & (df["datetime"] <= COVID_EXCLUDE_END)
    df = df[~covid_mask].reset_index(drop=True)
    print(f"Excluded COVID window {COVID_EXCLUDE_START} -> {COVID_EXCLUDE_END}: "
          f"removed {before - len(df):,} rows ({len(df):,} remain)")

if LOG_TRANSFORM_FLOW:
    df[SENSORS] = np.log1p(df[SENSORS])

gap_row = df[SENSORS].isna().any(axis=1)

hour_gap = df["datetime"].diff() != pd.Timedelta(hours=1)
new_segment = hour_gap | gap_row | gap_row.shift(fill_value=False)
df["segment_id"] = new_segment.cumsum()

df = df[~gap_row].reset_index(drop=True)

n_segments = df["segment_id"].nunique()
seg_lengths = df.groupby("segment_id").size()
print(f"{n_segments} contiguous segments after removing unfilled gap rows")
print(f"Segment lengths -> min: {seg_lengths.min()}, median: {seg_lengths.median():.0f}, "
      f"max: {seg_lengths.max()}")

df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

for sensor in SENSORS:
    df[f"{sensor}_rolling_std_24"] = (
        df.groupby("segment_id")[sensor]
        .transform(lambda s: s.shift(1).rolling(ROLLING_STD_WINDOW).std())
    )

for sensor in SENSORS:
    others = [s for s in SENSORS if s != sensor]
    df[f"{sensor}_neighbor_mean"] = df[others].mean(axis=1)

for sensor in SENSORS:
    df[f"{sensor}_target"] = (
        df.groupby("segment_id")[sensor].transform(lambda s: s.shift(-HORIZON))
    )

before = len(df)
df = df.dropna().reset_index(drop=True)
print(f"\nDropped {before - len(df)} edge rows (rolling-window warmup / target horizon at segment boundaries)")

FEATURE_COLS = (
    SENSORS
    + [f"{s}_rolling_std_24" for s in SENSORS]
    + [f"{s}_neighbor_mean" for s in SENSORS]
    + ["hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos", "is_weekend"]
)
TARGET_COLS = [f"{s}_target" for s in SENSORS]
KEEP_COLS = ["datetime", "segment_id"] + FEATURE_COLS + TARGET_COLS
df = df[KEEP_COLS]

output_path = PROCESSED_DIR / "GD0501_features.csv"
df.to_csv(output_path, index=False)

print(f"\nSaved -> {output_path}")
print(f"Shape: {df.shape}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"Target columns: {TARGET_COLS}")
