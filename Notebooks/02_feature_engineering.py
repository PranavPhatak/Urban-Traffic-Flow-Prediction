"""
Step 2: Feature engineering for GA0151 intersection LSTM forecasting.

Deliberately leaner than the original Feature_Engineering.ipynb, and fixes two
correctness issues found in that notebook:

1. LEAKAGE: the original `{sensor}_diff_1` / `_diff_24` were defined as
   `current_value - lag`, which algebraically reconstructs the current-row
   target with 100% accuracy (verified on this dataset). Not included here.

2. GAP BRIDGING: any `.shift()` / `.rolling()` computed on row position alone
   is wrong once you know ~4% of hours are missing (see 01_data_cleaning.py).
   A "lag_1" for the first row after a 35-day gap must NOT be the last row
   before the gap. This script computes every lag/rolling/target feature
   PER CONTIGUOUS SEGMENT (a run of truly consecutive hours), using a
   `segment_id` derived from datetime, so no feature ever crosses a gap.

Feature philosophy: for a recurrent model (LSTM), the sequence window itself
already encodes lag/rolling-window information -- feeding the same history a
second time as parallel hand-crafted lag columns is redundant and just adds
collinear, overfitting-prone dimensions. Kept:
  - raw flow (becomes the input sequence)
  - cyclical time encodings + is_weekend (the LSTM can't infer calendar
    structure from 48 raw hours alone as reliably as an explicit encoding)
  - rolling_std_24 per sensor (local volatility -- meaningfully different
    information than what a window of raw values gives the LSTM directly)
Dropped: lag_*, rolling_mean_*, diff_* (redundant with the sequence, and
diff_* was leaky besides).

Output: dataset/processed/GA0151_features.csv, including `segment_id` and
`datetime` -- the training script uses `segment_id` to make sure no window
crosses a data gap.
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED_DIR = Path("../dataset/processed")
SENSORS = ["GA0151_A", "GA0151_C", "GA0151_D"]
HORIZON = 1          # forecast horizon in hours; e.g. 24 for next-day-same-hour
ROLLING_STD_WINDOW = 24

# ----------------------------------------------------------------------------
# 1. Load cleaned data
# ----------------------------------------------------------------------------
df = pd.read_csv(PROCESSED_DIR / "GA0151_clean.csv", parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

# Rows that are still NaN after cleaning are unfilled long gaps -- they can't
# be used as inputs, but keeping them in place (for now) is what lets us
# detect segment boundaries correctly in the next step.
gap_row = df[SENSORS].isna().any(axis=1)

# ----------------------------------------------------------------------------
# 2. Segment ID: increments every time there's a break in true hourly
#    continuity OR a still-missing (unfilled) row. Everything downstream is
#    computed within a segment, never across one.
# ----------------------------------------------------------------------------
hour_gap = df["datetime"].diff() != pd.Timedelta(hours=1)
new_segment = hour_gap | gap_row | gap_row.shift(fill_value=False)
df["segment_id"] = new_segment.cumsum()

# Drop the still-missing rows themselves now that they've served their purpose
# of marking segment boundaries.
df = df[~gap_row].reset_index(drop=True)

n_segments = df["segment_id"].nunique()
seg_lengths = df.groupby("segment_id").size()
print(f"{n_segments} contiguous segments after removing unfilled gap rows")
print(f"Segment lengths -> min: {seg_lengths.min()}, median: {seg_lengths.median():.0f}, "
      f"max: {seg_lengths.max()}")

# ----------------------------------------------------------------------------
# 3. Calendar / cyclical features -- safe globally, no gap risk
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# 4. Rolling std (local volatility) -- computed per segment, shift(1) first so
#    it only ever sees strictly past values.
# ----------------------------------------------------------------------------
for sensor in SENSORS:
    df[f"{sensor}_rolling_std_24"] = (
        df.groupby("segment_id")[sensor]
        .transform(lambda s: s.shift(1).rolling(ROLLING_STD_WINDOW).std())
    )

# ----------------------------------------------------------------------------
# 5. Contemporaneous neighbor-sensor mean -- same timestamp, not shifted, so
#    no gap risk. (Note: only valid as a feature if all 3 sensors' current
#    reading is genuinely available at prediction time -- true for a fused
#    nowcasting/sensor-network setup; revisit if sensors are meant to be
#    predicted independently with no shared real-time feed.)
# ----------------------------------------------------------------------------
for sensor in SENSORS:
    others = [s for s in SENSORS if s != sensor]
    df[f"{sensor}_neighbor_mean"] = df[others].mean(axis=1)

# ----------------------------------------------------------------------------
# 6. Forward-looking targets -- per segment, so the target for the last few
#    rows of a segment (where t+HORIZON would fall in the next segment, i.e.
#    after a gap) is correctly left as NaN rather than silently pulled from
#    an unrelated time period.
# ----------------------------------------------------------------------------
for sensor in SENSORS:
    df[f"{sensor}_target"] = (
        df.groupby("segment_id")[sensor].transform(lambda s: s.shift(-HORIZON))
    )

# ----------------------------------------------------------------------------
# 7. Drop rows with any NaN (segment edges: start of rolling window, end of
#    target horizon)
# ----------------------------------------------------------------------------
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

output_path = PROCESSED_DIR / "GA0151_features.csv"
df.to_csv(output_path, index=False)

print(f"\nSaved -> {output_path}")
print(f"Shape: {df.shape}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"Target columns: {TARGET_COLS}")
