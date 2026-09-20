"""
Step 2: Feature engineering for GA0151 intersection LSTM forecasting.

CHANGE from previous version: excludes the COVID lockdown window from the
data BEFORE segment_id is computed. Yearly mean traffic flow is 25-40% lower
in 2020-2021 than every other year (2019, 2022, 2023) for all three sensors
-- that's a genuine regime shift (empty roads during lockdowns), not noise.
The original training set (Oct 2019 - Jul 2022) had ~half its rows from this
anomalous period while val/test (Jul 2022 onward) are entirely post-pandemic,
which likely explains why val loss plateaued early while train loss kept
improving during training.

This filter reuses the SAME segment-break machinery already used for real
sensor-outage gaps: dropping rows here creates a jump in `datetime` that
`hour_gap` detects downstream, so no lag/rolling/target feature or LSTM
window will bridge across the excluded period. No other logic changes.

Test this one change in isolation before touching model architecture --
see 03_train_lstm.py's "Next steps" comment.
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED_DIR = Path("../dataset/processed")
SENSORS = ["GD0151_B", "GD0151_C", "GD0151_D"]
HORIZON = 1          # forecast horizon in hours; e.g. 24 for next-day-same-hour
ROLLING_STD_WINDOW = 24

# COVID lockdown exclusion window. Chosen from the visible yearly-mean dip
# (UK national lockdown started 2020-03-23; Scotland lifted most remaining
# restrictions by early 2022). Adjust if you want a tighter/looser cut --
# widen it if the train/val gap is still large after this change, narrow it
# if removing it barely changes the result and you want the data back.
EXCLUDE_COVID_PERIOD = True
COVID_EXCLUDE_START = "2020-03-01"
COVID_EXCLUDE_END = "2021-12-31"

# NEW: log1p-transform the raw flow signal before anything is derived from it.
# Traffic counts are classic heavy-tailed/skewed data -- GA0151_A in
# particular has 516 IQR-outlier spikes out of ~33k rows (vs 1 for C, 31 for
# D) and the highest coefficient of variation of the three sensors. Under
# plain MSE, a spike to 200 produces a squared-error gradient ~400x a normal
# hour, which pulls training toward chasing noise. log1p compresses that
# without discarding the data. Applying it HERE (immediately after load,
# before rolling_std/neighbor_mean/target are computed) means every
# downstream feature and the target itself are consistently in log space --
# no mismatch between what train and eval see. Predictions are inverted with
# expm1 in 03_train_lstm.py before computing MAE/RMSE/R2, so all reported
# metrics stay in real vehicles/hour.
LOG_TRANSFORM_FLOW = True

# ----------------------------------------------------------------------------
# 1. Load cleaned data
# ----------------------------------------------------------------------------
df = pd.read_csv(PROCESSED_DIR / "GA0151_clean.csv", parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

# ----------------------------------------------------------------------------
# 1b. NEW: drop the COVID-affected window. This happens before segment_id is
#     computed, so the resulting gap in `datetime` is picked up by the same
#     `hour_gap` check used for real sensor outages -- no separate logic
#     needed, and no feature/window will ever bridge across it.
# ----------------------------------------------------------------------------
if EXCLUDE_COVID_PERIOD:
    before = len(df)
    covid_mask = (df["datetime"] >= COVID_EXCLUDE_START) & (df["datetime"] <= COVID_EXCLUDE_END)
    df = df[~covid_mask].reset_index(drop=True)
    print(f"Excluded COVID window {COVID_EXCLUDE_START} -> {COVID_EXCLUDE_END}: "
          f"removed {before - len(df):,} rows ({len(df):,} remain)")

# ----------------------------------------------------------------------------
# 1c. NEW: log1p-transform the raw flow columns. Done before gap_row/segment
#     logic (which only cares about NaN positions -- log1p(NaN) stays NaN, so
#     this doesn't interact with gap detection) and before every feature that
#     derives from SENSORS below, so rolling_std, neighbor_mean, and the
#     target are all computed consistently in log space.
# ----------------------------------------------------------------------------
if LOG_TRANSFORM_FLOW:
    df[SENSORS] = np.log1p(df[SENSORS])

# Rows that are still NaN after cleaning are unfilled long gaps -- they can't
# be used as inputs, but keeping them in place (for now) is what lets us
# detect segment boundaries correctly in the next step.
gap_row = df[SENSORS].isna().any(axis=1)

# ----------------------------------------------------------------------------
# 2. Segment ID: increments every time there's a break in true hourly
#    continuity OR a still-missing (unfilled) row. Everything downstream is
#    computed within a segment, never across one. The COVID exclusion above
#    now also triggers a segment break here, automatically.
# ----------------------------------------------------------------------------
hour_gap = df["datetime"].diff() != pd.Timedelta(hours=1)
new_segment = hour_gap | gap_row | gap_row.shift(fill_value=False)
df["segment_id"] = new_segment.cumsum()

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
#    no gap risk.
# ----------------------------------------------------------------------------
for sensor in SENSORS:
    others = [s for s in SENSORS if s != sensor]
    df[f"{sensor}_neighbor_mean"] = df[others].mean(axis=1)

# ----------------------------------------------------------------------------
# 6. Forward-looking targets -- per segment, so the target for the last few
#    rows of a segment is correctly left as NaN rather than silently pulled
#    from an unrelated time period.
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
