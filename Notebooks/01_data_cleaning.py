"""
Step 1: Data cleaning for GA0151 intersection (sensors A, C, D).

What this does beyond concatenate.py:
- Merges the 3 raw sensor files (same logic as concatenate.py).
- Reindexes onto a COMPLETE hourly datetime grid. The raw merge only contains
  timestamps that exist in the source files -- it does NOT guarantee every
  hour is present. This dataset has 1,420 missing hours out of 35,064
  expected, in 26 gap blocks, the longest being 35 days (2019-10-16 to
  2019-11-20) and 14 days (2023-07-12 to 2023-07-26). Those are almost
  certainly sensor/logging outages, not zero traffic.
- Interpolates only SHORT gaps (<= MAX_INTERPOLATE_HOURS, default 3h) with
  time-based linear interpolation -- a few missing hours next to known values
  is a reasonable imputation. Longer gaps are left as NaN on purpose: do not
  fabricate weeks of traffic data.
- Flags (does not delete) suspicious flat runs -- e.g. GA0151_A has a 20-hour
  run of exact 0 -- since these look more like sensor faults than real zero
  traffic, but deleting them automatically risks discarding a real road
  closure. Flagged spans are written to a review CSV.

Output: dataset/processed/GA0151_clean.csv with columns
  datetime, GA0151_A, GA0151_C, GA0151_D, was_imputed
Rows inside long (unfilled) gaps are still written with NaN sensor values --
the feature-engineering step uses `datetime` continuity, not row position, so
it will never treat a pre-gap row and a post-gap row as consecutive hours.
"""

from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path("../dataset/raw/flows")
PROCESSED_DIR = Path("../dataset/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_FILES = {
    "GD0151_B": "Gd0151_B.csv",
    "GD0151_C": "GD0151_C.csv",
    "GD0151_D": "GD0151_D.csv",
}
MAX_INTERPOLATE_HOURS = 3   # only bridge gaps this short or shorter
FLAT_RUN_FLAG_HOURS = 6     # flag identical-value runs at least this long

# ----------------------------------------------------------------------------
# 1. Load and merge raw sensors (same idea as concatenate.py)
# ----------------------------------------------------------------------------
frames = []
for sensor_id, filename in SENSOR_FILES.items():
    df = pd.read_csv(RAW_DIR / filename)[["date", "time", "flow"]]
    df = df.rename(columns={"flow": sensor_id})
    frames.append(df)
    print(f"Loaded {sensor_id}: {len(df):,} rows")

merged = frames[0]
for df in frames[1:]:
    merged = merged.merge(df, on=["date", "time"], how="outer")

merged["datetime"] = pd.to_datetime(merged["date"]) + pd.to_timedelta(merged["time"], unit="h")
merged = merged.drop(columns=["date", "time"]).sort_values("datetime").reset_index(drop=True)

dupes = merged["datetime"].duplicated().sum()
if dupes:
    print(f"WARNING: {dupes} duplicate timestamps found, keeping first occurrence")
    merged = merged.drop_duplicates(subset="datetime", keep="first")

SENSORS = list(SENSOR_FILES.keys())

# ----------------------------------------------------------------------------
# 2. Reindex onto a complete hourly grid
# ----------------------------------------------------------------------------
full_index = pd.date_range(merged["datetime"].min(), merged["datetime"].max(), freq="h")
clean = merged.set_index("datetime").reindex(full_index)
clean.index.name = "datetime"

n_missing = clean[SENSORS[0]].isna().sum()
print(f"\nReindexed to {len(full_index):,} hourly rows ({n_missing:,} were missing from source)")

# ----------------------------------------------------------------------------
# 3. Interpolate only short gaps; leave long gaps as NaN
# ----------------------------------------------------------------------------
clean["was_imputed"] = False

for sensor in SENSORS:
    is_na = clean[sensor].isna()
    # identify contiguous NaN run lengths
    run_id = (is_na != is_na.shift()).cumsum()
    run_lengths = is_na.groupby(run_id).transform("sum")
    short_gap_mask = is_na & (run_lengths <= MAX_INTERPOLATE_HOURS)

    interpolated = clean[sensor].interpolate(method="time", limit=MAX_INTERPOLATE_HOURS)
    clean.loc[short_gap_mask, sensor] = interpolated.loc[short_gap_mask]
    clean.loc[short_gap_mask, "was_imputed"] = True

    still_missing = clean[sensor].isna().sum()
    print(f"{sensor}: filled {short_gap_mask.sum()} short-gap hours,{still_missing} hours remain missing (long gaps, left as NaN)")

# ----------------------------------------------------------------------------
# 4. Flag (don't remove) suspicious flat/stuck runs for manual review
# ----------------------------------------------------------------------------
flagged_spans = []
for sensor in SENSORS:
    vals = clean[sensor]
    same_as_prev = vals == vals.shift()
    run_id = (~same_as_prev.fillna(False)).cumsum()
    for _, group in vals.groupby(run_id):
        if len(group) >= FLAT_RUN_FLAG_HOURS and group.notna().all():
            flagged_spans.append({
                "sensor": sensor,
                "start": group.index.min(),
                "end": group.index.max(),
                "length_hours": len(group),
                "value": group.iloc[0],
            })

flags_df = pd.DataFrame(flagged_spans).sort_values("length_hours", ascending=False)
flags_path = PROCESSED_DIR / "flagged_flat_runs_for_review.csv"
flags_df.to_csv(flags_path, index=False)
print(f"\nFlagged {len(flags_df)} flat/stuck runs >= {FLAT_RUN_FLAG_HOURS}h for manual review -> {flags_path}")
if len(flags_df):
    print(flags_df.head(10).to_string(index=False))

# ----------------------------------------------------------------------------
# 5. Save
# ----------------------------------------------------------------------------
clean = clean.reset_index()
output_path = PROCESSED_DIR / "GA0151_clean.csv"
clean.to_csv(output_path, index=False)

print(f"\nSaved cleaned dataset -> {output_path}")
print(f"Total rows: {len(clean):,}")
print(f"Rows still missing sensor data (unfilled long gaps): {clean[SENSORS].isna().any(axis=1).sum():,}")
