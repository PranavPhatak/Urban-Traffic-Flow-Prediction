from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path("../dataset/raw/flows")
PROCESSED_DIR = Path("../dataset/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_FILES = {
    "GD0501_B": "GD0501_B.csv",
    "GD0501_C": "GD0501_C.csv",
    "GD0501_D": "GD0501_D.csv",
}
MAX_INTERPOLATE_HOURS = 3   
FLAT_RUN_FLAG_HOURS = 6     

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

full_index = pd.date_range(merged["datetime"].min(), merged["datetime"].max(), freq="h")
clean = merged.set_index("datetime").reindex(full_index)
clean.index.name = "datetime"

n_missing = clean[SENSORS[0]].isna().sum()
print(f"\nReindexed to {len(full_index):,} hourly rows ({n_missing:,} were missing from source)")

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

clean = clean.reset_index()
output_path = PROCESSED_DIR / "GD0501_clean.csv"
clean.to_csv(output_path, index=False)

print(f"\nSaved cleaned dataset -> {output_path}")
print(f"Total rows: {len(clean):,}")
print(f"Rows still missing sensor data (unfilled long gaps): {clean[SENSORS].isna().any(axis=1).sum():,}")
