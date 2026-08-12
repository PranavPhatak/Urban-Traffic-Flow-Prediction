from pathlib import Path
import pandas as pd

DATASET_DIR = Path("dataset")
RAW_DIR = DATASET_DIR / "raw" / "flows"
PROCESSED_DIR = DATASET_DIR / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

sensor_files = [
    "GA0151_A.csv",
    "GA0151_C.csv",
    "GA0151_D.csv"
]

dataframes = []

for filename in sensor_files:
    file_path = RAW_DIR / filename
    df = pd.read_csv(file_path)
    sensor_id = Path(filename).stem
    df = df[["date", "time", "flow"]]
    df = df.rename(
        columns={"flow": sensor_id}
    )
    dataframes.append(df)
    print(f"Loaded {sensor_id}: {len(df):,} rows")

combined_df = dataframes[0]

for df in dataframes[1:]:
    combined_df = combined_df.merge(
        df,
        on=["date", "time"],
        how="outer"
    )

combined_df = combined_df.sort_values(
    ["date", "time"]
).reset_index(drop=True)

combined_df = combined_df[
    [
        "date",
        "time",
        "GA0151_A",
        "GA0151_C",
        "GA0151_D"
    ]
]

output_path = PROCESSED_DIR / "GA0151_intersection.csv"

combined_df.to_csv(
    output_path,
    index=False
)

print("\nConcatenation completed successfully!")

print(f"Output: {output_path}")
print(f"Total rows: {len(combined_df):,}")

print("\nColumns:")
print(combined_df.columns.tolist())

print("\nMissing values:")
print(combined_df.isnull().sum())

print("\nFirst 10 rows:")
print(combined_df.head(10))