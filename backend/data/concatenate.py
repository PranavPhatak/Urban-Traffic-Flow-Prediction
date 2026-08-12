from pathlib import Path
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATASET_DIR = Path("dataset")
RAW_DIR = DATASET_DIR / "raw" / "flows"
PROCESSED_DIR = DATASET_DIR / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Sensors to combine
# --------------------------------------------------

sensor_files = [
    "GA0151_A.csv",
    "GA0151_C.csv",
    "GA0151_D.csv"
]


# --------------------------------------------------
# Read sensor data
# --------------------------------------------------

dataframes = []

for filename in sensor_files:

    file_path = RAW_DIR / filename

    df = pd.read_csv(file_path)

    # Extract sensor ID
    sensor_id = Path(filename).stem

    # Keep only required columns
    df = df[["date", "time", "flow"]]

    # Rename flow column to sensor ID
    df = df.rename(
        columns={"flow": sensor_id}
    )

    dataframes.append(df)

    print(f"Loaded {sensor_id}: {len(df):,} rows")


# --------------------------------------------------
# Merge the three sensors
# --------------------------------------------------

combined_df = dataframes[0]

for df in dataframes[1:]:

    combined_df = combined_df.merge(
        df,
        on=["date", "time"],
        how="outer"
    )


# --------------------------------------------------
# Sort by date and time
# --------------------------------------------------

combined_df = combined_df.sort_values(
    ["date", "time"]
).reset_index(drop=True)


# --------------------------------------------------
# Column order
# --------------------------------------------------

combined_df = combined_df[
    [
        "date",
        "time",
        "GA0151_A",
        "GA0151_C",
        "GA0151_D"
    ]
]


# --------------------------------------------------
# Save processed dataset
# --------------------------------------------------

output_path = PROCESSED_DIR / "GA0151_intersection.csv"

combined_df.to_csv(
    output_path,
    index=False
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nConcatenation completed successfully!")

print(f"Output: {output_path}")
print(f"Total rows: {len(combined_df):,}")

print("\nColumns:")
print(combined_df.columns.tolist())

print("\nMissing values:")
print(combined_df.isnull().sum())

print("\nFirst 10 rows:")
print(combined_df.head(10))