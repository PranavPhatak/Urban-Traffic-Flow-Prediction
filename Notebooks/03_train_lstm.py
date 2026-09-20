"""
Step 3: Multi-output LSTM for GA0151 intersection traffic-flow forecasting.

Reads dataset/processed/GA0151_features.csv (output of 02_feature_engineering.py).
Predicts sensors A, C, D jointly (one model, 3 outputs) since they're the same
intersection and correlated.

Key correctness points carried over from cleaning/feature-engineering:
- `segment_id` marks contiguous true-hourly runs. Sequence windows are only
  built from rows that share one segment_id, so a window never silently spans
  a data gap (the longest gap in this dataset is 35 days).
- Chronological split (train / val / test by date), not random.
- Scaler fit on train only.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
PROCESSED_DIR = Path("../dataset/processed")
SENSORS = ["GA0151_A", "GA0151_C", "GA0151_D"]
WINDOW = 48          # hours of history fed to the LSTM per prediction
VAL_FRAC = 0.15
TEST_FRAC = 0.15
BATCH_SIZE = 64
EPOCHS = 100
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ----------------------------------------------------------------------------
# 1. Load engineered features
# ----------------------------------------------------------------------------
df = pd.read_csv(PROCESSED_DIR / "GA0151_features.csv", parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

FEATURE_COLS = [c for c in df.columns if c not in ("datetime", "segment_id") and not c.endswith("_target")]
TARGET_COLS = [f"{s}_target" for s in SENSORS]

print(f"Loaded {len(df):,} rows, {len(FEATURE_COLS)} features, {df['segment_id'].nunique()} segments")

# ----------------------------------------------------------------------------
# 2. Chronological split by TIME, never randomly shuffled
# ----------------------------------------------------------------------------
n = len(df)
train_end = int(n * (1 - VAL_FRAC - TEST_FRAC))
val_end = int(n * (1 - TEST_FRAC))

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

print(f"Train: {train_df['datetime'].min()} -> {train_df['datetime'].max()}  ({len(train_df)} rows)")
print(f"Val:   {val_df['datetime'].min()} -> {val_df['datetime'].max()}  ({len(val_df)} rows)")
print(f"Test:  {test_df['datetime'].min()} -> {test_df['datetime'].max()}  ({len(test_df)} rows)")

# ----------------------------------------------------------------------------
# 3. Scale -- fit ONLY on train
# ----------------------------------------------------------------------------
feature_scaler = StandardScaler().fit(train_df[FEATURE_COLS])
target_scaler = StandardScaler().fit(train_df[TARGET_COLS])

for split_df in (train_df, val_df, test_df):
    split_df[FEATURE_COLS] = feature_scaler.transform(split_df[FEATURE_COLS])
    split_df[TARGET_COLS] = target_scaler.transform(split_df[TARGET_COLS])

# ----------------------------------------------------------------------------
# 4. Windowing -- segment-aware. A window of WINDOW rows is only kept if every
#    row in it (and its target row) shares the same segment_id, i.e. is truly
#    contiguous hourly data with no gap stitched in.
# ----------------------------------------------------------------------------
def make_sequences(split_df, window):
    X_seq, y_seq = [], []
    feats = split_df[FEATURE_COLS].to_numpy()
    targets = split_df[TARGET_COLS].to_numpy()
    seg_ids = split_df["segment_id"].to_numpy()

    for i in range(window, len(split_df)):
        window_segs = seg_ids[i - window:i + 1]  # includes the target row
        if len(set(window_segs)) != 1:
            continue  # window would cross a gap -- skip it
        X_seq.append(feats[i - window:i])
        y_seq.append(targets[i])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = make_sequences(train_df, WINDOW)
X_val_seq, y_val_seq = make_sequences(val_df, WINDOW)
X_test_seq, y_test_seq = make_sequences(test_df, WINDOW)

print(f"\nSequence shapes -> train: {X_train_seq.shape}, val: {X_val_seq.shape}, test: {X_test_seq.shape}")

n_features = X_train_seq.shape[2]
n_outputs = len(TARGET_COLS)

# ----------------------------------------------------------------------------
# 5. Model -- 2 stacked LSTM layers, ordinary Dropout (not recurrent_dropout,
#    which disables cuDNN), LayerNormalization instead of BatchNormalization
#    (BatchNorm inside a recurrent stack is a known failure mode).
# ----------------------------------------------------------------------------
def build_model(window, n_features, n_outputs, sensor_names):
    inputs = layers.Input(shape=(window, n_features))

    # Shared trunk -- unchanged capacity, still learns the common temporal
    # structure across all 3 sensors.
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # NEW: per-sensor output heads instead of one shared Dense(16) bottleneck.
    # Each head is small (8 units) so this adds only ~250 extra params per
    # sensor -- a targeted capacity increase, not a general one, so it
    # shouldn't meaningfully raise overfitting risk versus the old model.
    head_outputs = []
    for name in sensor_names:
        h = layers.Dense(8, activation="relu", name=f"{name}_head")(x)
        out = layers.Dense(1, activation="linear", name=f"{name}_out")(h)
        head_outputs.append(out)

    outputs = layers.Concatenate()(head_outputs)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model

model = build_model(WINDOW, n_features, n_outputs, SENSORS)
model.summary()

# ----------------------------------------------------------------------------
# 6. Train with early stopping on val loss
# ----------------------------------------------------------------------------
early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr],
    verbose=2,
)

# ----------------------------------------------------------------------------
# 7. Evaluate on test set, in ORIGINAL flow units
# ----------------------------------------------------------------------------
y_pred_scaled = model.predict(X_test_seq)
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_true = target_scaler.inverse_transform(y_test_seq)

print()
for i, s in enumerate(SENSORS):
    mae = np.mean(np.abs(y_true[:, i] - y_pred[:, i]))
    rmse = np.sqrt(np.mean((y_true[:, i] - y_pred[:, i]) ** 2))
    r2 = r2_score(y_true[:, i], y_pred[:, i])
    print(f"{s} -> Test MAE: {mae:.2f} vehicles/hr | Test RMSE: {rmse:.2f} vehicles/hr | Test R2: {r2:.4f}")

# ----------------------------------------------------------------------------
# 8. Naive persistence baseline -- ALWAYS compare against this before trusting
#    the LSTM's numbers. Given lag-1 autocorrelation ~0.84-0.89 from the EDA,
#    "predict same as last known hour" can be a tough baseline to beat at
#    HORIZON=1.
# ----------------------------------------------------------------------------
# raw (unscaled) sensor values, aligned to the same rows used in the test windows
test_raw = pd.read_csv(PROCESSED_DIR / "GA0151_features.csv", parse_dates=["datetime"])
test_raw = test_raw.sort_values("datetime").reset_index(drop=True).iloc[val_end:].reset_index(drop=True)

print("\nNaive persistence baseline (predict = last known raw value):")
seg_ids = test_raw["segment_id"].to_numpy()
for i, s in enumerate(SENSORS):
    raw_vals = test_raw[s].to_numpy()
    targets = test_raw[f"{s}_target"].to_numpy()
    preds, actuals = [], []
    for idx in range(WINDOW, len(test_raw)):
        if seg_ids[idx] != seg_ids[idx - 1]:
            continue
        preds.append(raw_vals[idx - 1])
        actuals.append(targets[idx])
    preds, actuals = np.array(preds), np.array(actuals)
    mae = np.mean(np.abs(actuals - preds))
    rmse = np.sqrt(np.mean((actuals - preds) ** 2))
    print(f"{s} -> Baseline MAE: {mae:.2f} | Baseline RMSE: {rmse:.2f}")

# ----------------------------------------------------------------------------
# Next steps if the LSTM doesn't clearly beat the baseline above:
# - Check train_loss vs val_loss from the fit() log:
#     close + both mediocre -> underfitting (more capacity / longer WINDOW)
#     train << val, gap widening -> overfitting (cut hidden units first)
# - Try a longer HORIZON (e.g. 24h) in 02_feature_engineering.py -- 1-hour-
#   ahead forecasts are inherently close to persistence given the sensor
#   autocorrelation seen in the EDA.
# ----------------------------------------------------------------------------
