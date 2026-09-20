"""
Step 3: Multi-output LSTM for GD0501 intersection traffic-flow forecasting.

Reads dataset/processed/GA0151_features.csv (output of 02_feature_engineering.py).
Predicts sensors GD0501_B, GD0501_C, GD0501_D jointly (one model, 3 outputs)
since they're the same intersection and correlated.

Key correctness points carried over from cleaning/feature-engineering:
- `segment_id` marks contiguous true-hourly runs. Sequence windows are only
  built from rows that share one segment_id, so a window never silently spans
  a data gap (the longest gap in this dataset is 35 days).
- Chronological split (train / val / test by date), not random.
- Scaler fit on train only.

FIXED in this version (the reason the baseline looked "too low"):
- OFF-BY-ONE in window construction. In 02_feature_engineering.py, row i holds
  the flow at hour t_i and `<sensor>_target` on that row is the flow at
  t_i + HORIZON. The old make_sequences() fed rows [i-WINDOW, i) as input but
  used targets[i] as the label -- so the newest hour the model was allowed to
  see was t_{i-1} while the label was t_i + 1: a 2-HOUR-ahead forecast, not the
  1-hour horizon the config says. The persistence baseline (raw_vals[idx-1])
  had the same 2-hour gap, so it was being scored on a harder task than
  intended, which is why its R2 (0.47-0.59) sat far below what lag-1
  autocorrelation of 0.84-0.89 implies (R2 ~ 0.7-0.8).
  Now the window is rows [i-WINDOW+1, i] (INCLUDING row i, the latest known
  hour) and the label is targets[i] -> a true HORIZON-hour-ahead forecast.
- The baseline is now evaluated on EXACTLY the same test windows as the LSTM
  (previously it used a looser filter and a different sample set).
- Added a seasonal-naive baseline (same hour yesterday). Traffic has a strong
  daily cycle, and persistence ignores it while the LSTM gets hour/day/month
  features, so persistence alone is a weak yardstick.
- Added sanity checks that fail loudly if targets, windows or the scaler
  inversion are misaligned.
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
SENSORS = ["GD0501_B", "GD0501_C", "GD0501_D"]
HORIZON = 1          # hours ahead. MUST equal HORIZON in 02_feature_engineering.py
WINDOW = 48          # hours of history fed to the LSTM per prediction
SEASONAL_LAG = 24    # "same hour yesterday" baseline
VAL_FRAC = 0.15
TEST_FRAC = 0.15
BATCH_SIZE = 64
EPOCHS = 100
SEED = 42

assert HORIZON < SEASONAL_LAG, "seasonal baseline needs HORIZON < SEASONAL_LAG"
assert SEASONAL_LAG - HORIZON <= WINDOW - 1, "WINDOW too short for the seasonal baseline"

tf.keras.utils.set_random_seed(SEED)   # seeds python, numpy and tensorflow

# ----------------------------------------------------------------------------
# 1. Load engineered features
# ----------------------------------------------------------------------------
df = pd.read_csv(PROCESSED_DIR / "GD0501_features.csv", parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

FEATURE_COLS = [c for c in df.columns if c not in ("datetime", "segment_id") and not c.endswith("_target")]
TARGET_COLS = [f"{s}_target" for s in SENSORS]

print(f"Loaded {len(df):,} rows, {len(FEATURE_COLS)} features, {df['segment_id'].nunique()} segments")


def check_target_alignment(frame, horizon):
    """Verify 02's `<sensor>_target` really is the flow `horizon` hours later.

    For rows i and i+horizon that are in the same segment and exactly
    `horizon` hours apart, target[i] must equal sensor[i+horizon].
    """
    dt = frame["datetime"].to_numpy()
    seg = frame["segment_id"].to_numpy()
    ok = np.where(
        (seg[horizon:] == seg[:-horizon])
        & ((dt[horizon:] - dt[:-horizon]) == np.timedelta64(horizon, "h"))
    )[0]
    if len(ok) == 0:
        raise ValueError("No row pairs available to verify target alignment")
    for s in SENSORS:
        tgt_now = frame[f"{s}_target"].to_numpy()[ok]
        val_later = frame[s].to_numpy()[ok + horizon]
        if not np.allclose(tgt_now, val_later):
            raise ValueError(
                f"{s}_target is not the flow {horizon}h ahead -- is HORIZON here "
                f"the same as in 02_feature_engineering.py?"
            )
    print(f"Target alignment OK: <sensor>_target == flow {horizon}h ahead "
          f"({len(ok):,} row pairs checked)")


check_target_alignment(df, HORIZON)

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

# Keep an UNSCALED copy of the test split (log1p space, as written by step 02).
# Baselines and the ground truth are taken from this, so they never depend on
# the scaler.
test_raw = test_df.reset_index(drop=True)

# ----------------------------------------------------------------------------
# 3. Scale -- fit ONLY on train
# ----------------------------------------------------------------------------
feature_scaler = StandardScaler().fit(train_df[FEATURE_COLS])
target_scaler = StandardScaler().fit(train_df[TARGET_COLS])

for split_df in (train_df, val_df, test_df):
    split_df[FEATURE_COLS] = feature_scaler.transform(split_df[FEATURE_COLS])
    split_df[TARGET_COLS] = target_scaler.transform(split_df[TARGET_COLS])

# ----------------------------------------------------------------------------
# 4. Windowing -- segment-aware and correctly aligned.
#
#    A sample ends at row e:
#        input  = feature rows [e-WINDOW+1 ... e]   (INCLUDES row e, the latest
#                                                    hour we know about)
#        label  = targets[e]                        (flow at t_e + HORIZON)
#
#    The window is kept only if all its rows share one segment_id AND span
#    exactly WINDOW-1 hours, i.e. it is truly contiguous hourly data. The
#    label row is guaranteed to be in the same segment: step 02 dropped every
#    row whose target crossed a segment boundary.
# ----------------------------------------------------------------------------
def make_sequences(split_df, window):
    feats = split_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    targets = split_df[TARGET_COLS].to_numpy(dtype=np.float32)
    seg_ids = split_df["segment_id"].to_numpy()
    times = split_df["datetime"].to_numpy()

    end_idx = np.arange(window - 1, len(split_df))
    start_idx = end_idx - (window - 1)

    same_segment = seg_ids[start_idx] == seg_ids[end_idx]
    contiguous = (times[end_idx] - times[start_idx]) == np.timedelta64(window - 1, "h")
    end_idx = end_idx[same_segment & contiguous]

    if len(end_idx) == 0:
        raise ValueError("No valid windows in this split -- reduce WINDOW or change the split")

    X = np.stack([feats[e - window + 1:e + 1] for e in end_idx])
    y = targets[end_idx]
    return X, y, end_idx


X_train_seq, y_train_seq, _ = make_sequences(train_df, WINDOW)
X_val_seq, y_val_seq, _ = make_sequences(val_df, WINDOW)
X_test_seq, y_test_seq, test_end = make_sequences(test_df, WINDOW)

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

    # Shared trunk -- learns the common temporal structure across all 3 sensors.
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Per-sensor output heads (small: 8 units each).
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
# 7. Evaluate everything on the SAME test windows, in real vehicles/hour
# ----------------------------------------------------------------------------
def evaluate(label, y_true, y_pred):
    print(f"\n{label}")
    out = {}
    for i, s in enumerate(SENSORS):
        err = y_true[:, i] - y_pred[:, i]
        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err ** 2))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        print(f"{s} -> MAE: {mae:.2f} vehicles/hr | RMSE: {rmse:.2f} vehicles/hr | R2: {r2:.4f}")
        out[s] = {"mae": mae, "rmse": rmse, "r2": r2}
    return out


# Ground truth straight from the unscaled log1p targets (independent of the
# scaler), inverted back to vehicles/hour.
y_true = np.expm1(test_raw[TARGET_COLS].to_numpy()[test_end])

# LSTM: scaled -> log1p space -> vehicles/hour (clipped: flow can't be negative)
y_pred_scaled = model.predict(X_test_seq, verbose=0)
y_pred_log = target_scaler.inverse_transform(y_pred_scaled)
y_pred = np.clip(np.expm1(y_pred_log), 0, None)

# Sanity check: labels that went through the scaler round-trip must match the
# raw ground truth, otherwise windows and targets are misaligned.
y_true_from_scaler = np.expm1(target_scaler.inverse_transform(y_test_seq))
assert np.allclose(y_true, y_true_from_scaler, rtol=1e-3, atol=1e-2), \
    "Test labels do not match raw targets -- window/target misalignment"

# Baselines, built on the same rows (test_end) the LSTM is scored on.
#   persistence: the latest KNOWN hour (row e, same row the LSTM sees last)
#   seasonal:    flow at the target hour minus 24h = row e-(24-HORIZON),
#                which is always inside the window
raw_now = test_raw[SENSORS].to_numpy()                      # log1p space
persist_pred = np.expm1(raw_now[test_end])
seasonal_pred = np.expm1(raw_now[test_end - (SEASONAL_LAG - HORIZON)])

print(f"\nEvaluated on {len(test_end):,} test windows, horizon = {HORIZON}h")
lstm_res = evaluate("LSTM:", y_true, y_pred)
pers_res = evaluate("Naive persistence baseline (predict = latest known hour):", y_true, persist_pred)
seas_res = evaluate(f"Seasonal-naive baseline (same hour {SEASONAL_LAG}h earlier):", y_true, seasonal_pred)

print("\nLSTM improvement over the better of the two baselines (positive = LSTM better):")
for s in SENSORS:
    best_mae = min(pers_res[s]["mae"], seas_res[s]["mae"])
    best_rmse = min(pers_res[s]["rmse"], seas_res[s]["rmse"])
    print(f"{s} -> MAE {100 * (1 - lstm_res[s]['mae'] / best_mae):+.1f}% | "
          f"RMSE {100 * (1 - lstm_res[s]['rmse'] / best_rmse):+.1f}%")

# ----------------------------------------------------------------------------
# Next steps if the LSTM doesn't clearly beat the baselines above:
# - Check train_loss vs val_loss from the fit() log:
#     close + both mediocre -> underfitting (more capacity / longer WINDOW)
#     train << val, gap widening -> overfitting (cut hidden units first)
# - Val and test are later, contiguous slices of the data, so they can cover
#   different seasons than train. If val_loss stays well above train_loss, look
#   at which months each split contains before changing the architecture.
# ----------------------------------------------------------------------------
