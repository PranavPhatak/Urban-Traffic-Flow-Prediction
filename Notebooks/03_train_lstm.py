"""
Step 3: Multi-output LSTM for GD0501 intersection traffic-flow forecasting.

Reads dataset/processed/GD0501_features.csv (output of 02_feature_engineering.py).
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
- Stronger baselines (profile-adjusted persistence, Ridge on the same windows),
  all fit on train only. They sit AFTER training/evaluation of the LSTM and do
  not touch the model, data, seed or training loop, so LSTM results are
  unchanged by them.
- GD0501_D (the hardest sensor) gets dedicated single-sensor models trained
  AFTER the original 3-output model, and the final D prediction is whichever
  candidate has the best VALIDATION RMSE. The original 3-output model is
  trained exactly as before, and the B and C columns are never replaced, so
  their predictions/metrics are unchanged.
- The D candidates now include seed-averaged, flow-weighted and Huber-loss
  specialists (Huber limits the pull of unpredictable one-off surges/dropouts).
- A residual specialist for D that forecasts the CHANGE from the latest hour, so
  an unusual level (e.g. a busy night) is carried forward instead of being
  pulled back to the usual value for that hour. Chosen only if it wins on
  validation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

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
RIDGE_ALPHAS = [1, 10, 100, 1000, 10000]   # picked on VAL, never on test
SPECIALIST_SENSORS = ["GD0501_D"]   # sensors that also get a dedicated model (B/C left untouched)
SPECIALIST_SEEDS = 3                  # networks averaged per weighted variant (more = steadier, slower)
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
train_raw = train_df.reset_index(drop=True)   # for the train-only baselines below
val_raw = val_df.reset_index(drop=True)

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


X_train_seq, y_train_seq, train_end_idx = make_sequences(train_df, WINDOW)
X_val_seq, y_val_seq, val_end_idx = make_sequences(val_df, WINDOW)
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
        wape = 100 * np.sum(np.abs(err)) / np.sum(y_true[:, i])   # scale-free error %
        print(f"{s} -> MAE: {mae:.2f} vehicles/hr | RMSE: {rmse:.2f} vehicles/hr | "
              f"R2: {r2:.4f} | WAPE: {wape:.1f}%")
        out[s] = {"mae": mae, "rmse": rmse, "r2": r2, "wape": wape}
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

# ---------------------------------------------------------------------------
# 7a. Dedicated models for the weakest sensor(s).
#
# Why separate models instead of changing the shared one: any change to the
# 3-output network (size, loss, weights) changes the random init/gradients for
# B and C too. Extra models trained AFTER it leave B and C bit-for-bit the same,
# and the original model's D output stays in the running as a candidate.
#
# Candidates, all judged on VALIDATION real-unit RMSE (test is printed for
# information only and never used to choose):
#   multi-output   : D output of the original 3-output model
#   specialist     : 1 single-output model (same trunk, dedicated head), plain MSE
#   weighted       : mean of SPECIALIST_SEEDS specialists trained with a
#       flow-weighted loss. Training is in log1p space, where a 2-vs-5 vehicle
#       error at 3am costs as much as 200-vs-500 at rush hour, yet MAE/RMSE are
#       in vehicles/hr. Since d(vehicles) ~ (v+1) * d(log1p v), weighting by
#       (v+1) pulls the objective toward the metric we report.
#   weighted-huber : same, with a HUBER loss so a few unpredictable targets
#       (one-off surges, sensor dropouts) cannot dominate the gradient.
#   residual-huber : NEW. Same loss, but the network predicts the CHANGE from
#       the latest observed hour instead of the absolute level:
#           log1p(flow at t+H) = log1p(flow at t) + network output
#       Why: D's worst test windows are a night-time surge (31 May - 2 Jun,
#       63-173 veh/hr at 00:00-03:00). The direct models answered ~6 veh/hr
#       even when the hour they had just seen was 143, because "night = quiet"
#       is baked into their weights and a busy night is out of distribution for
#       them. With the residual form the forecast starts from what was just
#       observed and the network only has to learn the typical change, which is
#       small at night, so an out-of-distribution level is carried through
#       (persistence-like) instead of being pulled back to the usual night value.
#   blend-huber    : mean of weighted-huber and residual-huber
#   ensemble       : mean of the five models above
#
# Seed-averaging cuts the run-to-run noise of a single network. All averaging
# is done in vehicles/hour (not log space), which avoids the downward bias of
# averaging log-predictions before expm1.
# ---------------------------------------------------------------------------
def scaled_to_vehicles(y_scaled, j):
    """One target column: scaled -> log1p space -> vehicles/hour (>= 0)."""
    return np.clip(np.expm1(y_scaled * target_scaler.scale_[j] + target_scaler.mean_[j]), 0, None)


# Anchors for the residual model, taken from the UNSCALED frames on exactly the
# rows each window ends at: now_log = log1p flow of the newest hour in the
# window, tgt_log = log1p flow HORIZON hours later.
fi = [FEATURE_COLS.index(s) for s in SENSORS]
now_log = {
    "train": train_raw[SENSORS].to_numpy()[train_end_idx],
    "val": val_raw[SENSORS].to_numpy()[val_end_idx],
    "test": test_raw[SENSORS].to_numpy()[test_end],
}
tgt_log = {
    "train": train_raw[TARGET_COLS].to_numpy()[train_end_idx],
    "val": val_raw[TARGET_COLS].to_numpy()[val_end_idx],
    "test": test_raw[TARGET_COLS].to_numpy()[test_end],
}
# Sanity check: the anchor must equal the newest hour the network is given.
_newest_input = X_test_seq[:, -1, fi] * feature_scaler.scale_[fi] + feature_scaler.mean_[fi]
assert np.allclose(_newest_input, now_log["test"], atol=1e-3), \
    "Residual anchor does not match the newest input hour -- window misalignment"
_delta_train = tgt_log["train"] - now_log["train"]
res_mean, res_std = _delta_train.mean(axis=0), _delta_train.std(axis=0)   # train stats only


def to_vehicles(out, j, split, residual):
    """Network output -> vehicles/hour for a split ('train'/'val'/'test')."""
    if residual:
        log_pred = now_log[split][:, j] + out * res_std[j] + res_mean[j]
        return np.clip(np.expm1(log_pred), 0, None)
    return scaled_to_vehicles(out, j)


def build_single_model(window, n_features, name, huber):
    inputs = layers.Input(shape=(window, n_features))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation="relu", name=f"{name}_head")(x)
    out = layers.Dense(1, activation="linear", name=f"{name}_out")(x)
    m = models.Model(inputs, out, name=f"{name}_specialist")
    loss = tf.keras.losses.Huber(delta=1.0) if huber else "mse"   # delta in standardised units
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=loss)
    return m


def fit_specialist(j, sensor, weighted, huber, residual, seed):
    tf.keras.utils.set_random_seed(seed)
    m = build_single_model(WINDOW, n_features, sensor, huber)
    if residual:
        ytr = ((tgt_log["train"][:, j] - now_log["train"][:, j] - res_mean[j]) / res_std[j])[:, None]
        yva = ((tgt_log["val"][:, j] - now_log["val"][:, j] - res_mean[j]) / res_std[j])[:, None]
    else:
        ytr, yva = y_train_seq[:, [j]], y_val_seq[:, [j]]
    if weighted:
        w_tr = np.expm1(target_scaler.inverse_transform(y_train_seq))[:, j] + 1.0
        w_va = np.expm1(target_scaler.inverse_transform(y_val_seq))[:, j] + 1.0
        norm = w_tr.mean()                      # train mean for both, so scales match
        val_data = (X_val_seq, yva, w_va / norm)
        fit_kwargs = {"sample_weight": w_tr / norm}
    else:
        val_data = (X_val_seq, yva)
        fit_kwargs = {}
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]
    h = m.fit(X_train_seq, ytr, validation_data=val_data, epochs=EPOCHS,
              batch_size=BATCH_SIZE, callbacks=cbs, verbose=0, **fit_kwargs)
    tag = ("weighted " if weighted else "") + ("huber " if huber else "") + ("residual " if residual else "")
    print(f"  trained {tag}specialist for {sensor} (seed {seed}): "
          f"{len(h.history['loss'])} epochs, best val_loss {min(h.history['val_loss']):.4f}")
    return m


# name, flow-weighted?, huber?, residual?, seed offset, number of seeds averaged
# (seed offsets 1, 2 and 12 are the same seeds as the previous version of this script)
SPECIALIST_VARIANTS = [
    ("specialist",     False, False, False, 1,  1),
    ("weighted",       True,  False, False, 2,  SPECIALIST_SEEDS),
    ("weighted-huber", True,  True,  False, 12, SPECIALIST_SEEDS),
    ("residual-huber", True,  True,  True,  22, SPECIALIST_SEEDS),
]

y_val_true = np.expm1(target_scaler.inverse_transform(y_val_seq))
val_pred_scaled = model.predict(X_val_seq, verbose=0)
y_pred_final = y_pred.copy()          # B/C columns are never modified

for sensor in SPECIALIST_SENSORS:
    j = SENSORS.index(sensor)
    print(f"\nDedicated models for {sensor}:")
    veh_val = {"multi-output": scaled_to_vehicles(val_pred_scaled[:, j], j)}
    veh_test = {"multi-output": scaled_to_vehicles(y_pred_scaled[:, j], j)}
    for name, weighted, huber, residual, offset, n_seeds in SPECIALIST_VARIANTS:
        pv, pt = [], []
        for k in range(n_seeds):
            m = fit_specialist(j, sensor, weighted, huber, residual, SEED + offset + k)
            pv.append(to_vehicles(m.predict(X_val_seq, verbose=0)[:, 0], j, "val", residual))
            pt.append(to_vehicles(m.predict(X_test_seq, verbose=0)[:, 0], j, "test", residual))
            del m
        veh_val[name] = np.mean(pv, axis=0)
        veh_test[name] = np.mean(pt, axis=0)
    members = list(veh_val)
    veh_val["blend-huber"] = np.mean([veh_val["weighted-huber"], veh_val["residual-huber"]], axis=0)
    veh_test["blend-huber"] = np.mean([veh_test["weighted-huber"], veh_test["residual-huber"]], axis=0)
    veh_val["ensemble"] = np.mean([veh_val[k] for k in members], axis=0)
    veh_test["ensemble"] = np.mean([veh_test[k] for k in members], axis=0)

    print(f"\n{sensor} candidates (selection uses VAL RMSE only):")
    print(f"{'candidate':16s} {'val RMSE':>9s} {'val MAE':>8s} | {'test RMSE':>9s} {'test MAE':>8s} {'test R2':>8s}")
    val_rmse = {}
    for k in veh_val:
        ev = y_val_true[:, j] - veh_val[k]
        et = y_true[:, j] - veh_test[k]
        val_rmse[k] = np.sqrt(np.mean(ev ** 2))
        print(f"{k:16s} {val_rmse[k]:9.2f} {np.mean(np.abs(ev)):8.2f} | "
              f"{np.sqrt(np.mean(et ** 2)):9.2f} {np.mean(np.abs(et)):8.2f} "
              f"{r2_score(y_true[:, j], veh_test[k]):8.4f}")
    best = min(val_rmse, key=val_rmse.get)
    print(f"-> selected for {sensor}: {best}")
    y_pred_final[:, j] = veh_test[best]

# ---------------------------------------------------------------------------
# Baselines. Every one is scored on the same test windows (test_end) as the
# LSTM, and everything learned (profile, ridge weights, ridge alpha) uses TRAIN
# (and VAL for alpha) only -- nothing is tuned on test.
# ---------------------------------------------------------------------------
raw_now = test_raw[SENSORS].to_numpy()                      # log1p space

# (1) persistence: the latest KNOWN hour (row e, the same row the LSTM sees last)
persist_pred = np.expm1(raw_now[test_end])

# (2) seasonal naive: flow at the target hour minus 24h = row e-(24-HORIZON),
#     which is always inside the window
seasonal_pred = np.expm1(raw_now[test_end - (SEASONAL_LAG - HORIZON)])


# (3) profile-adjusted persistence: last known value + the TYPICAL hour-to-hour
#     change for that hour of the week (learned from train). Plain persistence
#     is blind to the morning/evening ramps; this keeps its "start from what we
#     just saw" strength and adds the expected ramp. Done in log1p space.
def hour_of_week(frame):
    dt = frame["datetime"]
    return (dt.dt.dayofweek * 24 + dt.dt.hour).to_numpy()


profile = train_raw.groupby(hour_of_week(train_raw))[SENSORS].median().reindex(range(168))
assert not profile.isna().any().any(), "train split is missing some hour-of-week bins"
profile = profile.to_numpy()                                # (168, n_sensors)

how_now = hour_of_week(test_raw)[test_end]
how_target = (how_now + HORIZON) % 168
profile_pred = np.clip(
    np.expm1(raw_now[test_end] + profile[how_target] - profile[how_now]), 0, None
)

# (4) Ridge (linear) on the exact same flattened windows the LSTM gets. This is
#     the honest "is the recurrent network actually needed?" check. alpha is
#     chosen on validation.
Xtr = X_train_seq.reshape(len(X_train_seq), -1)
Xva = X_val_seq.reshape(len(X_val_seq), -1)
Xte = X_test_seq.reshape(len(X_test_seq), -1)
best_val, best_alpha, ridge = None, None, None
for alpha in RIDGE_ALPHAS:
    candidate = Ridge(alpha=alpha).fit(Xtr, y_train_seq)
    val_mae = np.mean(np.abs(candidate.predict(Xva) - y_val_seq))
    if best_val is None or val_mae < best_val:
        best_val, best_alpha, ridge = val_mae, alpha, candidate
print(f"\nRidge baseline: alpha={best_alpha} (chosen on validation)")
ridge_pred = np.clip(
    np.expm1(target_scaler.inverse_transform(ridge.predict(Xte))), 0, None
)

print(f"\nEvaluated on {len(test_end):,} test windows, horizon = {HORIZON}h")
lstm_orig_res = evaluate("LSTM (original 3-output model, unchanged):", y_true, y_pred)
lstm_res = evaluate("LSTM FINAL (specialist sensors use the validation-selected model):", y_true, y_pred_final)
baseline_res = {
    "persistence": evaluate("Naive persistence baseline (predict = latest known hour):", y_true, persist_pred),
    "seasonal-24h": evaluate(f"Seasonal-naive baseline (same hour {SEASONAL_LAG}h earlier):", y_true, seasonal_pred),
    "profile-persistence": evaluate("Profile-adjusted persistence (latest hour + typical hourly change):", y_true, profile_pred),
    "ridge": evaluate("Ridge regression on the same windows (linear, no recurrence):", y_true, ridge_pred),
}

print("\nLSTM vs the STRONGEST baseline per sensor (positive = LSTM better):")
for s in SENSORS:
    best_mae_name = min(baseline_res, key=lambda k: baseline_res[k][s]["mae"])
    best_rmse_name = min(baseline_res, key=lambda k: baseline_res[k][s]["rmse"])
    best_mae = baseline_res[best_mae_name][s]["mae"]
    best_rmse = baseline_res[best_rmse_name][s]["rmse"]
    print(f"{s} -> MAE {100 * (1 - lstm_res[s]['mae'] / best_mae):+.1f}% (vs {best_mae_name}) | "
          f"RMSE {100 * (1 - lstm_res[s]['rmse'] / best_rmse):+.1f}% (vs {best_rmse_name})")

# ---------------------------------------------------------------------------
# Error diagnostics -- WHY a sensor scores lower. R2 is relative to each
# sensor's own variance, so it is not comparable across sensors; RMSE/std is.
# ---------------------------------------------------------------------------
print("\nError diagnostics (final LSTM, test set):")
for i, s in enumerate(SENSORS):
    err2 = (y_true[:, i] - y_pred_final[:, i]) ** 2
    k = max(1, int(0.01 * len(err2)))
    top_share = np.sort(err2)[-k:].sum() / err2.sum()
    print(f"{s}: mean {y_true[:, i].mean():.1f}, std {y_true[:, i].std():.1f} | "
          f"RMSE/std {np.sqrt(err2.mean()) / y_true[:, i].std():.3f} | "
          f"worst 1% of windows = {100 * top_share:.1f}% of squared error | "
          f"zero-flow targets: {(y_true[:, i] < 0.5).sum()}")

test_times = test_raw["datetime"].to_numpy()[test_end]
for sensor in SPECIALIST_SENSORS:
    j = SENSORS.index(sensor)
    abs_err = np.abs(y_true[:, j] - y_pred_final[:, j])
    worst = np.argsort(-abs_err)[:8]
    share = 100 * np.sum(abs_err[worst] ** 2) / np.sum(abs_err ** 2)
    print(f"\nWorst 8 test windows for {sensor} = {share:.1f}% of its total squared error")
    print("(target hour | actual | LSTM | persistence | actual flow of every sensor at that hour)")
    print("If B and C spiked too it was real traffic; if only this sensor did, suspect the sensor.")
    for w in worst:
        t = pd.Timestamp(test_times[w]) + pd.Timedelta(hours=HORIZON)
        others = "  ".join(f"{s.split('_')[-1]}={y_true[w, i]:.0f}" for i, s in enumerate(SENSORS))
        print(f"  {t} | {y_true[w, j]:7.1f} | {y_pred_final[w, j]:7.1f} | {persist_pred[w, j]:7.1f} | {others}")

# ----------------------------------------------------------------------------
# Next steps if the LSTM doesn't clearly beat the baselines above:
# - Check train_loss vs val_loss from the fit() log:
#     close + both mediocre -> underfitting (more capacity / longer WINDOW)
#     train << val, gap widening -> overfitting (cut hidden units first)
# - Val and test are later, contiguous slices of the data, so they can cover
#   different seasons than train. If val_loss stays well above train_loss, look
#   at which months each split contains before changing the architecture.
# ----------------------------------------------------------------------------
