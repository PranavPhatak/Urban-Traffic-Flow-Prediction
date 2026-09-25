from pathlib import Path
import json
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

PROCESSED_DIR = Path("../dataset/processed")
MODELS_DIR = Path("../models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SENSORS = ["GD0501_B", "GD0501_C", "GD0501_D"]
HORIZON = 1          
WINDOW = 48         
SEASONAL_LAG = 24    
RIDGE_ALPHAS = [1, 10, 100, 1000, 10000]   
SPECIALIST_SENSORS = ["GD0501_D"]   
SPECIALIST_SEEDS = 3                  
VAL_FRAC = 0.15
TEST_FRAC = 0.15
BATCH_SIZE = 64
EPOCHS = 100
SEED = 42

assert HORIZON < SEASONAL_LAG, "seasonal baseline needs HORIZON < SEASONAL_LAG"
assert SEASONAL_LAG - HORIZON <= WINDOW - 1, "WINDOW too short for the seasonal baseline"

tf.keras.utils.set_random_seed(SEED)   

df = pd.read_csv(PROCESSED_DIR / "GD0501_features.csv", parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

FEATURE_COLS = [c for c in df.columns if c not in ("datetime", "segment_id") and not c.endswith("_target")]
TARGET_COLS = [f"{s}_target" for s in SENSORS]

print(f"Loaded {len(df):,} rows, {len(FEATURE_COLS)} features, {df['segment_id'].nunique()} segments")


def check_target_alignment(frame, horizon):
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

n = len(df)
train_end = int(n * (1 - VAL_FRAC - TEST_FRAC))
val_end = int(n * (1 - TEST_FRAC))

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

print(f"Train: {train_df['datetime'].min()} -> {train_df['datetime'].max()}  ({len(train_df)} rows)")
print(f"Val:   {val_df['datetime'].min()} -> {val_df['datetime'].max()}  ({len(val_df)} rows)")
print(f"Test:  {test_df['datetime'].min()} -> {test_df['datetime'].max()}  ({len(test_df)} rows)")

test_raw = test_df.reset_index(drop=True)
train_raw = train_df.reset_index(drop=True)   
val_raw = val_df.reset_index(drop=True)

feature_scaler = StandardScaler().fit(train_df[FEATURE_COLS])
target_scaler = StandardScaler().fit(train_df[TARGET_COLS])

for split_df in (train_df, val_df, test_df):
    split_df[FEATURE_COLS] = feature_scaler.transform(split_df[FEATURE_COLS])
    split_df[TARGET_COLS] = target_scaler.transform(split_df[TARGET_COLS])

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

def build_model(window, n_features, n_outputs, sensor_names):
    inputs = layers.Input(shape=(window, n_features))

    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)

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

SHARED_MODEL_PATH = MODELS_DIR / "shared_multioutput_lstm.keras"
model.save(SHARED_MODEL_PATH)
joblib.dump(feature_scaler, MODELS_DIR / "feature_scaler.pkl")
joblib.dump(target_scaler, MODELS_DIR / "target_scaler.pkl")
print(f"\nSaved shared multi-output model (best val_loss checkpoint) -> {SHARED_MODEL_PATH}")
print(f"Saved feature/target scalers -> {MODELS_DIR}")

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

y_true = np.expm1(test_raw[TARGET_COLS].to_numpy()[test_end])

y_pred_scaled = model.predict(X_test_seq, verbose=0)
y_pred_log = target_scaler.inverse_transform(y_pred_scaled)
y_pred = np.clip(np.expm1(y_pred_log), 0, None)

y_true_from_scaler = np.expm1(target_scaler.inverse_transform(y_test_seq))
assert np.allclose(y_true, y_true_from_scaler, rtol=1e-3, atol=1e-2), \
    "Test labels do not match raw targets -- window/target misalignment"

def scaled_to_vehicles(y_scaled, j):
    
    return np.clip(np.expm1(y_scaled * target_scaler.scale_[j] + target_scaler.mean_[j]), 0, None)

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

_newest_input = X_test_seq[:, -1, fi] * feature_scaler.scale_[fi] + feature_scaler.mean_[fi]
assert np.allclose(_newest_input, now_log["test"], atol=1e-3), \
    "Residual anchor does not match the newest input hour -- window misalignment"
_delta_train = tgt_log["train"] - now_log["train"]
res_mean, res_std = _delta_train.mean(axis=0), _delta_train.std(axis=0)   


def to_vehicles(out, j, split, residual):
    
    if residual:
        log_pred = now_log[split][:, j] + out * res_std[j] + res_mean[j]
        return np.clip(np.expm1(log_pred), 0, None)
    return scaled_to_vehicles(out, j)

def window_series(raw_series, end_idx, window):
    vals = raw_series.astype(np.float32)
    return np.stack([vals[e - window + 1:e + 1] for e in end_idx])[..., None]


def neighbor_divergence_window(sensor, split_raw, end_idx):
    others = [s for s in SENSORS if s != sensor]
    divergence = split_raw[others].std(axis=1).to_numpy()   
    return window_series(divergence, end_idx, WINDOW)


divergence_window = {}
div_stats = {}   
for sensor in SPECIALIST_SENSORS:
    divergence_window[sensor] = {
        "train": neighbor_divergence_window(sensor, train_raw, train_end_idx),
        "val": neighbor_divergence_window(sensor, val_raw, val_end_idx),
        "test": neighbor_divergence_window(sensor, test_raw, test_end),
    }
    
    div_mean = divergence_window[sensor]["train"].mean()
    div_std = divergence_window[sensor]["train"].std() + 1e-8
    div_stats[sensor] = (float(div_mean), float(div_std))
    for split in ("train", "val", "test"):
        divergence_window[sensor][split] = (divergence_window[sensor][split] - div_mean) / div_std


def augmented_inputs(sensor, split, use_divergence):
    
    base = {"train": X_train_seq, "val": X_val_seq, "test": X_test_seq}[split]
    if not use_divergence:
        return base
    return np.concatenate([base, divergence_window[sensor][split]], axis=-1)


def build_single_model(window, n_features, name, huber, extra_features=0):
    inputs = layers.Input(shape=(window, n_features + extra_features))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation="relu", name=f"{name}_head")(x)
    out = layers.Dense(1, activation="linear", name=f"{name}_out")(x)
    m = models.Model(inputs, out, name=f"{name}_specialist")
    loss = tf.keras.losses.Huber(delta=1.0) if huber else "mse"   
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=loss)
    return m


def fit_specialist(j, sensor, weighted, huber, residual, seed, use_divergence=False):
    tf.keras.utils.set_random_seed(seed)
    m = build_single_model(WINDOW, n_features, sensor, huber,
                            extra_features=1 if use_divergence else 0)
    Xtr = augmented_inputs(sensor, "train", use_divergence)
    Xva = augmented_inputs(sensor, "val", use_divergence)
    if residual:
        ytr = ((tgt_log["train"][:, j] - now_log["train"][:, j] - res_mean[j]) / res_std[j])[:, None]
        yva = ((tgt_log["val"][:, j] - now_log["val"][:, j] - res_mean[j]) / res_std[j])[:, None]
    else:
        ytr, yva = y_train_seq[:, [j]], y_val_seq[:, [j]]
    if weighted:
        w_tr = np.expm1(target_scaler.inverse_transform(y_train_seq))[:, j] + 1.0
        w_va = np.expm1(target_scaler.inverse_transform(y_val_seq))[:, j] + 1.0
        norm = w_tr.mean()                      
        val_data = (Xva, yva, w_va / norm)
        fit_kwargs = {"sample_weight": w_tr / norm}
    else:
        val_data = (Xva, yva)
        fit_kwargs = {}
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]
    h = m.fit(Xtr, ytr, validation_data=val_data, epochs=EPOCHS,
              batch_size=BATCH_SIZE, callbacks=cbs, verbose=0, **fit_kwargs)
    tag = ("weighted " if weighted else "") + ("huber " if huber else "") + \
          ("residual " if residual else "") + ("div " if use_divergence else "")
    print(f"  trained {tag}specialist for {sensor} (seed {seed}): "
          f"{len(h.history['loss'])} epochs, best val_loss {min(h.history['val_loss']):.4f}")
    return m

SPECIALIST_VARIANTS = [
    ("specialist",         False, False, False, 1,  1,                False),
    ("weighted",           True,  False, False, 2,  SPECIALIST_SEEDS, False),
    ("weighted-huber",     True,  True,  False, 12, SPECIALIST_SEEDS, False),
    ("residual-huber",     True,  True,  True,  22, SPECIALIST_SEEDS, False),
    ("weighted-huber-div", True,  True,  False, 32, SPECIALIST_SEEDS, True),
    ("residual-huber-div", True,  True,  True,  42, SPECIALIST_SEEDS, True),
]

y_val_true = np.expm1(target_scaler.inverse_transform(y_val_seq))
val_pred_scaled = model.predict(X_val_seq, verbose=0)
y_pred_final = y_pred.copy()          
chosen_candidate = {s: "multi-output" for s in SENSORS}

trained_models = {}

COMPOSITE_MAP = {
    "blend-huber": ["weighted-huber", "residual-huber"],
    "blend-huber-div": ["weighted-huber-div", "residual-huber-div"],
}


def expand_candidate(name, ensemble_members):
    
    if name == "ensemble":
        subs = ensemble_members
    elif name in COMPOSITE_MAP:
        subs = COMPOSITE_MAP[name]
    else:
        return {name: 1.0}
    share = 1.0 / len(subs)
    result = {}
    for s in subs:
        for leaf, w in expand_candidate(s, ensemble_members).items():
            result[leaf] = result.get(leaf, 0.0) + w * share
    return result

def save_best_specialist(sensor, best_name, ensemble_members, val_rmse_map, veh_test_map, y_true_col, j):
    
    leaves = expand_candidate(best_name, ensemble_members)
    sensor_dir = MODELS_DIR / "specialists" / sensor
    sensor_dir.mkdir(parents=True, exist_ok=True)

    components, needs_residual, needs_divergence = [], False, False
    for leaf, weight in leaves.items():
        if leaf == "multi-output":
            components.append({"leaf": "multi-output", "weight": weight, "seed_model_paths": []})
            continue
        info = trained_models[sensor][leaf]
        needs_residual = needs_residual or info["residual"]
        needs_divergence = needs_divergence or info["use_divergence"]
        seed_paths = []
        for k, seed_model in enumerate(info["seed_models"]):
            path = sensor_dir / f"{leaf}__seed{k}.keras"
            seed_model.save(path)
            seed_paths.append(str(path.relative_to(MODELS_DIR)))
        components.append({
            "leaf": leaf, "weight": weight, "seed_model_paths": seed_paths,
            "residual": info["residual"], "use_divergence": info["use_divergence"],
        })

    err = y_true_col - veh_test_map[best_name]
    manifest = {
        "sensor": sensor,
        "selected_candidate": best_name,
        "val_rmse": float(val_rmse_map[best_name]),
        "test_rmse": float(np.sqrt(np.mean(err ** 2))),
        "test_mae": float(np.mean(np.abs(err))),
        "test_r2": float(r2_score(y_true_col, veh_test_map[best_name])),
        "components": components,
        "window": WINDOW,
        "horizon": HORIZON,
        "feature_cols": FEATURE_COLS,
        "residual_stats": {"mean": float(res_mean[j]), "std": float(res_std[j])} if needs_residual else None,
        "divergence_stats": (
            {"mean": div_stats[sensor][0], "std": div_stats[sensor][1]} if needs_divergence else None
        ),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(sensor_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved winning '{best_name}' model(s) for {sensor} -> {sensor_dir}")


for sensor in SPECIALIST_SENSORS:
    j = SENSORS.index(sensor)
    print(f"\nDedicated models for {sensor}:")
    veh_val = {"multi-output": scaled_to_vehicles(val_pred_scaled[:, j], j)}
    veh_test = {"multi-output": scaled_to_vehicles(y_pred_scaled[:, j], j)}
    trained_models[sensor] = {}
    for name, weighted, huber, residual, offset, n_seeds, use_divergence in SPECIALIST_VARIANTS:
        Xva_pred = augmented_inputs(sensor, "val", use_divergence)
        Xte_pred = augmented_inputs(sensor, "test", use_divergence)
        pv, pt, seed_models = [], [], []
        for k in range(n_seeds):
            m = fit_specialist(j, sensor, weighted, huber, residual, SEED + offset + k, use_divergence)
            pv.append(to_vehicles(m.predict(Xva_pred, verbose=0)[:, 0], j, "val", residual))
            pt.append(to_vehicles(m.predict(Xte_pred, verbose=0)[:, 0], j, "test", residual))
            seed_models.append(m)   # kept (not deleted) so the winning variant can be saved below
        veh_val[name] = np.mean(pv, axis=0)
        veh_test[name] = np.mean(pt, axis=0)
        trained_models[sensor][name] = {
            "seed_models": seed_models, "residual": residual, "use_divergence": use_divergence,
        }
    members = list(veh_val)
    veh_val["blend-huber"] = np.mean([veh_val["weighted-huber"], veh_val["residual-huber"]], axis=0)
    veh_test["blend-huber"] = np.mean([veh_test["weighted-huber"], veh_test["residual-huber"]], axis=0)
    veh_val["blend-huber-div"] = np.mean([veh_val["weighted-huber-div"], veh_val["residual-huber-div"]], axis=0)
    veh_test["blend-huber-div"] = np.mean([veh_test["weighted-huber-div"], veh_test["residual-huber-div"]], axis=0)
    members = list(veh_val)   # refresh to include the two new blends in "ensemble" below
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
    chosen_candidate[sensor] = best

    if best == "multi-output":
        print(f"No extra artifact needed for {sensor}: the winner is the shared model, already saved.")
    else:
        save_best_specialist(sensor, best, members, val_rmse, veh_test, y_true[:, j], j)

raw_now = test_raw[SENSORS].to_numpy()                     

persist_pred = np.expm1(raw_now[test_end])

seasonal_pred = np.expm1(raw_now[test_end - (SEASONAL_LAG - HORIZON)])

def hour_of_week(frame):
    dt = frame["datetime"]
    return (dt.dt.dayofweek * 24 + dt.dt.hour).to_numpy()


profile = train_raw.groupby(hour_of_week(train_raw))[SENSORS].median().reindex(range(168))
assert not profile.isna().any().any(), "train split is missing some hour-of-week bins"
profile = profile.to_numpy()                               

how_now = hour_of_week(test_raw)[test_end]
how_target = (how_now + HORIZON) % 168
profile_pred = np.clip(
    np.expm1(raw_now[test_end] + profile[how_target] - profile[how_now]), 0, None
)

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

training_summary = {
    "run_at": datetime.now(timezone.utc).isoformat(),
    "config": {"WINDOW": WINDOW, "HORIZON": HORIZON, "SEED": SEED},
    "shared_model_path": str(SHARED_MODEL_PATH.relative_to(MODELS_DIR)),
    "feature_scaler_path": "feature_scaler.pkl",
    "target_scaler_path": "target_scaler.pkl",
    "sensors": {},
}
for s in SENSORS:
    winner = chosen_candidate[s]
    training_summary["sensors"][s] = {
        "winning_candidate": winner,
        "test_mae": lstm_res[s]["mae"],
        "test_rmse": lstm_res[s]["rmse"],
        "test_r2": lstm_res[s]["r2"],
        "test_wape": lstm_res[s]["wape"],
        "artifact": (
            str(SHARED_MODEL_PATH.relative_to(MODELS_DIR)) if winner == "multi-output"
            else str((MODELS_DIR / "specialists" / s / "manifest.json").relative_to(MODELS_DIR))
        ),
    }
training_summary["baselines"] = baseline_res

summary_path = MODELS_DIR / "training_summary.json"
with open(summary_path, "w") as f:
    json.dump(training_summary, f, indent=2)

print(f"\nSaved run summary -> {summary_path}")
print("Per-sensor winning artifact:")
for s in SENSORS:
    print(f"  {s}: {chosen_candidate[s]} -> {training_summary['sensors'][s]['artifact']}")