import os
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

MODELS_DIR = Path(os.environ.get("TRAFFIC_MODELS_DIR", "./models")).resolve()

ROLLING_STD_WINDOW = 24 


def _resolve_model_path(rel_path: str) -> Path:
    normalized = rel_path.replace("\\", "/")
    candidate = MODELS_DIR / normalized
    if candidate.exists():
        return candidate
    flat = MODELS_DIR / Path(normalized).name
    if flat.exists():
        return flat
    raise FileNotFoundError(
        f"Could not find model file for '{rel_path}' at {candidate} or {flat}"
    )

app = FastAPI(
    title="GD0501 Traffic Flow Forecast API",
    description="1-hour-ahead vehicle-flow forecasts for sensors GD0501_B/C/D.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_shared_model = None
_feature_scaler = None
_target_scaler = None
_specialists: Dict[str, dict] = {}     
SENSORS: List[str] = []
FEATURE_COLS: List[str] = []
WINDOW = None
HORIZON = None
MIN_HISTORY_HOURS = None
_load_error = None

try:
    _shared_model = load_model(MODELS_DIR / "shared_multioutput_lstm.keras")
    _feature_scaler = joblib.load(MODELS_DIR / "feature_scaler.pkl")
    _target_scaler = joblib.load(MODELS_DIR / "target_scaler.pkl")

    FEATURE_COLS = list(_feature_scaler.feature_names_in_)
    SENSORS = [c[: -len("_target")] for c in _target_scaler.feature_names_in_]
    WINDOW = _shared_model.input_shape[1]
    HORIZON = 1  
    MIN_HISTORY_HOURS = WINDOW + ROLLING_STD_WINDOW

    specialists_dir = MODELS_DIR / "specialists"
    if specialists_dir.exists():
        for sensor_dir in specialists_dir.iterdir():
            manifest_path = sensor_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            import json
            manifest = json.loads(manifest_path.read_text())
            leaf_models: Dict[str, list] = {}
            for comp in manifest["components"]:
                if comp["leaf"] == "multi-output":
                    continue
                leaf_models[comp["leaf"]] = [
                    load_model(_resolve_model_path(p)) for p in comp["seed_model_paths"]
                ]
            _specialists[manifest["sensor"]] = {"manifest": manifest, "leaf_models": leaf_models}

except Exception as exc:  # noqa: BLE001
    _load_error = f"Failed to load models/scalers: {exc}"

def _validate_contiguous_hourly(dt: pd.Series) -> None:
    gaps = dt.diff().iloc[1:]
    if not (gaps == pd.Timedelta(hours=1)).all():
        bad = gaps[gaps != pd.Timedelta(hours=1)]
        raise HTTPException(
            status_code=400,
            detail=(
                "Input records must be exactly hourly with no gaps or duplicates. "
                f"Found a break of {bad.iloc[0]} after row {bad.index[0]}."
            ),
        )


def build_feature_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.sort_values("datetime").reset_index(drop=True).copy()
    _validate_contiguous_hourly(df["datetime"])

    df[SENSORS] = np.log1p(df[SENSORS])

    for sensor in SENSORS:
        df[f"{sensor}_rolling_std_24"] = df[sensor].shift(1).rolling(ROLLING_STD_WINDOW).std()

    for sensor in SENSORS:
        others = [s for s in SENSORS if s != sensor]
        df[f"{sensor}_neighbor_mean"] = df[others].mean(axis=1)

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

    df = df.dropna().reset_index(drop=True) 
    return df


def scaled_to_vehicles(y_scaled: float, j: int) -> float:
    log_val = y_scaled * _target_scaler.scale_[j] + _target_scaler.mean_[j]
    return float(np.clip(np.expm1(log_val), 0, None))


def residual_to_vehicles(delta_scaled: float, now_log: float, res_mean: float, res_std: float) -> float:
    log_pred = now_log + delta_scaled * res_std + res_mean
    return float(np.clip(np.expm1(log_pred), 0, None))

class HourlyRecord(BaseModel):
    datetime: str 
    values: Dict[str, float] = Field(..., description="One entry per sensor, e.g. {'GD0501_B': 42.0, ...}")


class PredictRequest(BaseModel):
    records: List[HourlyRecord] = Field(
        ...,
        description=(
            "Hourly, contiguous, oldest first. Needs at least WINDOW + 24 hours "
            "(see /health) so every row's 24h rolling std can be computed."
        ),
    )


class SensorPrediction(BaseModel):
    predicted_vehicles_per_hour: float
    model_used: str


class PredictResponse(BaseModel):
    target_datetime: str
    predictions: Dict[str, SensorPrediction]
    hours_received: int
    hours_used_in_window: int


class HealthResponse(BaseModel):
    status: str
    sensors: List[str]
    window: Optional[int]
    horizon: Optional[int]
    min_history_hours_required: Optional[int]
    specialist_sensors: List[str]
    error: Optional[str] = None

def _predict_from_frame(feat_df: pd.DataFrame, raw_hours_count: int) -> PredictResponse:
    if len(feat_df) < WINDOW:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Only {len(feat_df)} usable hourly rows after feature engineering "
                f"(need >= {WINDOW}). Send at least {MIN_HISTORY_HOURS} raw hours."
            ),
        )

    window_df = feat_df.iloc[-WINDOW:]
    X_scaled = _feature_scaler.transform(window_df[FEATURE_COLS]).astype(np.float32)
    X_scaled = X_scaled.reshape(1, WINDOW, len(FEATURE_COLS))

    shared_scaled_out = _shared_model.predict(X_scaled, verbose=0)[0] 
    raw_log_window = window_df[SENSORS].to_numpy(dtype=np.float32)     

    predictions: Dict[str, SensorPrediction] = {}
    for j, sensor in enumerate(SENSORS):
        shared_vehicles = scaled_to_vehicles(shared_scaled_out[j], j)

        if sensor not in _specialists:
            predictions[sensor] = SensorPrediction(
                predicted_vehicles_per_hour=round(shared_vehicles, 2),
                model_used="multi-output (shared model)",
            )
            continue

        manifest = _specialists[sensor]["manifest"]
        leaf_models = _specialists[sensor]["leaf_models"]
        now_log = float(raw_log_window[-1, j])   
        res_stats = manifest.get("residual_stats")
        div_stats = manifest.get("divergence_stats")

        total = 0.0
        for comp in manifest["components"]:
            leaf, weight = comp["leaf"], comp["weight"]
            if leaf == "multi-output":
                total += weight * shared_vehicles
                continue

            models_for_leaf = leaf_models[leaf]
            X_leaf = X_scaled
            if comp["use_divergence"]:
                others = [s for s in SENSORS if s != sensor]
                other_idx = [SENSORS.index(s) for s in others]
                divergence = raw_log_window[:, other_idx].std(axis=1)  
                divergence = (divergence - div_stats["mean"]) / div_stats["std"]
                X_leaf = np.concatenate(
                    [X_scaled, divergence.reshape(1, WINDOW, 1).astype(np.float32)], axis=-1
                )

            seed_vehicles = []
            for m in models_for_leaf:
                out = float(m.predict(X_leaf, verbose=0)[0, 0])
                if comp["residual"]:
                    seed_vehicles.append(
                        residual_to_vehicles(out, now_log, res_stats["mean"], res_stats["std"])
                    )
                else:
                    seed_vehicles.append(scaled_to_vehicles(out, j))
            total += weight * float(np.mean(seed_vehicles))

        predictions[sensor] = SensorPrediction(
            predicted_vehicles_per_hour=round(total, 2),
            model_used=manifest["selected_candidate"],
        )

    target_dt = pd.Timestamp(feat_df["datetime"].iloc[-1]) + timedelta(hours=HORIZON)
    return PredictResponse(
        target_datetime=target_dt.isoformat(),
        predictions=predictions,
        hours_received=raw_hours_count,
        hours_used_in_window=WINDOW,
    )

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if _load_error is None else "error",
        sensors=SENSORS,
        window=WINDOW,
        horizon=HORIZON,
        min_history_hours_required=MIN_HISTORY_HOURS,
        specialist_sensors=list(_specialists.keys()),
        error=_load_error,
    )


@app.get("/manifest")
def manifest():
    summary_path = MODELS_DIR / "training_summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="training_summary.json not found in MODELS_DIR")
    import json
    return json.loads(summary_path.read_text())


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _load_error is not None:
        raise HTTPException(status_code=503, detail=_load_error)

    rows = []
    for r in req.records:
        row = {"datetime": pd.Timestamp(r.datetime)}
        for sensor in SENSORS:
            if sensor not in r.values:
                raise HTTPException(status_code=400, detail=f"Missing sensor '{sensor}' in a record")
            row[sensor] = r.values[sensor]
        rows.append(row)
    raw_df = pd.DataFrame(rows)

    if len(raw_df) < MIN_HISTORY_HOURS:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {MIN_HISTORY_HOURS} hourly records, got {len(raw_df)}.",
        )

    feat_df = build_feature_frame(raw_df)
    return _predict_from_frame(feat_df, raw_hours_count=len(raw_df))


@app.post("/predict_csv", response_model=PredictResponse)
def predict_csv(rows: List[dict]):
    if _load_error is not None:
        raise HTTPException(status_code=503, detail=_load_error)

    try:
        raw_df = pd.DataFrame(
            [{"datetime": pd.Timestamp(r["datetime"]), **{s: float(r[s]) for s in SENSORS}} for r in rows]
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing required column: {exc}") from exc

    if len(raw_df) < MIN_HISTORY_HOURS:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {MIN_HISTORY_HOURS} hourly records, got {len(raw_df)}.",
        )

    feat_df = build_feature_frame(raw_df)
    return _predict_from_frame(feat_df, raw_hours_count=len(raw_df))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
