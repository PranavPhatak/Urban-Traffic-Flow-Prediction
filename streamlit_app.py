import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("TRAFFIC_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Traffic Flow Forecaster", page_icon="🚦", layout="centered")
st.title("🚦 GD0501 Traffic Flow Forecaster")
st.caption("LSTM · 1-hour-ahead vehicle flow · sensors B, C, D")

with st.sidebar:
    st.subheader("API status")
    health = None
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        if health["status"] == "ok":
            st.success("Connected — models & scalers loaded")
        else:
            st.error(f"API error: {health.get('error')}")
        st.write(f"Sensors: **{', '.join(health['sensors'])}**")
        st.write(f"Window: **{health['window']}** hours")
        st.write(f"Horizon: **{health['horizon']}** hour(s) ahead")
        st.write(f"Min history needed: **{health['min_history_hours_required']}** hours")
        if health["specialist_sensors"]:
            st.write(f"Dedicated models for: **{', '.join(health['specialist_sensors'])}**")
        else:
            st.write("No sensor has a dedicated model — all use the shared model.")
    except Exception:
        st.error(f"Cannot reach API at {API_URL}\nStart it with:\n`uvicorn api:app --reload`")

    st.divider()
    if st.button("Show winning model per sensor"):
        try:
            m = requests.get(f"{API_URL}/manifest", timeout=5).json()
            for s, info in m["sensors"].items():
                st.write(
                    f"**{s}**: `{info['winning_candidate']}` "
                    f"(test MAE {info['test_mae']:.1f}, R² {info['test_r2']:.3f})"
                )
        except Exception as e:
            st.warning(f"Couldn't load /manifest: {e}")

SENSORS = health["sensors"] if health and health.get("sensors") else ["GD0501_B", "GD0501_C", "GD0501_D"]
MIN_HOURS = health["min_history_hours_required"] if health else 72

st.write(
    f"Upload an **hourly, contiguous** CSV with columns `datetime` + one column per "
    f"sensor ({', '.join(SENSORS)}), raw vehicle counts (not log-transformed — the API "
    f"does that internally). You need at least **{MIN_HOURS} consecutive hours**."
)

uploaded = st.file_uploader("Traffic history CSV", type=["csv"])

with st.expander("Don't have a CSV handy? Download a synthetic template"):
    st.caption(
        "This is randomly generated data purely to show the expected shape and "
        "column names — it is NOT real traffic data, so don't read anything into "
        "the resulting prediction. Use it to check the app runs end-to-end."
    )
    n_hours = max(MIN_HOURS, 72)
    rng = pd.date_range(datetime(2024, 1, 1), periods=n_hours, freq="h")
    rs = np.random.RandomState(0)
    template = pd.DataFrame({"datetime": rng.strftime("%Y-%m-%dT%H:%M:%S")})
    for s in SENSORS:
        base = 20 + 15 * np.sin(2 * np.pi * rng.hour / 24)
        template[s] = np.clip(base + rs.normal(0, 5, size=n_hours), 0, None).round(1)
    st.download_button(
        "Download template.csv",
        template.to_csv(index=False).encode(),
        file_name="traffic_template.csv",
        mime="text/csv",
    )

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write(f"Loaded **{len(df)}** rows.")

    missing = [c for c in ["datetime"] + SENSORS if c not in df.columns]
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
    else:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        st.dataframe(df.tail(10), use_container_width=True)

        # quick look at recent history for each sensor
        chart_df = df.set_index("datetime")[SENSORS].tail(72)
        st.line_chart(chart_df)

        if len(df) < MIN_HOURS:
            st.error(f"Need at least {MIN_HOURS} hourly rows, got {len(df)}.")
        elif st.button("Forecast next hour", type="primary"):
            rows = df[["datetime"] + SENSORS].copy()
            rows["datetime"] = rows["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S")
            payload = rows.to_dict(orient="records")
            with st.spinner("Running inference..."):
                try:
                    resp = requests.post(f"{API_URL}/predict_csv", json=payload, timeout=60)
                    resp.raise_for_status()
                    result = resp.json()

                    st.subheader(f"Forecast for {result['target_datetime']}")
                    cols = st.columns(len(result["predictions"]))
                    for col, (sensor, pred) in zip(cols, result["predictions"].items()):
                        with col:
                            last_actual = df[sensor].iloc[-1]
                            st.metric(
                                sensor,
                                f"{pred['predicted_vehicles_per_hour']:.1f} veh/hr",
                                delta=f"{pred['predicted_vehicles_per_hour'] - last_actual:+.1f} vs last hour",
                            )
                            st.caption(f"model: {pred['model_used']}")
                except requests.exceptions.RequestException as e:
                    detail = ""
                    try:
                        detail = e.response.json().get("detail", "")
                    except Exception:
                        pass
                    st.error(f"Request failed: {e}\n{detail}")

st.divider()
st.caption(
    "Shared model predicts B and C directly. D uses whichever candidate "
    "(shared model or a dedicated specialist blend) scored best on validation "
    "during training — check the sidebar for which one is currently active."
)
