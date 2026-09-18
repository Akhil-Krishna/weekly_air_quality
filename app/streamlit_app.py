"""
Air Quality Risk Predictor -- Streamlit App

Two views:
  - Historical: pick a past date/time, see actual weather + pollution +
    predicted risk vs. the real outcome.
  - Forecast: pulls live Open-Meteo forecast data and predicts upcoming risk,
    turning this into a genuine decision-support tool rather than a static
    dashboard.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import os
import json
import datetime as dt

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.feature_engineering import add_time_features, add_rolling_features, add_lag_features
from src.fetch_weather import fetch_weather_forecast
from src.explainability import global_feature_importance

RISK_COLORS = {
    "Good": "#00A651",
    "Moderate": "#FFD400",
    "Unhealthy (Sensitive Groups)": "#FF8C00",
    "Unhealthy": "#E3312C",
    "Very Unhealthy": "#8B008B",
    "Hazardous": "#7E0023",
}

st.set_page_config(page_title="Air Quality Risk Predictor", page_icon="\U0001F32B", layout="wide")


@st.cache_resource
def load_model_artifacts():
    model = joblib.load(config.BEST_MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    le = joblib.load(config.LABEL_ENCODER_PATH)
    with open(config.FEATURE_LIST_PATH) as f:
        feature_cols = json.load(f)
    with open(config.METRICS_PATH) as f:
        metrics = json.load(f)
    return model, scaler, le, feature_cols, metrics


@st.cache_data
def load_historical_data():
    df = pd.read_csv(config.FEATURES_PATH, parse_dates=["datetime"])
    return df


def risk_badge(category):
    color = RISK_COLORS.get(category, "#999999")
    st.markdown(
        f"""
        <div style="background-color:{color}; padding: 18px; border-radius: 12px;
                    text-align:center; color:white; font-size:22px; font-weight:700;">
            {category}
        </div>
        """,
        unsafe_allow_html=True,
    )


def predict_risk(model, scaler, le, feature_cols, row_df):
    X = scaler.transform(row_df[feature_cols])
    pred_idx = model.predict(X)
    return le.inverse_transform(pred_idx)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("\U0001F32B Air Quality Risk Predictor")
st.caption(f"City: **{config.CITY_NAME}**  |  Coordinates: {config.LATITUDE}, {config.LONGITUDE}")

if not (os.path.exists(config.BEST_MODEL_PATH) and os.path.exists(config.FEATURES_PATH)):
    st.error(
        "No trained model / feature data found yet.\n\n"
        "Run the pipeline first:\n\n"
        "```\npython run_pipeline.py\n```\n\n"
        "This will fetch data, engineer features, train models, and save the best one "
        "so this app has something to load."
    )
    st.stop()

model, scaler, le, feature_cols, metrics = load_model_artifacts()
hist_df = load_historical_data()

with st.sidebar:
    st.header("Model info")
    st.write(f"**Best model:** {metrics['best_model']}")
    best_result = metrics["results"][metrics["best_model"]]
    st.metric("Macro F1 (test set)", f"{best_result['f1_macro']:.3f}")
    st.metric("Macro Precision", f"{best_result['precision_macro']:.3f}")
    st.metric("Macro Recall", f"{best_result['recall_macro']:.3f}")
    st.caption(f"Test period starts: {config.SPLIT_DATE}")

    st.divider()
    st.subheader("Model comparison")
    comp_rows = []
    for name, r in metrics["results"].items():
        comp_rows.append({
            "Model": name,
            "F1 (macro)": round(r["f1_macro"], 3),
            "Precision": round(r["precision_macro"], 3),
            "Recall": round(r["recall_macro"], 3),
        })
    st.dataframe(pd.DataFrame(comp_rows).set_index("Model"), use_container_width=True)

tab_hist, tab_forecast, tab_explain = st.tabs(
    ["\U0001F4C5 Historical View", "\U0001F52E Forecast View", "\U0001F9E0 Explainability"]
)

# ---------------------------------------------------------------------------
# TAB 1: Historical View
# ---------------------------------------------------------------------------
with tab_hist:
    st.subheader("Explore a past date/time")

    min_dt = hist_df["datetime"].min().to_pydatetime()
    max_dt = hist_df["datetime"].max().to_pydatetime()

    col1, col2 = st.columns(2)
    with col1:
        picked_date = st.date_input(
            "Date", value=max_dt.date(), min_value=min_dt.date(), max_value=max_dt.date()
        )
    with col2:
        picked_hour = st.slider("Hour of day", 0, 23, 12)

    picked_dt = pd.Timestamp(dt.datetime.combine(picked_date, dt.time(hour=picked_hour)))
    nearest_row = hist_df.iloc[(hist_df["datetime"] - picked_dt).abs().argsort()[:1]]

    if nearest_row.empty:
        st.warning("No data available near that date/time.")
    else:
        row = nearest_row.iloc[0]
        actual_dt = row["datetime"]
        st.caption(f"Showing nearest available record: **{actual_dt}**")

        pred_category = predict_risk(model, scaler, le, feature_cols, nearest_row)[0]
        actual_category = row["risk_category"]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Actual weather**")
            st.write(f"Temperature: {row.get('temperature_2m', float('nan')):.1f} °C")
            st.write(f"Humidity: {row.get('relative_humidity_2m', float('nan')):.0f}%")
            st.write(f"Wind speed: {row.get('wind_speed_10m', float('nan')):.1f} km/h")
            st.write(f"Pressure: {row.get('surface_pressure', float('nan')):.0f} hPa")
        with c2:
            st.markdown("**Actual pollution reading**")
            st.write(f"PM2.5: {row.get('pm25', float('nan')):.1f} µg/m³")
            st.write(f"PM10: {row.get('pm10', float('nan')):.1f} µg/m³")
            st.write(f"NO2: {row.get('no2', float('nan')):.1f} µg/m³")
            st.write(f"AQI: {row.get('aqi', float('nan')):.0f}")
        with c3:
            st.markdown("**Predicted risk**")
            risk_badge(pred_category)
            st.write("")
            st.markdown("**Actual risk**")
            risk_badge(actual_category)
            if pred_category == actual_category:
                st.success("Prediction matched the actual outcome.")
            else:
                st.warning("Prediction did not match the actual outcome.")

        st.divider()
        st.subheader("Recent trend")
        window_days = st.slider("Trend window (days)", 3, 30, 14, key="hist_window")
        trend_df = hist_df[
            (hist_df["datetime"] >= actual_dt - pd.Timedelta(days=window_days)) &
            (hist_df["datetime"] <= actual_dt)
        ]
        fig = px.line(trend_df, x="datetime", y="aqi", title="AQI trend")
        fig.add_hline(y=50, line_dash="dot", line_color="green", annotation_text="Good")
        fig.add_hline(y=100, line_dash="dot", line_color="gold", annotation_text="Moderate")
        fig.add_hline(y=150, line_dash="dot", line_color="orange", annotation_text="USG")
# ---------------------------------------------------------------------------
# TAB 2: Forecast View
# ---------------------------------------------------------------------------
with tab_forecast:
    st.subheader("Upcoming risk forecast")
    st.caption(
        "Pulls live forecast weather from Open-Meteo and predicts risk ahead of time. "
        "Note: since future pollutant readings are unknown, the model carries forward "
        "the most recent known pollution rolling-average/lag features as a reasonable "
        "assumption (weather features use the real forecast)."
    )

    forecast_days = st.slider("Forecast horizon (days)", 1, 7, 3)

    if st.button("Fetch forecast & predict", type="primary"):
        with st.spinner("Fetching forecast from Open-Meteo..."):
            try:
                fc_weather = fetch_weather_forecast(
                    config.LATITUDE, config.LONGITUDE,
                    config.WEATHER_HOURLY_VARS, forecast_days=forecast_days,
                )
            except Exception as e:
                st.error(f"Could not fetch forecast data: {e}")
                st.stop()

        merged_hist = pd.read_csv(config.MERGED_PATH, parse_dates=["datetime"])
        weather_cols = ["temperature_2m", "relative_humidity_2m", "surface_pressure",
                         "wind_speed_10m", "precipitation"]
        pollutant_cols = [c for c in ["pm25", "pm10", "no2"] if c in merged_hist.columns]

        # Combine a tail of real historical weather with the forecast so rolling
        # windows at the start of the forecast horizon are still meaningful.
        tail = merged_hist[["datetime"] + weather_cols + pollutant_cols].tail(24 * 8).copy()
        combined_weather = pd.concat(
            [tail[["datetime"] + weather_cols], fc_weather[["datetime"] + weather_cols]],
            ignore_index=True,
        ).drop_duplicates(subset="datetime").sort_values("datetime")

        combined_weather = add_rolling_features(combined_weather, weather_cols, windows_hours=(24, 24 * 7))
        combined_weather = add_lag_features(combined_weather, weather_cols, lags_hours=(24,))

        # Pollutant rolling/lag features: carry forward the last known real values,
        # since future pollution is exactly what we're trying to predict.
        tail_pollutant_feats = add_rolling_features(tail, pollutant_cols, windows_hours=(24, 24 * 7))
        tail_pollutant_feats = add_lag_features(tail_pollutant_feats, pollutant_cols, lags_hours=(24,))
        pollutant_feat_cols = [c for c in tail_pollutant_feats.columns if "_roll_" in c or "_lag_" in c]
        pollutant_feat_cols = [c for c in pollutant_feat_cols
                                if any(p in c for p in pollutant_cols)]
        last_known_pollutant_feats = tail_pollutant_feats[pollutant_feat_cols].dropna().iloc[-1]

        fc_df = combined_weather[combined_weather["datetime"].isin(fc_weather["datetime"])].copy()
        for col, val in last_known_pollutant_feats.items():
            fc_df[col] = val

        fc_df = add_time_features(fc_df)

        # Align columns with what the model expects
        for col in feature_cols:
            if col not in fc_df.columns:
                fc_df[col] = 0
        fc_df = fc_df.dropna(subset=[c for c in feature_cols if c in fc_df.columns])

        if fc_df.empty:
            st.warning("Not enough data to build forecast features (try a shorter horizon).")
        else:
            preds = predict_risk(model, scaler, le, feature_cols, fc_df)
            fc_df["predicted_risk"] = preds

            st.markdown("#### Next few hours")
            preview_cols = st.columns(min(6, len(fc_df)))
            for i, col in enumerate(preview_cols):
                if i >= len(fc_df):
                    break
                r = fc_df.iloc[i]
                with col:
                    st.caption(pd.Timestamp(r["datetime"]).strftime("%a %H:%M"))
                    risk_badge(r["predicted_risk"])

            st.divider()
            st.markdown("#### Full forecast timeline")
            fig = px.scatter(
                fc_df, x="datetime", y="temperature_2m", color="predicted_risk",
                color_discrete_map=RISK_COLORS,
                title="Predicted risk over the forecast window (y = temperature, for context)",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                fc_df[["datetime", "temperature_2m", "relative_humidity_2m",
                       "wind_speed_10m", "predicted_risk"]].rename(columns={
                    "temperature_2m": "Temp (°C)",
                    "relative_humidity_2m": "Humidity (%)",
                    "wind_speed_10m": "Wind (km/h)",
                    "predicted_risk": "Predicted Risk",
                    "datetime": "Time",
                }),
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# TAB 3: Explainability
# ---------------------------------------------------------------------------
with tab_explain:
    st.subheader("What drives the model's predictions?")
    imp_df = global_feature_importance(model, feature_cols, top_n=15)
    fig = px.bar(
        imp_df.sort_values("importance"), x="importance", y="feature",
        orientation="h", title=f"Top 15 features -- {metrics['best_model']} (global importance)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "This shows which weather and pollution-history variables most influence "
        "the model's risk predictions overall. For a true per-prediction SHAP "
        "breakdown, see `src/explainability.py::shap_explain`, which can be wired "
        "into either tab above to explain one specific row/prediction."
    )
