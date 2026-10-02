"""
Air Quality Risk Predictor -- Streamlit App

Five tabs:
  - Historical: pick a past date/time, see actual weather + pollution +
    predicted risk vs. the real outcome.
  - Forecast: pulls live Open-Meteo forecast data and predicts upcoming risk,
    turning this into a genuine decision-support tool rather than a static
    dashboard.
  - Explainability: which features actually drive the model's predictions.
  - City Comparison: Delhi vs. Auckland, real computed statistics.
  - Allergy Outlook: live species-level pollen forecasts (via the Atmospore
    API, with a daily call budget + cache fallback), alongside an honest
    comparison to MetService's own pollen forecasting.
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
from src.feature_engineering import (
    add_time_features, add_rolling_features, add_lag_features,
    POLLUTANT_COLS, POLLUTANT_ROLL_SHIFT_HOURS, WEATHER_FEATURE_COLS,
)
from src.labeling import RISK_CATEGORY_ORDER
from src.utils import read_pipeline_csv
from src.fetch_weather import fetch_weather_forecast
from src.explainability import global_feature_importance, explain_single_prediction
from src.fetch_pollen import get_pollen_forecast, get_call_budget_status, POLLEN_RISK_COLORS

# How far behind config.END_DATE a city's pollution data may fall before the
# forecast tab discloses it. END_DATE is already today-7 by design, so this is
# measured against END_DATE, not against the current time -- see the comment at
# the staleness check in the Forecast tab.
#
# Cities collecting normally sit at or slightly ahead of END_DATE, so nothing
# is shown. It exists for the case where a city's data genuinely stops early,
# which a reader of a forecast deserves to be told about.
STALE_POLLUTION_DAYS = 30

RISK_COLORS = {
    "Good": "#00A651",
    "Moderate": "#FFD400",
    "Unhealthy (Sensitive Groups)": "#FF8C00",
    "Unhealthy": "#E3312C",
    "Very Unhealthy": "#8B008B",
    "Hazardous": "#7E0023",
}

st.set_page_config(page_title="Air Quality Risk Predictor", layout="wide")


def get_city_paths(city_slug):
    """Mirrors config.py's own per-city path pattern, but computed for
    whichever city is selected live in the app -- not just the one CITY_NAME
    config.py happens to be set to."""
    data_processed_dir = os.path.join(config.BASE_DIR, "data", "processed", city_slug)
    models_dir = os.path.join(config.BASE_DIR, "models", city_slug)
    return {
        "features": os.path.join(data_processed_dir, "features.csv"),
        "merged": os.path.join(data_processed_dir, "merged_clean.csv"),
        "best_model": os.path.join(models_dir, "best_model.joblib"),
        "scaler": os.path.join(models_dir, "scaler.joblib"),
        "label_encoder": os.path.join(models_dir, "label_encoder.joblib"),
        "feature_list": os.path.join(models_dir, "feature_list.json"),
        "metrics": os.path.join(models_dir, "model_comparison.json"),
    }


# Every artifact load_model_artifacts() opens. has_model used to check only
# best_model.joblib and features.csv, so a city missing scaler.joblib or
# model_comparison.json took the "model available" branch and then crashed
# inside the loader instead of degrading to the no-model explanation.
REQUIRED_MODEL_ARTIFACTS = ("best_model", "scaler", "label_encoder",
                            "feature_list", "metrics", "features")


def city_has_model(city_slug):
    paths = get_city_paths(city_slug)
    return all(os.path.exists(paths[k]) for k in REQUIRED_MODEL_ARTIFACTS)


@st.cache_resource
def load_model_artifacts(city_slug):
    paths = get_city_paths(city_slug)
    model = joblib.load(paths["best_model"])
    scaler = joblib.load(paths["scaler"])
    le = joblib.load(paths["label_encoder"])
    with open(paths["feature_list"]) as f:
        feature_cols = json.load(f)
    with open(paths["metrics"]) as f:
        metrics = json.load(f)
    return model, scaler, le, feature_cols, metrics


@st.cache_data
def load_features_data(city_slug):
    return read_pipeline_csv(get_city_paths(city_slug)["features"])


@st.cache_data
def load_merged_data(city_slug):
    paths = get_city_paths(city_slug)
    if not os.path.exists(paths["merged"]):
        return None
    return read_pipeline_csv(paths["merged"])


@st.cache_data
def compute_city_summary(city_slug):
    """Real summary stats used by the 'no model' explanation and the City
    Comparison tab -- computed fresh from each city's actual merged data."""
    df = load_merged_data(city_slug)
    if df is None or df.empty:
        return None
    return {
        "rows": len(df),
        "date_min": df["datetime"].min(),
        "date_max": df["datetime"].max(),
        "mean_aqi": df["aqi"].mean(),
        "median_aqi": df["aqi"].median(),
        "category_pct": (df["risk_category"].value_counts(normalize=True) * 100).round(1).to_dict(),
        "source": df["source"].iloc[0] if "source" in df.columns else "unknown",
    }


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


def pollen_risk_badge(risk_level):
    color = POLLEN_RISK_COLORS.get(str(risk_level).lower(), "#999999")
    st.markdown(
        f"""
        <div style="background-color:{color}; padding: 18px; border-radius: 12px;
                    text-align:center; color:white; font-size:22px; font-weight:700;">
            {str(risk_level).title()}
        </div>
        """,
        unsafe_allow_html=True,
    )


def predict_risk(model, scaler, le, feature_cols, row_df):
    X = scaler.transform(row_df[feature_cols])
    pred_idx = model.predict(X)
    return le.inverse_transform(pred_idx)


# ---------------------------------------------------------------------------
# Header + City selector
# ---------------------------------------------------------------------------
st.title("Air Quality Risk Predictor")

city_names = list(config.CITIES.keys())
default_idx = city_names.index(config.CITY_NAME) if config.CITY_NAME in city_names else 0
selected_city = st.selectbox("City", city_names, index=default_idx)
city_slug = selected_city.strip().lower().replace(" ", "_")
lat = config.CITIES[selected_city]["lat"]
lon = config.CITIES[selected_city]["lon"]

st.caption(f"City: **{selected_city}**  |  Coordinates: {lat}, {lon}  |  "
           f"Local timezone: {config.city_timezone(selected_city)}  "
           f"(timestamps below are stored in UTC)")

paths = get_city_paths(city_slug)
has_model = city_has_model(city_slug)
city_tz = config.city_timezone(selected_city)
southern_hemisphere = lat < 0

model = scaler = le = feature_cols = metrics = None
if has_model:
    model, scaler, le, feature_cols, metrics = load_model_artifacts(city_slug)
    hist_df = load_features_data(city_slug)

with st.sidebar:
    st.header("Model info")
    if has_model:
        st.write(f"**Best model:** {metrics['best_model']}")
        best_result = metrics["results"][metrics["best_model"]]
        st.metric("Macro F1 (test set)", f"{best_result['f1_macro']:.3f}")
        st.metric("Macro Precision", f"{best_result['precision_macro']:.3f}")
        st.metric("Macro Recall", f"{best_result['recall_macro']:.3f}")
        # Read the split out of the saved metadata rather than recomputing it.
        # The app used to display a freshly-derived config.SPLIT_DATE, which
        # drifted with the wall clock and so disagreed with the split the saved
        # model had actually been evaluated on.
        split_date = metrics.get("split_date")
        if split_date:
            st.caption(f"Test period: {str(split_date)[:16]} to "
                       f"{str(metrics.get('test_end', ''))[:16]} [UTC]")
            st.caption(f"{metrics.get('train_rows', '?')} train rows / "
                       f"{metrics.get('test_rows', '?')} test rows")
        else:
            st.caption("Test period: not recorded in this model artifact -- "
                       "re-run the pipeline to record it.")
        n_present = best_result.get("n_test_classes_present")
        if n_present:
            st.caption(f"Macro scores averaged over the {n_present} risk "
                       f"categor(y/ies) present in the test set.")

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
        st.dataframe(pd.DataFrame(comp_rows).set_index("Model"), width="stretch")
    else:
        st.caption(f"No trained model for **{selected_city}** -- see the "
                   "Historical tab for why, and the City Comparison tab for "
                   "what its real data looks like instead.")

tab_hist, tab_forecast, tab_explain, tab_compare, tab_allergy = st.tabs([
    "Historical View", "Forecast View", "Explainability",
    "City Comparison", "Allergy Comparison (vs MetService)",
])

# ---------------------------------------------------------------------------
# TAB 1: Historical View
# ---------------------------------------------------------------------------
with tab_hist:
    if not has_model:
        st.warning(f"No trained model available for **{selected_city}**.")
        summary = compute_city_summary(city_slug)
        if summary is not None and summary["rows"] > 0:
            cats = summary["category_pct"]
            if len(cats) == 1:
                only_cat = list(cats.keys())[0]
                st.info(
                    f"**Why no model:** {selected_city}'s entire dataset "
                    f"({summary['rows']} hourly readings, "
                    f"{summary['date_min'].date()} to {summary['date_max'].date()}) "
                    f"fell into a single risk category (**{only_cat}**, "
                    f"{cats[only_cat]}% of hours). Classification models need at "
                    "least 2 classes to learn to distinguish -- there's nothing to "
                    "learn here, so training was correctly skipped rather than "
                    f"forced. Mean AQI was **{summary['mean_aqi']:.1f}** across the "
                    "whole period -- genuinely clean air, not a data problem. See "
                    "the City Comparison tab for how this contrasts with Delhi."
                )
            else:
                st.info(
                    f"{selected_city} has real data with multiple risk categories "
                    "but hasn't been trained on yet. Point config.py at this city "
                    "and run `python run_pipeline.py`."
                )
            st.subheader(f"What {selected_city}'s raw data actually looks like")
            merged = load_merged_data(city_slug)
            fig = px.line(merged, x="datetime", y="aqi",
                          title=f"{selected_city} AQI over time (real data)")
            st.plotly_chart(fig, width="stretch")
        else:
            st.error("No data at all for this city yet. Run the pipeline first.")
    else:
        st.subheader("Explore a past date/time")

        # Dates and hours are picked on the LOCAL clock, because that is the
        # clock a person reading this thinks in. The underlying datetime column
        # stays UTC -- that is the key weather and pollution were joined on.
        if "datetime_local" not in hist_df.columns:
            st.error(
                "This city's features.csv predates the timezone fix and has no "
                "datetime_local column. Re-run: python run_pipeline.py"
            )
            st.stop()

        min_dt = hist_df["datetime_local"].min().to_pydatetime()
        max_dt = hist_df["datetime_local"].max().to_pydatetime()

        # Months that actually contain data, newest first.
        month_starts = (hist_df["datetime_local"].dt.to_period("M")
                        .drop_duplicates().sort_values(ascending=False))
        month_labels = [p.strftime("%B %Y") for p in month_starts]
        label_to_period = dict(zip(month_labels, month_starts))

        st.caption(
            f"Data available from {min_dt:%d %b %Y} to {max_dt:%d %b %Y} "
            f"({city_tz} local time) -- {len(month_labels)} months."
        )

        # Keys are per-city so switching cities cannot leave a stored date
        # outside the new city's range, which st.date_input rejects outright.
        date_key = f"hist_date_{city_slug}"
        month_key = f"hist_month_{city_slug}"

        def _jump_to_month():
            """Move the date picker to the first day of the chosen month.

            Without this, reaching an older month means stepping the calendar
            back one month at a time -- and using the year dropdown lands on
            the same month number in that year, which for the earliest year is
            usually before min_value and so renders entirely greyed out. That
            made October-December 2025 look unavailable when they were not.
            """
            period = label_to_period.get(st.session_state.get(month_key))
            if period is None:
                return
            in_month = hist_df.loc[
                hist_df["datetime_local"].dt.to_period("M") == period, "datetime_local"
            ]
            if len(in_month):
                st.session_state[date_key] = in_month.min().date()

        if date_key not in st.session_state:
            st.session_state[date_key] = max_dt.date()

        col1, col2, col3 = st.columns([1.1, 1, 1])
        with col1:
            st.selectbox(
                "Jump to month", month_labels, key=month_key,
                on_change=_jump_to_month,
                help="Every month listed here contains data. Picking one moves "
                     "the date below to its first available day.",
            )
        with col2:
            picked_date = st.date_input(
                "Date", min_value=min_dt.date(), max_value=max_dt.date(),
                key=date_key,
            )
        with col3:
            picked_hour = st.slider("Hour of day (local)", 0, 23, 12)

        picked_dt = pd.Timestamp(dt.datetime.combine(picked_date, dt.time(hour=picked_hour)))
        nearest_row = hist_df.iloc[
            (hist_df["datetime_local"] - picked_dt).abs().argsort()[:1]
        ]

        if nearest_row.empty:
            st.warning("No data available near that date/time.")
        else:
            row = nearest_row.iloc[0]
            actual_dt = row["datetime"]
            st.caption(f"Showing nearest available record: "
                       f"**{row['datetime_local']}** local ({city_tz}) "
                       f"/ {actual_dt} UTC")

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

            with st.expander("Why did the model predict this? (per-prediction SHAP breakdown)"):
                try:
                    X_row_scaled = scaler.transform(nearest_row[feature_cols])
                    predicted_class_idx = le.transform([pred_category])[0]
                    shap_df = explain_single_prediction(
                        model, X_row_scaled, feature_cols, predicted_class_idx, top_n=10
                    )
                    fig_shap = px.bar(
                        shap_df.sort_values("shap_value"), x="shap_value", y="feature",
                        orientation="h",
                        title=f"Top factors behind this prediction of '{pred_category}'",
                        color="shap_value", color_continuous_scale=["#E3312C", "#DDDDDD", "#00A651"],
                    )
                    st.plotly_chart(fig_shap, width="stretch")
                    st.caption(
                        "Positive values pushed the prediction toward this specific risk "
                        "category for this specific date/hour; negative values pushed "
                        "away from it. This is different from the Explainability tab's "
                        "chart, which shows overall importance across all predictions, "
                        "not this one row."
                    )
                except RuntimeError as e:
                    st.info(f"Per-prediction explanation not available here: {e}")

            st.divider()
            st.subheader("Recent trend")
            window_days = st.slider("Trend window (days)", 3, 30, 14, key="hist_window")
            trend_df = hist_df[
                (hist_df["datetime"] >= actual_dt - pd.Timedelta(days=window_days)) &
                (hist_df["datetime"] <= actual_dt)
            ]
            fig = px.line(trend_df, x="datetime_local", y="aqi",
                          title=f"AQI trend ({city_tz} local time)")
            fig.add_hline(y=50, line_dash="dot", line_color="green", annotation_text="Good")
            fig.add_hline(y=100, line_dash="dot", line_color="gold", annotation_text="Moderate")
            fig.add_hline(y=150, line_dash="dot", line_color="orange", annotation_text="USG")
            fig.add_hline(y=200, line_dash="dot", line_color="red", annotation_text="Unhealthy")
            st.plotly_chart(fig, width="stretch")
# ---------------------------------------------------------------------------
# TAB 2: Forecast View
# ---------------------------------------------------------------------------
with tab_forecast:
    if not has_model:
        st.info(f"No trained model for **{selected_city}** -- forecast prediction "
                "isn't available here. See the Historical tab for why.")
    else:
        st.subheader("Upcoming risk forecast")
        st.caption(
            "Pulls live forecast weather from Open-Meteo and predicts risk ahead "
            "of time. Future pollutant readings are exactly what is being "
            "predicted, so the pollution-history features carry forward the most "
            "recent real measurements; the weather features are the real forecast."
        )

        forecast_days = st.slider("Forecast horizon (days)", 1, 7, 3)

        if st.button("Fetch forecast & predict", type="primary"):
            with st.spinner("Fetching forecast from Open-Meteo..."):
                try:
                    # past_days gives us the real hours immediately before the
                    # horizon, from the same endpoint and the same grid, so the
                    # 24h/7d rolling weather features are computed over one
                    # contiguous series.
                    #
                    # This used to splice the tail of merged_clean.csv onto the
                    # forecast and roll straight across the join. That file is
                    # only as fresh as the last pipeline run, so the 24h mean at
                    # the start of the horizon averaged across a months-wide
                    # hole -- and the hole grew every day it was not re-run.
                    fc_weather = fetch_weather_forecast(
                        lat, lon, config.WEATHER_HOURLY_VARS,
                        forecast_days=forecast_days, past_days=8,
                    )
                except Exception as e:
                    st.error(f"Could not fetch forecast data: {e}")
                    st.stop()

            # Use the full configured weather variable list. A local, shorter
            # copy of this list used to omit wind_direction_10m, which then got
            # silently backfilled with 0 -- asserting due north, every hour, to
            # a model trained on real wind directions.
            weather_cols = [c for c in WEATHER_FEATURE_COLS if c in fc_weather.columns]
            missing_weather = [c for c in config.WEATHER_HOURLY_VARS
                               if c not in fc_weather.columns]
            if missing_weather:
                st.error(
                    "Open-Meteo did not return these forecast variables that the "
                    f"model needs: {missing_weather}. Not predicting, rather than "
                    "substituting zeros for them."
                )
                st.stop()

            wx = add_rolling_features(fc_weather, weather_cols,
                                      windows_hours=(24, 24 * 7), shift_hours=0)
            wx = add_lag_features(wx, weather_cols, lags_hours=(24,))

            # Pollutant rolling/lag features carry forward the last known real
            # values, since future pollution is what we are trying to predict.
            merged_hist = load_merged_data(city_slug)
            pollutant_cols = [c for c in POLLUTANT_COLS
                              if merged_hist is not None and c in merged_hist.columns]
            last_known_pollutant_feats = None
            if pollutant_cols:
                hist = merged_hist[["datetime"] + pollutant_cols].copy()
                hist_feats = add_rolling_features(
                    hist, pollutant_cols, windows_hours=(24, 24 * 7),
                    shift_hours=POLLUTANT_ROLL_SHIFT_HOURS,
                )
                hist_feats = add_lag_features(hist_feats, pollutant_cols,
                                              lags_hours=(24,))
                pollutant_feat_cols = [
                    c for c in hist_feats.columns
                    if ("_roll_" in c or "_lag_" in c)
                    and any(c.startswith(pol + "_") for pol in pollutant_cols)
                ]
                last_known_pollutant_feats = (
                    hist_feats[pollutant_feat_cols].ffill().iloc[-1]
                )
                # A pollutant with no usable history at all (a sensor that
                # never reported) leaves NaN here, which would quietly drop
                # every forecast row at the dropna below. Fall back to that
                # feature's own historical mean and say so -- a mean is a
                # defensible stand-in for a 24h average, which is more than
                # could be said for the blanket zero this used to get.
                unavailable = [c for c in pollutant_feat_cols
                               if pd.isna(last_known_pollutant_feats[c])]
                if unavailable:
                    means = hist_feats[pollutant_feat_cols].mean()
                    for c in unavailable:
                        last_known_pollutant_feats[c] = means[c]
                    still_nan = [c for c in unavailable
                                 if pd.isna(last_known_pollutant_feats[c])]
                    if still_nan:
                        st.error(
                            f"{selected_city} has no pollution history at all "
                            f"for: {still_nan}. The model needs those features, "
                            "so no forecast is produced rather than one built "
                            "on invented numbers."
                        )
                        st.stop()
                    st.info(
                        f"No recent reading to carry forward for {len(unavailable)} "
                        "pollution-history feature(s); substituted each one's "
                        "historical mean. Shown for transparency."
                    )

                # Staleness is measured against config.END_DATE -- the newest
                # hour the pipeline is ALLOWED to collect -- not against now.
                #
                # END_DATE is date.today() - 7, because the archive APIs lag
                # real time. So the freshest possible reading is always about a
                # week old, and an earlier version of this check compared
                # against now with a 48-hour threshold: it fired on every
                # forecast for every city, including moments after a clean run,
                # and told the user to re-run the pipeline, which could not
                # have changed the outcome. Delhi reads 6.7 days old against
                # now but 0.9 days AHEAD of END_DATE: as fresh as it can be.
                last_reading = merged_hist["datetime"].max()
                days_behind = (pd.Timestamp(config.END_DATE) - last_reading).days
                if days_behind > STALE_POLLUTION_DAYS:
                    st.info(
                        f"Pollution readings for {selected_city} end on "
                        f"{last_reading:%d %b %Y}, about "
                        f"{days_behind / 30:.0f} month(s) before this forecast "
                        "window. The forecast below is driven by the live "
                        "weather forecast; its pollution baseline comes from "
                        "that earlier period, so treat the risk levels as "
                        "indicative."
                    )

            # The horizon runs from the current hour onward; the past_days block
            # exists only to give the rolling windows real history to stand on.
            now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
            fc_df = wx[wx["datetime"] >= now_utc].copy()

            if last_known_pollutant_feats is not None:
                for col, val in last_known_pollutant_feats.items():
                    fc_df[col] = val

            fc_df = add_time_features(fc_df, tz=city_tz,
                                      southern_hemisphere=southern_hemisphere)

            # Any feature still missing here is a real training/serving mismatch,
            # not something to paper over with a zero. add_time_features pins
            # every season and time_of_day level, so the dummies always exist.
            still_missing = [c for c in feature_cols if c not in fc_df.columns]
            if still_missing:
                st.error(
                    "These features the model was trained on could not be built "
                    f"for the forecast: {still_missing}. Refusing to substitute "
                    "zeros -- re-run the pipeline so training and serving agree."
                )
                st.stop()

            fc_df = fc_df.dropna(subset=feature_cols)

            if fc_df.empty:
                st.warning("Not enough contiguous recent weather to build the "
                           "rolling features for this horizon. Try again shortly, "
                           "or pick a shorter horizon.")
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
                        st.caption(pd.Timestamp(r["datetime_local"]).strftime("%a %H:%M"))
                        risk_badge(r["predicted_risk"])
                st.caption(f"Times shown in {city_tz} local time.")

                st.divider()
                st.markdown("#### Full forecast timeline")
                fig = px.scatter(
                    fc_df, x="datetime_local", y="temperature_2m",
                    color="predicted_risk", color_discrete_map=RISK_COLORS,
                    category_orders={"predicted_risk": RISK_CATEGORY_ORDER},
                    title="Predicted risk over the forecast window "
                          "(y = temperature, for context)",
                )
                fig.update_layout(xaxis_title=f"Local time ({city_tz})")
                st.plotly_chart(fig, width="stretch")

                st.dataframe(
                    fc_df[["datetime_local", "temperature_2m",
                           "relative_humidity_2m", "wind_speed_10m",
                           "wind_direction_10m", "predicted_risk"]].rename(columns={
                        "temperature_2m": "Temp (C)",
                        "relative_humidity_2m": "Humidity (%)",
                        "wind_speed_10m": "Wind (km/h)",
                        "wind_direction_10m": "Wind dir (deg)",
                        "predicted_risk": "Predicted Risk",
                        "datetime_local": f"Local time ({city_tz})",
                    }),
                    width="stretch",
                )

# ---------------------------------------------------------------------------
# TAB 3: Explainability
# ---------------------------------------------------------------------------
with tab_explain:
    if not has_model:
        st.info(f"No trained model for **{selected_city}** -- nothing to explain yet.")
    else:
        st.subheader("What drives the model's predictions?")
        imp_df = global_feature_importance(model, feature_cols, top_n=15)
        fig = px.bar(
            imp_df.sort_values("importance"), x="importance", y="feature",
            orientation="h", title=f"Top 15 features -- {metrics['best_model']} (global importance)",
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown(
            "This shows which weather and pollution-history variables most influence "
            "the model's risk predictions overall, averaged across every prediction. "
            "For a per-prediction breakdown -- why the model predicted a specific risk "
            "category for one specific date/hour -- see the 'Why did the model predict "
            "this?' expander in the Historical View tab."
        )

# ---------------------------------------------------------------------------
# TAB 4: City Comparison (Delhi vs Auckland case study)
# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader("City comparison: a real-world contrast")
    st.caption(
        "Real computed statistics from each city's own collected data -- not "
        "illustrative numbers. The motivation for running more than one city is "
        "the contrast itself: a high-variance, high-pollution city gives a "
        "classifier something to learn, and a clean-air one makes the problem "
        "itself disappear."
    )

    summaries = {c: compute_city_summary(c.strip().lower().replace(" ", "_")) for c in city_names}
    available = {c: s for c, s in summaries.items() if s is not None}

    if len(available) < 2:
        st.warning("Need data for at least 2 cities to compare. Run the pipeline "
                    "for more cities first.")
    else:
        cols = st.columns(len(available))
        for col, (city, s) in zip(cols, available.items()):
            with col:
                st.markdown(f"### {city}")
                st.metric("Mean AQI", f"{s['mean_aqi']:.1f}")
                st.metric("Median AQI", f"{s['median_aqi']:.1f}")
                st.caption(f"{s['rows']} hourly readings, "
                           f"{s['date_min'].date()} to {s['date_max'].date()}")
                st.caption(f"Data source: {s['source']}")

        st.divider()
        st.markdown("#### Mean AQI, side by side")
        bar_df = pd.DataFrame({
            "City": list(available.keys()),
            "Mean AQI": [s["mean_aqi"] for s in available.values()],
        })
        fig = px.bar(bar_df, x="City", y="Mean AQI", color="City",
                     title="Mean AQI over the full collection period")
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### Risk category distribution, side by side")
        cat_rows = []
        for city, s in available.items():
            for cat, pct in s["category_pct"].items():
                cat_rows.append({"City": city, "Risk category": cat, "% of hours": pct})
        cat_df = pd.DataFrame(cat_rows)
        fig2 = px.bar(cat_df, x="City", y="% of hours", color="Risk category",
                      color_discrete_map=RISK_COLORS,
                      category_orders={"Risk category": RISK_CATEGORY_ORDER},
                      title="Time spent in each risk category")
        st.plotly_chart(fig2, width="stretch")

        # Written from the data rather than hardcoded to the original
        # Delhi/Auckland pair, which went stale as soon as the four extra NZ
        # cities were collected: the prose named two cities while the charts
        # above it iterated over all six.
        ranked = sorted(available.items(), key=lambda kv: kv[1]["mean_aqi"], reverse=True)
        dirtiest, dirtiest_s = ranked[0]
        cleanest, cleanest_s = ranked[-1]
        single_category = [c for c, sm in available.items() if len(sm["category_pct"]) == 1]
        multi_category = [c for c, sm in available.items() if len(sm["category_pct"]) > 1]

        lines = [
            f"**Why this matters for the model:** across the cities collected so "
            f"far, **{dirtiest}** has the highest mean AQI "
            f"(**{dirtiest_s['mean_aqi']:.1f}**) and **{cleanest}** the lowest "
            f"(**{cleanest_s['mean_aqi']:.1f}**) -- a "
            f"{dirtiest_s['mean_aqi'] / max(cleanest_s['mean_aqi'], 0.1):.0f}x gap."
        ]
        if multi_category:
            lines.append(
                f"{', '.join(multi_category)} span more than one risk category, "
                "so a classifier has something to distinguish there."
            )
        if single_category:
            lines.append(
                f"{', '.join(single_category)} stayed inside a single risk "
                "category for the entire collection period. That is not a data "
                "problem -- it is a genuinely clean-air result, and it is why no "
                "model is trained for those cities (see the Historical tab)."
            )
        st.markdown(" ".join(lines))

# ---------------------------------------------------------------------------
# TAB 5: Allergy Outlook (live pollen via Atmospore + MetService comparison)
# ---------------------------------------------------------------------------
with tab_allergy:
    st.subheader("Allergy Outlook")
    st.caption(
        "Live species-level pollen forecasts from the Atmospore API, shown "
        "alongside an honest comparison to MetService's own pollen "
        "forecasting -- including why MetService's feed isn't something "
        "this app can call directly."
    )

    # -- Live pollen forecast -------------------------------------------------
    st.markdown("#### Live pollen forecast")

    budget = get_call_budget_status()
    st.progress(
        min(1.0, budget["calls_used_today"] / budget["calls_budget"]),
        text=f"Atmospore calls used today: {budget['calls_used_today']} / "
             f"{budget['calls_budget']} (shared across all cities, resets 00:00 UTC)",
    )

    force_refresh = st.button(
        "🔄 Refresh now",
        help="Spends one call from today's shared budget to get an "
             "up-to-the-minute forecast, instead of reusing today's cache.",
    )

    with st.spinner(f"Getting pollen forecast for {selected_city}..."):
        pollen = get_pollen_forecast(city_slug, lat, lon, force_refresh=force_refresh)

    status = pollen["status"]
    if status == "live":
        st.success(f"Live data, fetched just now -- {pollen['calls_used_today']}/"
                   f"{pollen['calls_budget']} calls used today.")
    elif status == "cached":
        st.info(f"{pollen['message']} (fetched {pollen['cache_age_hours']:.1f}h ago, "
                f"{pollen['calls_used_today']}/{pollen['calls_budget']} calls used today.)")
    elif status == "stale_cache":
        st.warning(f"{pollen['message']} Last updated "
                   f"{pollen['cache_age_hours']:.1f}h ago -- may not reflect "
                   "today's conditions exactly.")
    elif status == "limit_reached":
        st.error(pollen["message"])
    elif status == "error":
        st.warning(pollen["message"])

    if pollen["parsed"] is not None:
        parsed = pollen["parsed"]
        daily_df = pd.DataFrame(parsed["daily"])
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        today_row = daily_df.iloc[0]

        risk_col, chart_col = st.columns([1, 2])
        with risk_col:
            st.markdown("**Today's overall pollen risk**")
            pollen_risk_badge(today_row["overall_risk"])
            st.caption(f"Units: {parsed['units']}")
            if parsed.get("generated_at"):
                st.caption(f"Model run: {parsed['generated_at']}")

        with chart_col:
            cat_today = pd.DataFrame({
                "Category": ["Tree", "Grass", "Weed"],
                "Level": [today_row["tree"], today_row["grass"], today_row["weed"]],
            })
            fig_today = px.bar(cat_today, x="Category", y="Level", color="Category",
                                title="Today's pollen level by category")
            st.plotly_chart(fig_today, width="stretch")

        st.markdown("#### Multi-day outlook")
        trend_df = daily_df.melt(
            id_vars=["date", "overall_risk"],
            value_vars=["tree", "grass", "weed"],
            var_name="Category", value_name="Level",
        )
        fig_trend = px.bar(trend_df, x="date", y="Level", color="Category",
                            barmode="stack", title=f"Pollen forecast -- {selected_city}")
        st.plotly_chart(fig_trend, width="stretch")

        st.markdown("#### This week's top allergens")
        top_df = pd.DataFrame(parsed["top_species"])
        if not top_df.empty:
            top_df = top_df.rename(columns={
                "display_name": "Species", "category": "Category",
                "value": f"Peak value ({parsed['units']})", "risk_level": "Risk level",
            })[["Species", "Category", f"Peak value ({parsed['units']})", "Risk level"]]
            st.dataframe(top_df, width="stretch", hide_index=True)
    else:
        st.caption("No live pollen data available right now for this city -- see "
                   "the general seasonal pattern further down instead.")

    st.divider()

    # -- Existing honest MetService comparison (unchanged, still accurate) --
    st.markdown("#### Methodology comparison")
    comparison_table = pd.DataFrame([
        {"Aspect": "What it predicts", "This project": "Air pollution risk (PM2.5/PM10/NO2 -> AQI category) + live species-level pollen (Atmospore)",
         "MetService": "Pollen/allergen risk (grass, tree, weed, fungal spore counts)"},
        {"Aspect": "Data source", "This project": "Open public APIs (Open-Meteo, OpenAQ, Atmospore)",
         "MetService": "In-house meteorologist forecasts + a real-time pollen API"},
        {"Aspect": "Public API access", "This project": "Fully open; pollen via Atmospore's free tier (100 calls/day)",
         "MetService": "Pollen API is a licensed commercial product (used by advertisers, not public)"},
        {"Aspect": "Model transparency", "This project": "Open pipeline, explainability built in (see Explainability tab)",
         "MetService": "Not disclosed publicly"},
        {"Aspect": "Geographic scope", "This project": "Any city worldwide (global weather + pollen coverage)",
         "MetService": "New Zealand only"},
        {"Aspect": "Forecast horizon", "This project": "Up to 7 days ahead (air quality) / up to 14 days (pollen)",
         "MetService": "Daily, in-season only (~34 weeks/year)"},
    ])
    st.dataframe(comparison_table, width="stretch", hide_index=True)

    st.markdown("#### Why this isn't MetService's own pollen feed")
    st.info(
        "MetService's real-time pollen feed is a licensed commercial API "
        "(confirmed via a public case study of it being used in an "
        "advertising campaign), not something freely queryable. The pollen "
        "data above instead comes from Atmospore, an independent global "
        "pollen-forecast model -- a genuinely different data source, not a "
        "scrape or estimate of MetService's own numbers."
    )

    st.markdown("#### General NZ pollen season pattern (for context)")
    st.caption("Paraphrased from MetService's published pollen season description -- "
               "background context, separate from the live Atmospore data above.")
    season_table = pd.DataFrame([
        {"Period": "Jul - Aug", "Main pollen source": "Pine (Pinus)"},
        {"Period": "Aug - Sep", "Main pollen source": "Deciduous trees (oak, elm, birch), macrocarpa, hazelnut"},
        {"Period": "Oct - Dec", "Main pollen source": "Grasses (the dominant seasonal allergen)"},
        {"Period": "Jan - Feb", "Main pollen source": "Olive, privet, and weed pollen (chenopod/amaranth)"},
        {"Period": "Feb - Mar", "Main pollen source": "Fungal spores"},
    ])
    st.dataframe(season_table, width="stretch", hide_index=True)

    st.markdown("#### What this project offers instead")
    summary = compute_city_summary(city_slug)
    if summary:
        st.write(
            f"For **{selected_city}** right now: mean AQI of **{summary['mean_aqi']:.1f}** "
            f"across {summary['rows']} hourly readings, plus the live pollen "
            "outlook above -- two genuinely complementary environmental "
            "health signals MetService's pollen forecast alone doesn't cover."
        )