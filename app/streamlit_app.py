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
  - Allergy Comparison: how this project's approach compares to MetService's
    pollen forecasting, and why a live data comparison isn't possible.
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
from src.explainability import global_feature_importance, explain_single_prediction

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
    paths = get_city_paths(city_slug)
    return pd.read_csv(paths["features"], parse_dates=["datetime"])


@st.cache_data
def load_merged_data(city_slug):
    paths = get_city_paths(city_slug)
    if not os.path.exists(paths["merged"]):
        return None
    return pd.read_csv(paths["merged"], parse_dates=["datetime"])


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

st.caption(f"City: **{selected_city}**  |  Coordinates: {lat}, {lon}")

paths = get_city_paths(city_slug)
has_model = os.path.exists(paths["best_model"]) and os.path.exists(paths["features"])

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
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No data at all for this city yet. Run the pipeline first.")
    else:
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
                    st.plotly_chart(fig_shap, use_container_width=True)
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
            fig = px.line(trend_df, x="datetime", y="aqi", title="AQI trend")
            fig.add_hline(y=50, line_dash="dot", line_color="green", annotation_text="Good")
            fig.add_hline(y=100, line_dash="dot", line_color="gold", annotation_text="Moderate")
            fig.add_hline(y=150, line_dash="dot", line_color="orange", annotation_text="USG")
            fig.add_hline(y=200, line_dash="dot", line_color="red", annotation_text="Unhealthy")
            st.plotly_chart(fig, use_container_width=True)
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
                        lat, lon, config.WEATHER_HOURLY_VARS, forecast_days=forecast_days,
                    )
                except Exception as e:
                    st.error(f"Could not fetch forecast data: {e}")
                    st.stop()

            merged_hist = load_merged_data(city_slug)
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
    if not has_model:
        st.info(f"No trained model for **{selected_city}** -- nothing to explain yet.")
    else:
        st.subheader("What drives the model's predictions?")
        imp_df = global_feature_importance(model, feature_cols, top_n=15)
        fig = px.bar(
            imp_df.sort_values("importance"), x="importance", y="feature",
            orientation="h", title=f"Top 15 features -- {metrics['best_model']} (global importance)",
        )
        st.plotly_chart(fig, use_container_width=True)

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
    st.subheader("Delhi vs. Auckland: a real-world contrast")
    st.caption(
        "Real computed statistics from each city's own collected data -- not "
        "illustrative numbers. This is the core motivation for a two-city design: "
        "one high-variance, high-pollution city and one clean-air baseline."
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
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Risk category distribution, side by side")
        cat_rows = []
        for city, s in available.items():
            for cat, pct in s["category_pct"].items():
                cat_rows.append({"City": city, "Risk category": cat, "% of hours": pct})
        cat_df = pd.DataFrame(cat_rows)
        fig2 = px.bar(cat_df, x="City", y="% of hours", color="Risk category",
                      color_discrete_map=RISK_COLORS, title="Time spent in each risk category")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            "**Why this matters for the model:** Delhi has real variance across "
            "risk categories, so a classifier has something to learn. Auckland's "
            "air stayed almost entirely in one category for the whole period -- "
            "not a data problem, a genuinely clean-air result. That's why Auckland "
            "has no trained model yet (see the Historical tab), and it's direct "
            "evidence that comparing a high-variance city against a clean-air "
            "baseline is a meaningful test of whether this approach generalizes."
        )

# ---------------------------------------------------------------------------
# TAB 5: Allergy Comparison (vs MetService)
# ---------------------------------------------------------------------------
with tab_allergy:
    st.subheader("How this project compares to MetService's allergy forecasting")
    st.caption(
        "MetService (New Zealand's national weather service) publishes a pollen/"
        "allergy forecast. This tab compares that system to this project's "
        "approach honestly -- including where a live data comparison genuinely "
        "isn't possible, and why."
    )

    st.markdown("#### Methodology comparison")
    comparison_table = pd.DataFrame([
        {"Aspect": "What it predicts", "This project": "Air pollution risk (PM2.5/PM10/NO2 -> AQI category)",
         "MetService": "Pollen/allergen risk (grass, tree, weed, fungal spore counts)"},
        {"Aspect": "Data source", "This project": "Open public APIs (Open-Meteo, OpenAQ)",
         "MetService": "In-house meteorologist forecasts + a real-time pollen API"},
        {"Aspect": "Public API access", "This project": "Fully open, no cost, no key needed for weather",
         "MetService": "Pollen API is a licensed commercial product (used by advertisers, not public)"},
        {"Aspect": "Model transparency", "This project": "Open pipeline, explainability built in (see Explainability tab)",
         "MetService": "Not disclosed publicly"},
        {"Aspect": "Geographic scope", "This project": "Any city with weather data (global)",
         "MetService": "New Zealand only"},
        {"Aspect": "Forecast horizon", "This project": "Up to 7 days ahead (Forecast tab)",
         "MetService": "Daily, in-season only (~34 weeks/year)"},
    ])
    st.dataframe(comparison_table, use_container_width=True, hide_index=True)

    st.markdown("#### Why there's no live pollen data in this app")
    st.info(
        "Two things were checked directly before deciding this: Open-Meteo's Air "
        "Quality API does include pollen data, but it only covers Europe -- not "
        "New Zealand or India. And MetService's real-time pollen feed is a "
        "licensed commercial API (confirmed via a public case study of it being "
        "used in an advertising campaign), not something freely available to "
        "query. Rather than build a fragile, untested scraper against a page "
        "that loads its data client-side, this comparison stays at the "
        "methodology level -- which is also the more honest comparison, since "
        "MetService's exact model isn't public either."
    )

    st.markdown("#### General NZ pollen season pattern (for context, not live data)")
    st.caption("Paraphrased from MetService's published pollen season description -- "
               "a general calendar, not a real-time feed.")
    season_table = pd.DataFrame([
        {"Period": "Jul - Aug", "Main pollen source": "Pine (Pinus)"},
        {"Period": "Aug - Sep", "Main pollen source": "Deciduous trees (oak, elm, birch), macrocarpa, hazelnut"},
        {"Period": "Oct - Dec", "Main pollen source": "Grasses (the dominant seasonal allergen)"},
        {"Period": "Jan - Feb", "Main pollen source": "Olive, privet, and weed pollen (chenopod/amaranth)"},
        {"Period": "Feb - Mar", "Main pollen source": "Fungal spores"},
    ])
    st.dataframe(season_table, use_container_width=True, hide_index=True)

    st.markdown("#### What this project offers instead")
    summary = compute_city_summary(city_slug)
    if summary:
        st.write(
            f"For **{selected_city}** right now: mean AQI of **{summary['mean_aqi']:.1f}** "
            f"across {summary['rows']} hourly readings -- a pollution-based risk signal "
            "MetService's pollen forecast doesn't cover at all, and a genuinely "
            "complementary (not competing) piece of environmental health information."
        )
