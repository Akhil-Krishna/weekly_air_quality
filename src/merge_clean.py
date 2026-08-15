"""
Merge Open-Meteo weather data with OpenAQ air quality data on datetime,
clean missing values and outliers, and produce one tidy dataframe.

Usage:
    python -m src.merge_clean
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.labeling import label_dataframe


def load_raw():
    weather = pd.read_csv(config.WEATHER_RAW_PATH, parse_dates=["datetime"])
    aq = pd.read_csv(config.AQ_RAW_PATH, parse_dates=["datetime"])
    return weather, aq


def clean_weather(df):
    df = df.copy()
    df = df.drop_duplicates(subset="datetime").sort_values("datetime")
    # Interpolate small gaps, cap at a reasonable physical range for sanity
    for col in ["temperature_2m", "relative_humidity_2m", "surface_pressure",
                "wind_speed_10m", "wind_direction_10m", "precipitation"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.set_index("datetime").resample("h").mean().reset_index()
    df = df.interpolate(method="linear", limit=6)
    return df


def clean_pollutants(df):
    df = df.copy()
    df = df.drop_duplicates(subset="datetime").sort_values("datetime")
    pollutant_cols = [c for c in ["pm25", "pm10", "no2"] if c in df.columns]

    # 'source' (openaq_ground / openmeteo_model) is metadata, not a numeric
    # reading -- resample().mean() can't average a string column, so carry it
    # through separately and reattach after resampling.
    source_series = None
    if "source" in df.columns:
        source_series = df.set_index("datetime")["source"].resample("h").first()

    for col in pollutant_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Negative readings are sensor errors -> null them out
        df.loc[df[col] < 0, col] = np.nan
        # Outlier clipping via IQR (winsorize rather than drop, to keep timeline continuous)
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 3 * iqr
        df[col] = df[col].clip(upper=upper)

    df = df[["datetime"] + pollutant_cols].set_index("datetime").resample("h").mean().reset_index()
    df[pollutant_cols] = df[pollutant_cols].interpolate(method="linear", limit=6)

    if source_series is not None:
        df = df.merge(source_series.rename("source").reset_index(), on="datetime", how="left")
        df["source"] = df["source"].ffill().bfill()

    return df


def merge_and_clean():
    weather_raw, aq_raw = load_raw()
    weather = clean_weather(weather_raw)
    aq = clean_pollutants(aq_raw)

    merged = pd.merge(weather, aq, on="datetime", how="inner")
    merged = merged.dropna(subset=["pm25"], how="all")  # need at least PM2.5 to label risk

    merged = label_dataframe(merged)
    merged = merged.dropna(subset=["risk_category"]).reset_index(drop=True)

    print(f"Weather rows: {len(weather)} | AQ rows: {len(aq)} | Merged (labeled) rows: {len(merged)}")
    print("\nRisk category distribution:")
    print(merged["risk_category"].value_counts())

    merged.to_csv(config.MERGED_PATH, index=False)
    print(f"\nSaved merged & cleaned dataset to {config.MERGED_PATH}")
    return merged


if __name__ == "__main__":
    merge_and_clean()
