"""
Feature engineering: time-based features, rolling averages, and lag features.

Usage:
    python -m src.feature_engineering
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def add_time_features(df):
    df = df.copy()
    dt = df["datetime"]
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = dt.dt.month

    def month_to_season(m):
        if m in (12, 1, 2):
            return "winter"
        if m in (3, 4, 5):
            return "spring"
        if m in (6, 7, 8):
            return "summer"
        return "autumn"

    df["season"] = df["month"].apply(month_to_season)

    def hour_to_bucket(h):
        if 5 <= h < 12:
            return "morning"
        if 12 <= h < 17:
            return "afternoon"
        if 17 <= h < 21:
            return "evening"
        return "night"

    df["time_of_day"] = df["hour"].apply(hour_to_bucket)

    # Cyclical encodings so the model understands hour 23 is close to hour 0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df = pd.get_dummies(df, columns=["season", "time_of_day"], drop_first=False)
    return df


def add_rolling_features(df, cols, windows_hours=(24, 24 * 7)):
    df = df.copy().sort_values("datetime")
    for col in cols:
        if col not in df.columns:
            continue
        for w in windows_hours:
            label = "24h" if w == 24 else f"{w // 24}d"
            df[f"{col}_roll_{label}_mean"] = (
                df[col].rolling(window=w, min_periods=max(3, w // 4)).mean()
            )
            df[f"{col}_roll_{label}_std"] = (
                df[col].rolling(window=w, min_periods=max(3, w // 4)).std()
            )
    return df


def add_lag_features(df, cols, lags_hours=(24,)):
    df = df.copy().sort_values("datetime")
    for col in cols:
        if col not in df.columns:
            continue
        for lag in lags_hours:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


def build_features():
    df = pd.read_csv(config.MERGED_PATH, parse_dates=["datetime"])

    weather_cols = ["temperature_2m", "relative_humidity_2m", "surface_pressure",
                     "wind_speed_10m", "precipitation"]
    pollutant_cols = [c for c in ["pm25", "pm10", "no2"] if c in df.columns]

    df = add_time_features(df)
    df = add_rolling_features(df, weather_cols + pollutant_cols, windows_hours=(24, 24 * 7))
    df = add_lag_features(df, pollutant_cols + weather_cols, lags_hours=(24,))

    # Rolling/lag features create NaNs at the start of the series -- drop those rows
    feature_cols_needed = [c for c in df.columns if "_roll_" in c or "_lag_" in c]
    df = df.dropna(subset=feature_cols_needed).reset_index(drop=True)

    df.to_csv(config.FEATURES_PATH, index=False)
    print(f"Built {df.shape[1]} columns x {df.shape[0]} rows -> {config.FEATURES_PATH}")
    return df


if __name__ == "__main__":
    build_features()
