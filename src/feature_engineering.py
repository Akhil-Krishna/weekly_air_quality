"""
Feature engineering: time-based features, rolling averages, and lag features.

Three things in here are load-bearing and were each wrong at some point:

1. Rolling/lag windows are measured in HOURS, not rows. The timeline has gaps
   (station outages, API limits), so rolling(window=24) over raw rows averaged
   whatever 24 records happened to sit next to each other -- in the worst case
   spanning 46 days, under a column named pm25_roll_24h_mean. Everything here
   is computed on a gap-filled hourly grid, so one row really is one hour.

2. Pollutant-derived rolling features are shifted back an hour so they do NOT
   include the current row. They used to, and risk_category is computed
   directly from the current hour's pm25/pm10/no2 -- so those 8 features
   carried the answer, diluted 1/24 and 1/168. train_models.py drops the raw
   pollutants from the feature set, but that does nothing about averages of
   them that include the present moment.

3. Time-of-day and season come from LOCAL time, and seasons flip for southern
   hemisphere cities. The join key is UTC (see fetch_weather.py), so reading
   the hour straight off it placed Delhi midnight at 05:30 local time; and
   hardcoding Dec-Feb as winter inverted every season label for the NZ cities.

Usage:
    python -m src.feature_engineering
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import to_local

# Pollutant columns are treated differently from weather throughout: the risk
# label is derived from them, so their history can only be used with a lag.
# Weather is an exogenous input and may be used at the current hour.
POLLUTANT_COLS = ["pm25", "pm10", "no2"]

# Hours of lag applied to pollutant-derived rolling windows so the current
# (target-bearing) observation is excluded.
POLLUTANT_ROLL_SHIFT_HOURS = 1

WEATHER_FEATURE_COLS = ["temperature_2m", "relative_humidity_2m",
                        "surface_pressure", "wind_speed_10m",
                        "wind_direction_10m", "precipitation"]

SEASON_LEVELS = ("winter", "spring", "summer", "autumn")
TIME_OF_DAY_LEVELS = ("morning", "afternoon", "evening", "night")

_NORTHERN_SEASONS = {
    (12, 1, 2): "winter",
    (3, 4, 5): "spring",
    (6, 7, 8): "summer",
    (9, 10, 11): "autumn",
}
_FLIP = {"winter": "summer", "summer": "winter",
         "spring": "autumn", "autumn": "spring"}


def month_to_season(month, southern_hemisphere=False):
    """Meteorological season for a month, correct for both hemispheres."""
    for months, name in _NORTHERN_SEASONS.items():
        if month in months:
            return _FLIP[name] if southern_hemisphere else name
    raise ValueError(f"Not a month: {month!r}")


def hour_to_bucket(h):
    if 5 <= h < 12:
        return "morning"
    if 12 <= h < 17:
        return "afternoon"
    if 17 <= h < 21:
        return "evening"
    return "night"


def add_time_features(df, tz="UTC", southern_hemisphere=False):
    """Add calendar/clock features, derived from LOCAL time.

    The datetime column is UTC (the join key). A datetime_local column is added
    alongside it for display and for these features; it is deliberately not a
    model feature itself.
    """
    df = df.copy()
    df["datetime_local"] = to_local(df["datetime"], tz)
    local = df["datetime_local"]

    df["hour"] = local.dt.hour
    df["day_of_week"] = local.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = local.dt.month

    df["season"] = df["month"].apply(
        lambda m: month_to_season(m, southern_hemisphere=southern_hemisphere)
    )
    df["time_of_day"] = df["hour"].apply(hour_to_bucket)

    # Cyclical encodings so the model understands hour 23 is close to hour 0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Pin the dummy columns to the full set of levels. get_dummies on its own
    # only emits the levels present in THIS frame, so a 3-day forecast window
    # produced a different column set than training did.
    for season in SEASON_LEVELS:
        df[f"season_{season}"] = (df["season"] == season).astype(int)
    for bucket in TIME_OF_DAY_LEVELS:
        df[f"time_of_day_{bucket}"] = (df["time_of_day"] == bucket).astype(int)

    return df


def _hourly_grid(df):
    """Reindex onto a complete, gap-free hourly index.

    Missing hours become all-NaN rows. That is the point: once every row is
    exactly one hour after the last, a window of N rows IS a window of N hours,
    and min_periods can do its job of rejecting windows that are mostly hole.
    """
    g = (df.drop_duplicates(subset="datetime")
           .sort_values("datetime")
           .set_index("datetime"))
    full = pd.date_range(g.index.min(), g.index.max(), freq="h")
    full.name = "datetime"
    return g.reindex(full)


def _attach(df, grid_features):
    """Merge features computed on the hourly grid back onto the real rows."""
    return df.merge(grid_features.reset_index(), on="datetime", how="left")


def add_rolling_features(df, cols, windows_hours=(24, 24 * 7), shift_hours=0):
    """Trailing rolling mean/std over true HOUR windows.

    shift_hours=0  -> window ends at the current hour (fine for weather)
    shift_hours=1  -> window ends one hour ago, excluding the current
                      observation (required for anything the label is derived
                      from; see the module docstring)
    """
    df = df.copy().sort_values("datetime").reset_index(drop=True)
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df

    grid = _hourly_grid(df[["datetime"] + cols])
    feats = pd.DataFrame(index=grid.index)

    for col in cols:
        series = grid[col]
        if shift_hours:
            series = series.shift(shift_hours)
        for w in windows_hours:
            label = "24h" if w == 24 else f"{w // 24}d"
            roll = series.rolling(window=w, min_periods=max(3, w // 4))
            feats[f"{col}_roll_{label}_mean"] = roll.mean()
            feats[f"{col}_roll_{label}_std"] = roll.std()

    return _attach(df, feats)


def add_lag_features(df, cols, lags_hours=(24,)):
    """Value from exactly N hours earlier, or NaN if that hour has no data."""
    df = df.copy().sort_values("datetime").reset_index(drop=True)
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df

    grid = _hourly_grid(df[["datetime"] + cols])
    feats = pd.DataFrame(index=grid.index)
    for col in cols:
        for lag in lags_hours:
            feats[f"{col}_lag_{lag}h"] = grid[col].shift(lag)

    return _attach(df, feats)


def build_features(city_name=None):
    city_name = city_name or config.CITY_NAME
    tz = config.city_timezone(city_name)
    lat = (config.CITIES.get(city_name) or {}).get("lat", config.LATITUDE)
    southern = lat < 0

    df = pd.read_csv(config.MERGED_PATH, parse_dates=["datetime"])
    n_in = len(df)

    weather_cols = [c for c in WEATHER_FEATURE_COLS if c in df.columns]
    pollutant_cols = [c for c in POLLUTANT_COLS if c in df.columns]

    df = add_time_features(df, tz=tz, southern_hemisphere=southern)
    # Weather may use the current hour; pollutants may not.
    df = add_rolling_features(df, weather_cols, windows_hours=(24, 24 * 7), shift_hours=0)
    df = add_rolling_features(df, pollutant_cols, windows_hours=(24, 24 * 7),
                              shift_hours=POLLUTANT_ROLL_SHIFT_HOURS)
    df = add_lag_features(df, pollutant_cols + weather_cols, lags_hours=(24,))

    # Rolling/lag features are NaN at the start of the series and after any gap
    # longer than the window -- those rows genuinely have no history to stand
    # on, so drop them rather than impute a number nobody measured.
    feature_cols_needed = [c for c in df.columns if "_roll_" in c or "_lag_" in c]
    df = df.dropna(subset=feature_cols_needed).reset_index(drop=True)

    df.to_csv(config.FEATURES_PATH, index=False)
    print(f"Built {df.shape[1]} columns x {df.shape[0]} rows -> {config.FEATURES_PATH}")
    print(f"  ({n_in - len(df)} of {n_in} merged rows dropped for lacking a "
          f"complete 24h/7d history -- gaps in the timeline, not a bug)")
    print(f"  clock features from {tz}  |  southern-hemisphere seasons: {southern}")
    return df


if __name__ == "__main__":
    build_features()
