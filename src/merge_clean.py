"""
Merge Open-Meteo weather data with OpenAQ air quality data on datetime,
clean missing values and outliers, and produce one tidy dataframe.

Both sources are in UTC (see fetch_weather.py), which is what makes the join
below valid. It used to join Open-Meteo local time against OpenAQ UTC, which
silently offset every label from its own features -- by 5.5h for Delhi and by
12-13h for the NZ cities, the latter varying with DST.

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

WEATHER_NUMERIC_COLS = ["temperature_2m", "relative_humidity_2m", "surface_pressure",
                        "wind_speed_10m", "wind_direction_10m", "precipitation"]

# Longest run of missing hours to fill by interpolation. Beyond this the gap is
# left as a hole, and feature_engineering.py drops the affected rows rather
# than inventing a day of weather.
MAX_INTERPOLATE_HOURS = 6


def load_raw():
    weather = pd.read_csv(config.WEATHER_RAW_PATH, parse_dates=["datetime"])
    aq = pd.read_csv(config.AQ_RAW_PATH, parse_dates=["datetime"])
    return weather, aq


def clean_weather(df):
    df = df.copy()
    df = df.drop_duplicates(subset="datetime").sort_values("datetime")
    for col in WEATHER_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.set_index("datetime").resample("h").mean()
    # Interpolate only short gaps, and only numeric columns.
    df = df.interpolate(method="linear", limit=MAX_INTERPOLATE_HOURS)
    return df.reset_index()


def clean_pollutants(df):
    """Clean pollutant readings without destroying the extreme tail.

    Outlier handling here used to be IQR winsorizing at q3 + 3*IQR, computed
    over the WHOLE dataframe. That had two problems:

      * it fit a cleaning parameter on train and test together, so test-period
        statistics leaked into the training data, and
      * for a heavy-tailed quantity like PM2.5 in Delhi it clipped away exactly
        the extreme readings the Very Unhealthy and Hazardous classes are made
        of -- removing the rare classes the model is being asked to predict.

    A pollutant concentration is heavy-tailed by nature; being far above the
    75th percentile is a real air quality event, not a data error. So only
    physically impossible values are rejected now: negatives (a known sensor
    failure mode) and readings above config.POLLUTANT_SANITY_CEILINGS, which
    sit well above any concentration ever recorded at ground level.
    """
    df = df.copy()
    df = df.drop_duplicates(subset="datetime").sort_values("datetime")
    pollutant_cols = [c for c in ["pm25", "pm10", "no2"] if c in df.columns]

    # 'source' (openaq_ground / openmeteo_model) is metadata, not a numeric
    # reading -- resample().mean() cannot average a string column, so carry it
    # through separately and reattach after resampling.
    source_series = None
    if "source" in df.columns:
        source_series = df.set_index("datetime")["source"].resample("h").first()

    for col in pollutant_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        n_before = df[col].notna().sum()

        df.loc[df[col] < 0, col] = np.nan
        ceiling = config.POLLUTANT_SANITY_CEILINGS.get(col)
        if ceiling is not None:
            n_over = int((df[col] > ceiling).sum())
            if n_over:
                print(f"  {col}: nulled {n_over} reading(s) above the "
                      f"{ceiling:.0f} ug/m3 sanity ceiling (sensor fault range)")
            df.loc[df[col] > ceiling, col] = np.nan

        n_dropped = n_before - df[col].notna().sum()
        if n_dropped:
            print(f"  {col}: {n_dropped} implausible reading(s) removed of {n_before}")

    df = (df[["datetime"] + pollutant_cols]
          .set_index("datetime").resample("h").mean().reset_index())
    df[pollutant_cols] = df[pollutant_cols].interpolate(
        method="linear", limit=MAX_INTERPOLATE_HOURS
    )

    if source_series is not None:
        df = df.merge(source_series.rename("source").reset_index(), on="datetime", how="left")
        df["source"] = df["source"].ffill().bfill()

    return df


def report_coverage(merged):
    """Print how much of the nominal hourly timeline actually survived.

    Worth seeing on every run: a low number here is the signal that something
    upstream is dropping data (it was 59% for Delhi while the OpenAQ paginator
    was stopping after one page), and the size of the gaps determines how many
    rows feature_engineering.py then has to drop for want of history.
    """
    if merged.empty:
        print("Coverage: no rows at all.")
        return
    lo, hi = merged["datetime"].min(), merged["datetime"].max()
    expected = len(pd.date_range(lo, hi, freq="h"))
    pct = 100.0 * len(merged) / expected
    print(f"\nTimeline coverage: {len(merged)} of {expected} hours "
          f"({pct:.1f}%) between {lo} and {hi} [UTC]")

    gaps = merged["datetime"].diff()
    big = merged.loc[gaps > pd.Timedelta(hours=1)]
    if len(big):
        worst = gaps.max()
        print(f"  {len(big)} gap(s) longer than 1h; largest is "
              f"{worst.total_seconds() / 3600:.0f}h")
        if pct < 85:
            print("  NOTE: coverage below 85% -- check the collection step "
                  "before reading much into the model metrics.")


def merge_and_clean():
    weather_raw, aq_raw = load_raw()
    weather = clean_weather(weather_raw)
    aq = clean_pollutants(aq_raw)

    merged = pd.merge(weather, aq, on="datetime", how="inner")
    merged = merged.dropna(subset=["pm25"])  # need at least PM2.5 to label risk

    merged = label_dataframe(merged)
    merged = merged.dropna(subset=["risk_category"]).reset_index(drop=True)

    print(f"Weather rows: {len(weather)} | AQ rows: {len(aq)} | "
          f"Merged (labeled) rows: {len(merged)}")
    print("\nRisk category distribution:")
    print(merged["risk_category"].value_counts())
    report_coverage(merged)

    merged.to_csv(config.MERGED_PATH, index=False)
    print(f"\nSaved merged & cleaned dataset to {config.MERGED_PATH}")
    return merged


if __name__ == "__main__":
    merge_and_clean()
