"""
Fetch historical hourly weather data from the Open-Meteo Archive API.

Open-Meteo's archive endpoint is free and requires no API key.
Docs: https://open-meteo.com/en/docs/historical-weather-api

All timestamps are requested and returned in UTC (config.TIMEZONE_FETCH), the
same clock OpenAQ reports in, so merge_clean.py can join the two sources on
`datetime` safely. Local-clock features are derived later in
feature_engineering.py from config.CITIES[city]["tz"].

Usage:
    python -m src.fetch_weather
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import get_json_with_retries, chunk_date_range


def fetch_weather_history(lat, lon, start_date, end_date, hourly_vars):
    """Fetch hourly weather history, chunked into <=90 day windows to be polite to the API."""
    all_frames = []
    for chunk_start, chunk_end in chunk_date_range(start_date, end_date, chunk_days=90):
        print(f"Fetching weather {chunk_start} -> {chunk_end} ...")
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
            "hourly": ",".join(hourly_vars),
            "timezone": config.TIMEZONE_FETCH,
        }
        data = get_json_with_retries(config.OPEN_METEO_ARCHIVE_URL, params=params)
        hourly = data.get("hourly", {})
        if not hourly or "time" not in hourly:
            print("  WARNING: no hourly data returned for this window.")
            continue
        df_chunk = pd.DataFrame(hourly)
        all_frames.append(df_chunk)

    if not all_frames:
        raise RuntimeError("No weather data was retrieved for the requested date range.")

    df = pd.concat(all_frames, ignore_index=True)
    df = df.rename(columns={"time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    return df


def fetch_weather_forecast(lat, lon, hourly_vars, forecast_days=5, past_days=8):
    """Fetch upcoming forecast weather (for the app's Forecast view), preceded
    by `past_days` of recent real observations.

    The past window matters: the model's 24h and 7d rolling weather features
    need the hours immediately BEFORE the forecast horizon to be meaningful.
    Splicing in the tail of merged_clean.csv instead doesn't work -- that file
    is only as fresh as the last pipeline run, so the "24h mean" at the start
    of the horizon ended up averaging across a multi-week hole in the timeline.
    Open-Meteo's forecast endpoint serves recent past hours from the same
    endpoint on the same grid, which gives one genuinely contiguous series.

    Returns UTC timestamps, matching fetch_weather_history.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly_vars),
        "forecast_days": forecast_days,
        "past_days": past_days,
        "timezone": config.TIMEZONE_FETCH,
    }
    data = get_json_with_retries(config.OPEN_METEO_FORECAST_URL, params=params)
    hourly = data.get("hourly", {})
    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    return df


def main():
    print(f"Collecting historical weather for {config.CITY_NAME} "
          f"({config.LATITUDE}, {config.LONGITUDE})")
    print(f"Date range: {config.START_DATE} to {config.END_DATE}")

    df = fetch_weather_history(
        config.LATITUDE, config.LONGITUDE,
        config.START_DATE, config.END_DATE,
        config.WEATHER_HOURLY_VARS,
    )
    df.to_csv(config.WEATHER_RAW_PATH, index=False)
    print(f"Saved {len(df)} rows to {config.WEATHER_RAW_PATH}")


if __name__ == "__main__":
    main()
