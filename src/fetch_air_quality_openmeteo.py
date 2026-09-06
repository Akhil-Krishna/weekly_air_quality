

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import get_json_with_retries, chunk_date_range

# Open-Meteo Air Quality API field name -> our internal column name
FIELD_RENAME = {
    "pm2_5": "pm25",
    "pm10": "pm10",
    "nitrogen_dioxide": "no2",
    "us_aqi": "us_aqi_openmeteo",
}


def fetch_air_quality_openmeteo_history(lat, lon, start_date, end_date, hourly_vars):
    """Fetch hourly historical air quality from Open-Meteo, chunked like fetch_weather.py."""
    all_frames = []
    for chunk_start, chunk_end in chunk_date_range(start_date, end_date, chunk_days=90):
        print(f"Fetching Open-Meteo air quality {chunk_start} -> {chunk_end} ...")
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
            "hourly": ",".join(hourly_vars),
            "timezone": "auto",
        }
        data = get_json_with_retries(config.OPEN_METEO_AQ_URL, params=params)
        hourly = data.get("hourly", {})
        if not hourly or "time" not in hourly:
            print("  WARNING: no hourly air quality data returned for this window.")
            continue
        df_chunk = pd.DataFrame(hourly)
        all_frames.append(df_chunk)

    if not all_frames:
        raise RuntimeError(
            "No air quality data was retrieved from Open-Meteo for the requested range. "
            "Note: Open-Meteo's air quality historical archive has a shorter usable "
            "window than the weather archive -- if this keeps failing for very old "
            "dates, shorten HISTORY_MONTHS in config.py."
        )

    df = pd.concat(all_frames, ignore_index=True)
    df = df.rename(columns={"time": "datetime", **FIELD_RENAME})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    return df


def fetch_air_quality_openmeteo_forecast(lat, lon, hourly_vars, forecast_days=5):
    """Fetch upcoming forecast air quality (mirrors fetch_weather.fetch_weather_forecast)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly_vars),
        "forecast_days": forecast_days,
        "timezone": "auto",
    }
    data = get_json_with_retries(config.OPEN_METEO_AQ_URL, params=params)
    hourly = data.get("hourly", {})
    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "datetime", **FIELD_RENAME})
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def main():
    print(f"Collecting Open-Meteo air quality for {config.CITY_NAME} "
          f"({config.LATITUDE}, {config.LONGITUDE})")
    print(f"Date range: {config.START_DATE} to {config.END_DATE}")
    print("(Model/satellite-based CAMS reanalysis -- not a physical ground sensor.)")

    df = fetch_air_quality_openmeteo_history(
        config.LATITUDE, config.LONGITUDE,
        config.START_DATE, config.END_DATE,
        config.OPEN_METEO_AQ_HOURLY_VARS,
    )
    df.to_csv(config.AQ_OPENMETEO_RAW_PATH, index=False)
    print(f"Saved {len(df)} rows to {config.AQ_OPENMETEO_RAW_PATH}")
    return df


if __name__ == "__main__":
    main()
