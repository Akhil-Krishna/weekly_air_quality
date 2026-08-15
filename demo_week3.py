"""
Week 3 demo script -- run this live to show the merge + cleaning pipeline
working on the real Delhi data collected in week 1.

Usage:
    python demo_week3.py
"""

import sys
sys.path.insert(0, ".")
import pandas as pd
import config
from src.merge_clean import merge_and_clean

print("=== BEFORE: raw weather + raw air quality (separate, uncleaned) ===")
weather_raw = pd.read_csv(config.WEATHER_RAW_PATH)
aq_raw = pd.read_csv(config.AQ_RAW_PATH)
print(f"Weather raw:      {len(weather_raw):>6} rows  | columns: {list(weather_raw.columns)}")
print(f"Air quality raw:  {len(aq_raw):>6} rows  | columns: {list(aq_raw.columns)}")
print(f"\nMissing values in raw air quality data:\n{aq_raw.isna().sum()}")

print("\n=== RUNNING merge_clean.py ===")
merged = merge_and_clean()

print("\n=== AFTER: merged, cleaned, and labeled ===")
print(f"Final merged dataset: {len(merged)} rows")
print(f"Columns: {list(merged.columns)}")

# Check whichever pollutant columns actually came back for this city --
# not every station measures all three (e.g. some only report PM2.5).
pollutant_cols_present = [c for c in ["pm25", "pm10", "no2"] if c in merged.columns]
check_cols = ["temperature_2m"] + pollutant_cols_present + ["aqi"]
print(f"\nMissing values after cleaning:\n{merged[check_cols].isna().sum()}")
if len(pollutant_cols_present) < 3:
    missing = [c for c in ["pm25", "pm10", "no2"] if c not in pollutant_cols_present]
    print(f"\nNote: this city's station(s) only report {pollutant_cols_present} "
          f"-- no {missing} sensor was found nearby. AQI is still computed "
          f"correctly (max of whichever sub-indices are available).")
print(f"\nSample rows:")
print(merged[["datetime", "temperature_2m", "pm25", "aqi", "risk_category"]].head(5).to_string(index=False))

print(f"\nDone -- merged_clean.csv saved to data/processed/{config.CITY_SLUG}/")
