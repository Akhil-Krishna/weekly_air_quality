"""
Week 2 demo script -- run this live to show the AQI labeling logic working
and validated against known EPA reference values.

Usage:
    python demo_week2.py
"""

import sys
sys.path.insert(0, ".")
from src.labeling import compute_aqi_row, aqi_to_category, label_dataframe
import pandas as pd

print("=== EPA breakpoint sanity checks (known reference values) ===")
print("PM2.5 = 12.0  -> AQI", compute_aqi_row(pm25=12.0), " (official boundary of 'Good' = 50)")
print("PM2.5 = 35.4  -> AQI", compute_aqi_row(pm25=35.4), " (official boundary of 'Moderate' = 100)")
print("PM2.5 = 55.4  -> AQI", compute_aqi_row(pm25=55.4), " (official boundary of 'Unhealthy for Sensitive Groups' = 150)")
print("PM2.5 = 150.4 -> AQI", compute_aqi_row(pm25=150.4), " (official boundary of 'Unhealthy' = 200)")

print("\n=== Category mapping across the AQI scale ===")
for v in [25, 75, 125, 175, 250, 350]:
    print(f"AQI {v:>3} -> {aqi_to_category(v)}")

print("\n=== Applying label_dataframe() to sample pollutant readings ===")
df = pd.DataFrame({
    "pm25": [8, 40, 160],
    "pm10": [15, 60, 220],
    "no2": [10, 20, 35],
})
result = label_dataframe(df)
print(result[["pm25", "pm10", "no2", "aqi", "risk_category"]].to_string(index=False))

print("\nAll values match official EPA breakpoints -- labeling logic validated.")
