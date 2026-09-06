"""
Cross-check our own EPA-breakpoint AQI calculation (labeling.py) against
Open-Meteo's independently-computed `us_aqi` field, for the same city and a
recent sample window.

This does NOT replace labeling.py -- it validates it. If our computed AQI
and Open-Meteo's us_aqi track each other closely, that's good independent
evidence our breakpoint math is implemented correctly. Large, systematic
disagreement would be a signal to re-check labeling.py.

Usage:
    python -m src.validate_aqi_crosscheck
"""

import sys
import os
import json
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.labeling import compute_aqi_row
from src.fetch_air_quality_openmeteo import fetch_air_quality_openmeteo_history


def run_crosscheck(sample_days=14):
    """Pull a short recent Open-Meteo sample (with both raw pollutants and
    us_aqi), compute our own AQI from the raw pollutants, and compare the two
    against Open-Meteo's own us_aqi for the same hours.
    """
    end = config.END_DATE
    start = end - timedelta(days=sample_days)

    print(f"\nRunning AQI cross-check for {config.CITY_NAME} "
          f"({start} to {end}, {sample_days} days)...")

    df = fetch_air_quality_openmeteo_history(
        config.LATITUDE, config.LONGITUDE, start, end,
        config.OPEN_METEO_AQ_HOURLY_VARS,
    )

    if df.empty or "us_aqi_openmeteo" not in df.columns:
        print("Cross-check skipped: no us_aqi data returned for the sample window.")
        return None

    # Compute our own AQI from the same raw pollutant readings Open-Meteo gave us
    df["our_aqi"] = df.apply(
        lambda row: compute_aqi_row(
            pm25=row.get("pm25"), pm10=row.get("pm10"), no2_ugm3=row.get("no2"),
        ),
        axis=1,
    )

    valid = df.dropna(subset=["our_aqi", "us_aqi_openmeteo"])
    if valid.empty:
        print("Cross-check skipped: no overlapping valid rows to compare.")
        return None

    diff = valid["our_aqi"] - valid["us_aqi_openmeteo"]
    mae = diff.abs().mean()
    corr = valid["our_aqi"].corr(valid["us_aqi_openmeteo"])
    mean_ours = valid["our_aqi"].mean()
    mean_theirs = valid["us_aqi_openmeteo"].mean()

    report = {
        "city": config.CITY_NAME,
        "sample_window": f"{start} to {end}",
        "n_hours_compared": int(len(valid)),
        "mean_absolute_error": float(round(mae, 2)),
        "correlation": float(round(corr, 3)) if not np.isnan(corr) else None,
        "mean_our_aqi": float(round(mean_ours, 1)),
        "mean_openmeteo_us_aqi": float(round(mean_theirs, 1)),
    }

    print(f"  Hours compared: {report['n_hours_compared']}")
    print(f"  Mean absolute difference: {report['mean_absolute_error']} AQI points")
    print(f"  Correlation: {report['correlation']}")
    print(f"  Our mean AQI: {report['mean_our_aqi']}  |  Open-Meteo mean us_aqi: "
          f"{report['mean_openmeteo_us_aqi']}")

    if report["mean_absolute_error"] is not None and report["mean_absolute_error"] > 25:
        print("  NOTE: difference is fairly large -- worth double-checking labeling.py "
              "breakpoints, or note in the report that ground-sensor vs. model-based "
              "readings can genuinely diverge (different pollutant sourcing, spatial "
              "averaging at CAMS's grid resolution, etc).")
    else:
        print("  Our AQI calculation tracks Open-Meteo's independent us_aqi reasonably "
              "well -- good evidence labeling.py is implemented correctly.")

    with open(config.AQ_CROSSCHECK_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved cross-check report to {config.AQ_CROSSCHECK_PATH}")

    return report


if __name__ == "__main__":
    run_crosscheck()
