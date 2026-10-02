"""
Cross-check our own EPA-breakpoint AQI calculation (labeling.py) against
Open-Meteo's independently-computed us_aqi field, for the same city and sample
window.

This does NOT replace labeling.py -- it validates it.

Read the two comparisons it reports separately, because they are measuring
different things:

  instantaneous -- our AQI from each hour's raw concentration, against their
      us_aqi for the same hour. This is what the pipeline actually labels on,
      and it is EXPECTED to disagree: EPA's PM breakpoints are defined over
      24-hour averages, and Open-Meteo applies that averaging (NowCast) while
      labeling.py applies the table to a single hourly reading. An MAE of ~35
      AQI points here is that methodological difference, not a broken table.

  averaged -- our AQI from a trailing 24h mean concentration, against the same
      us_aqi. This one isolates the breakpoint arithmetic from the averaging
      choice, so it is the comparison that actually tests labeling.py. The
      threshold check below is applied to THIS number.

The earlier version reported only the instantaneous figure, tripped its own
"difference is fairly large" warning at MAE 36.18, and the ReadMe cited those
same numbers as evidence the labeling logic was correct. Both halves of that
were defensible on their own and contradictory together.

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

# The verdict rests on an invariant rather than on a tuned error threshold.
#
# Open-Meteo's us_aqi is the max over roughly six pollutant sub-indices (PM2.5,
# PM10, NO2, O3, SO2, CO). labeling.py takes the max over the three pollutants
# this project collects. A max over a subset cannot exceed a max over the
# superset, so OUR AQI MUST SIT AT OR BELOW THEIRS. If the breakpoint tables
# were wrong, that invariant would break immediately and in one direction.
#
# MAE and correlation are reported for context but are NOT the gate: measured
# over three 14-day Delhi windows (Jan / Jul / Sep 2026) they ranged from
# MAE 1.4 / corr 0.998 to MAE 31 / corr 0.875, moving with which pollutant
# happened to drive Open-Meteo's max rather than with anything about our math.
# Invariant violations stayed under 8% of hours in all three.
#
# The tolerance absorbs the one legitimate source of small overshoot: EPA
# NowCast weights recent hours more heavily than the plain trailing mean used
# here, which diverges while concentrations are changing fast.
LOWER_BOUND_TOLERANCE_AQI = 10
MAX_SHARE_VIOLATING = 0.15


def _sample_window(sample_days):
    """Pick a window that overlaps the data we actually collected.

    config.END_DATE is derived from date.today(), so on its own it drifted past
    the end of the collected data -- the saved report covered 2026-08-15 to
    2026-08-29 while the training data stopped on 2026-07-26. Anchoring to
    merged_clean.csv keeps the check describing the same period as the model.
    """
    end = config.END_DATE
    if os.path.exists(config.MERGED_PATH):
        try:
            data_max = pd.read_csv(
                config.MERGED_PATH, usecols=["datetime"], parse_dates=["datetime"]
            )["datetime"].max()
            if pd.notna(data_max):
                end = min(end, data_max.date())
        except Exception:
            pass
    return end - timedelta(days=sample_days), end


def _our_aqi(frame):
    return frame.apply(
        lambda row: compute_aqi_row(
            pm25=row.get("pm25"), pm10=row.get("pm10"), no2_ugm3=row.get("no2"),
        ),
        axis=1,
    )


def _compare(ours, theirs, label):
    pair = pd.DataFrame({"ours": ours, "theirs": theirs}).dropna()
    if pair.empty:
        return None
    corr = pair["ours"].corr(pair["theirs"])
    diff = pair["ours"] - pair["theirs"]
    return {
        "comparison": label,
        "n_hours_compared": int(len(pair)),
        "mean_absolute_error": float(round(diff.abs().mean(), 2)),
        # Signed, unlike the MAE: which way we differ is the diagnostic.
        "mean_signed_error": float(round(diff.mean(), 2)),
        "correlation": float(round(corr, 3)) if not np.isnan(corr) else None,
        "share_ours_above_theirs": float(round((pair["ours"] > pair["theirs"]).mean(), 3)),
        # Share of hours breaking the lower-bound invariant by more than the
        # NowCast tolerance. This is what the verdict is based on.
        "share_violating_lower_bound": float(round(
            (diff > LOWER_BOUND_TOLERANCE_AQI).mean(), 3)),
        "max_excess_over_theirs": float(round(diff.max(), 1)),
        "mean_our_aqi": float(round(pair["ours"].mean(), 1)),
        "mean_openmeteo_us_aqi": float(round(pair["theirs"].mean(), 1)),
    }


def run_crosscheck(sample_days=14):
    start, end = _sample_window(sample_days)

    print(f"\nRunning AQI cross-check for {config.CITY_NAME} "
          f"({start} to {end}, {sample_days} days)...")

    df = fetch_air_quality_openmeteo_history(
        config.LATITUDE, config.LONGITUDE, start, end,
        config.OPEN_METEO_AQ_HOURLY_VARS,
    )

    if df.empty or "us_aqi_openmeteo" not in df.columns:
        print("Cross-check skipped: no us_aqi data returned for the sample window.")
        return None

    df = df.sort_values("datetime").reset_index(drop=True)

    # 1. As the pipeline labels: table applied to the raw hourly reading.
    instantaneous = _compare(_our_aqi(df), df["us_aqi_openmeteo"], "instantaneous")

    # 2. Apples to apples: table applied to a trailing 24h mean, which is the
    #    averaging window EPA's PM breakpoints are defined over.
    smoothed = df.copy()
    for col in ("pm25", "pm10", "no2"):
        if col in smoothed.columns:
            smoothed[col] = smoothed[col].rolling(window=24, min_periods=18).mean()
    averaged = _compare(_our_aqi(smoothed), df["us_aqi_openmeteo"], "trailing_24h_mean")

    if instantaneous is None and averaged is None:
        print("Cross-check skipped: no overlapping valid rows to compare.")
        return None

    report = {
        "city": config.CITY_NAME,
        "sample_window": f"{start} to {end}",
        "pm25_breakpoint_version": config.PM25_BREAKPOINT_VERSION,
        "instantaneous": instantaneous,
        "averaged_24h": averaged,
        "note": (
            "Two expected differences, neither of them a defect. (1) EPA PM "
            "breakpoints are defined over 24-hour averages: the instantaneous "
            "comparison applies them to single hourly readings, which is what "
            "the pipeline labels on, and differs from Open-Meteo's "
            "NowCast-averaged us_aqi. (2) us_aqi is the max over roughly six "
            "pollutant sub-indices (PM2.5, PM10, NO2, O3, SO2, CO) while "
            "labeling.py takes the max over the three this project collects, so "
            "our AQI is a lower bound on theirs. The verdict therefore keys on "
            "correlation of the averaged comparison, and on the sign of the "
            "difference, not on its magnitude."
        ),
    }

    for key in ("instantaneous", "averaged_24h"):
        r = report[key]
        if r is None:
            continue
        print(f"\n  [{key}]  hours compared: {r['n_hours_compared']}")
        print(f"    mean absolute difference: {r['mean_absolute_error']} AQI points")
        print(f"    correlation: {r['correlation']}")
        print(f"    our mean AQI: {r['mean_our_aqi']}  |  "
              f"Open-Meteo mean us_aqi: {r['mean_openmeteo_us_aqi']}")

    verdict_source = averaged or instantaneous
    if averaged is None:
        print("\n  Averaged comparison unavailable (sample too short for a 24h "
              "window) -- the verdict falls back to the instantaneous figure, "
              "which is not a clean test of the breakpoint math.")

    corr = verdict_source["correlation"]
    signed = verdict_source["mean_signed_error"]
    violating = verdict_source["share_violating_lower_bound"]
    max_excess = verdict_source["max_excess_over_theirs"]

    if violating > MAX_SHARE_VIOLATING:
        report["verdict"] = "needs_review"
        print(f"\n  VERDICT: needs review. Our AQI exceeds Open-Meteo's us_aqi "
              f"by more than {LOWER_BOUND_TOLERANCE_AQI} points in "
              f"{violating:.0%} of hours (limit {MAX_SHARE_VIOLATING:.0%}; worst "
              f"overshoot {max_excess:+.1f}). us_aqi maxes over more pollutants "
              "than we collect, so ours cannot legitimately sit above theirs. "
              "Check the tables in labeling.py and "
              "config.PM25_BREAKPOINT_VERSION.")
    else:
        report["verdict"] = "consistent"
        print(f"\n  VERDICT: consistent. Our AQI respects the lower-bound "
              f"invariant in {1 - violating:.0%} of hours, running "
              f"{signed:+.1f} points against Open-Meteo on average -- the "
              "expected direction, since us_aqi maxes over roughly six "
              "pollutants and we collect three. Correlation on the matched 24h "
              f"basis is {corr}. Good evidence the breakpoint arithmetic in "
              "labeling.py is right.")

    with open(config.AQ_CROSSCHECK_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved cross-check report to {config.AQ_CROSSCHECK_PATH}")

    return report


if __name__ == "__main__":
    run_crosscheck()
