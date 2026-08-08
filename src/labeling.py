"""
Convert raw pollutant concentrations into an AQI value and a human-readable
risk category, using the standard US EPA breakpoint tables.

The overall AQI for a given hour is the MAXIMUM of the individual pollutant
sub-indices (this mirrors how real-world AQI is reported).
"""

import numpy as np
import pandas as pd

# (conc_low, conc_high, aqi_low, aqi_high) breakpoints per pollutant.
# PM2.5 and PM10 in ug/m3 (24h-style breakpoints applied here to hourly/rolling
# values as a reasonable approximation, which is standard practice for
# near-real-time / project-level AQI estimation).
# NO2 breakpoints are in ppb; OpenAQ typically reports NO2 in ug/m3, so we
# convert using the standard factor 1 ppb NO2 ~= 1.88 ug/m3 at 25C/1atm.
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]

NO2_BREAKPOINTS_PPB = [
    (0, 53, 0, 50),
    (54, 100, 51, 100),
    (101, 360, 101, 150),
    (361, 649, 151, 200),
    (650, 1249, 201, 300),
    (1250, 1649, 301, 400),
    (1650, 2049, 401, 500),
]

NO2_UGM3_TO_PPB = 1 / 1.88

RISK_CATEGORIES = [
    (0, 50, "Good"),
    (51, 100, "Moderate"),
    (101, 150, "Unhealthy (Sensitive Groups)"),
    (151, 200, "Unhealthy"),
    (201, 300, "Very Unhealthy"),
    (301, 10_000, "Hazardous"),
]


def _sub_index(value, breakpoints):
    if pd.isna(value) or value < 0:
        return np.nan
    for lo, hi, aqi_lo, aqi_hi in breakpoints:
        if lo <= value <= hi:
            return aqi_lo + (aqi_hi - aqi_lo) / (hi - lo) * (value - lo)
    # Above the top breakpoint: extrapolate off the last band, capped at 500
    lo, hi, aqi_lo, aqi_hi = breakpoints[-1]
    if value > hi:
        return 500.0
    return np.nan


def compute_aqi_row(pm25=None, pm10=None, no2_ugm3=None):
    """Compute the overall AQI (max of sub-indices) for one set of readings."""
    sub_indices = []
    if pm25 is not None:
        sub_indices.append(_sub_index(pm25, PM25_BREAKPOINTS))
    if pm10 is not None:
        sub_indices.append(_sub_index(pm10, PM10_BREAKPOINTS))
    if no2_ugm3 is not None:
        no2_ppb = no2_ugm3 * NO2_UGM3_TO_PPB
        sub_indices.append(_sub_index(no2_ppb, NO2_BREAKPOINTS_PPB))

    valid = [s for s in sub_indices if not pd.isna(s)]
    if not valid:
        return np.nan
    return max(valid)


def aqi_to_category(aqi):
    if pd.isna(aqi):
        return np.nan
    for lo, hi, label in RISK_CATEGORIES:
        if lo <= aqi <= hi:
            return label
    return "Hazardous"


def label_dataframe(df, pm25_col="pm25", pm10_col="pm10", no2_col="no2"):
    """Add 'aqi' and 'risk_category' columns to a dataframe with pollutant columns."""
    df = df.copy()
    df["aqi"] = df.apply(
        lambda row: compute_aqi_row(
            pm25=row.get(pm25_col),
            pm10=row.get(pm10_col),
            no2_ugm3=row.get(no2_col),
        ),
        axis=1,
    )
    df["risk_category"] = df["aqi"].apply(aqi_to_category)
    return df
