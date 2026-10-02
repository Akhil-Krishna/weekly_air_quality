"""
Convert raw pollutant concentrations into an AQI value and a human-readable
risk category, using the US EPA breakpoint tables.

The overall AQI for a given hour is the MAXIMUM of the individual pollutant
sub-indices (this mirrors how real-world AQI is reported).

Two things here are easy to get wrong and are worth reading before editing:

1. EPA's published tables have GAPS between bands (PM2.5 runs 0.0-12.0 then
   12.1-35.4). That is not a typo in the table -- EPA's method truncates the
   concentration to the table's own precision first, which makes the gaps
   unreachable. Skipping that truncation step meant a reading of 12.05 matched
   no band at all and returned NaN, so that pollutant silently dropped out of
   the max() and the hour's AQI was understated whenever it was the driver.
   (~0.3% of readings in the collected data.)

2. The PM2.5 table was revised by EPA in 2024. Which one is in use is a
   config switch, not a silent default -- see config.PM25_BREAKPOINT_VERSION.
"""

import numpy as np
import pandas as pd

try:
    import config
except ImportError:  # importable standalone (e.g. for unit tests)
    config = None

# (conc_low, conc_high, aqi_low, aqi_high) breakpoints per pollutant.
#
# PM2.5, 2024 revision (EPA final rule, Feb 2024). The headline change is the
# Good/Moderate boundary dropping from 12.0 to 9.0 ug/m3; the upper bands were
# restructured too.
PM25_BREAKPOINTS_2024 = [
    (0.0, 9.0, 0, 50),
    (9.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 125.4, 151, 200),
    (125.5, 225.4, 201, 300),
    (225.5, 325.4, 301, 500),
]

# PM2.5, pre-2024 table. Kept only so results generated before the switch can
# be reproduced; not the current standard.
PM25_BREAKPOINTS_2012 = [
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

# Decimal places EPA truncates each pollutant to before looking the value up
# in the table above. This is step 1 of the official procedure, and it is what
# makes the gaps between bands unreachable.
TRUNCATE_DECIMALS = {"pm25": 1, "pm10": 0, "no2_ppb": 0}


def pm25_breakpoints(version=None):
    """The PM2.5 table named by config.PM25_BREAKPOINT_VERSION (default 2024)."""
    if version is None:
        version = getattr(config, "PM25_BREAKPOINT_VERSION", "2024") if config else "2024"
    version = str(version)
    if version == "2024":
        return PM25_BREAKPOINTS_2024
    if version == "2012":
        return PM25_BREAKPOINTS_2012
    raise ValueError(
        f"Unknown PM25_BREAKPOINT_VERSION {version!r} -- expected '2024' or '2012'."
    )


# Risk categories as (upper_bound_inclusive, label), evaluated in order.
# Expressed as upper bounds rather than EPA's printed (51, 100)-style ranges
# because those ranges have the same gap problem as the concentration tables:
# a lookup of 50.5 matched nothing, fell through to a bare "Hazardous" default,
# and labelled a near-Good reading as the worst category on the scale.
RISK_CATEGORY_BANDS = [
    (50, "Good"),
    (100, "Moderate"),
    (150, "Unhealthy (Sensitive Groups)"),
    (200, "Unhealthy"),
    (300, "Very Unhealthy"),
    (float("inf"), "Hazardous"),
]

# Canonical severity order. Used for label encoding and for every chart axis,
# so a confusion matrix reads Good -> Hazardous instead of alphabetically
# (which put "Hazardous" between "Good" and "Moderate").
RISK_CATEGORY_ORDER = [label for _, label in RISK_CATEGORY_BANDS]


def _truncate(value, decimals):
    """Truncate (not round) toward zero, as the EPA AQI procedure specifies."""
    factor = 10 ** decimals
    return np.trunc(value * factor) / factor


def _sub_index(value, breakpoints, decimals=1):
    """AQI sub-index for one pollutant concentration.

    Returns NaN only for genuinely unusable input (missing or negative), never
    for a value that simply fell between two printed bands.
    """
    if value is None or pd.isna(value) or value < 0:
        return np.nan

    value = _truncate(float(value), decimals)

    for lo, hi, aqi_lo, aqi_hi in breakpoints:
        if lo <= value <= hi:
            if hi == lo:
                return float(aqi_lo)
            return aqi_lo + (aqi_hi - aqi_lo) / (hi - lo) * (value - lo)

    top_hi = breakpoints[-1][1]
    if value > top_hi:
        # Above the top of the scale. EPA caps the reported index at 500.
        return 500.0

    # Unreachable after truncation, but if the table is ever edited into a
    # genuinely discontinuous state, fall back to the band immediately below
    # rather than dropping this pollutant out of the overall max() in silence.
    below = [bp for bp in breakpoints if bp[1] < value]
    if below:
        return float(below[-1][3])
    return float(breakpoints[0][2])


def compute_aqi_row(pm25=None, pm10=None, no2_ugm3=None, pm25_version=None):
    """Compute the overall AQI (max of sub-indices) for one set of readings."""
    sub_indices = []
    if pm25 is not None:
        sub_indices.append(
            _sub_index(pm25, pm25_breakpoints(pm25_version), TRUNCATE_DECIMALS["pm25"])
        )
    if pm10 is not None:
        sub_indices.append(
            _sub_index(pm10, PM10_BREAKPOINTS, TRUNCATE_DECIMALS["pm10"])
        )
    if no2_ugm3 is not None and not pd.isna(no2_ugm3):
        no2_ppb = no2_ugm3 * NO2_UGM3_TO_PPB
        sub_indices.append(
            _sub_index(no2_ppb, NO2_BREAKPOINTS_PPB, TRUNCATE_DECIMALS["no2_ppb"])
        )

    valid = [s for s in sub_indices if not pd.isna(s)]
    if not valid:
        return np.nan
    return max(valid)


def aqi_to_category(aqi):
    """Map an AQI value to its risk category.

    Accepts any AQI, including one computed elsewhere (Open-Meteo us_aqi, say),
    not only output from compute_aqi_row.
    """
    if aqi is None or pd.isna(aqi):
        return np.nan
    # EPA reports AQI as a whole number; round before banding so 50.4 reads as
    # Good rather than tipping into Moderate.
    aqi = float(np.round(float(aqi)))
    if aqi < 0:
        return np.nan
    for upper, label in RISK_CATEGORY_BANDS:
        if aqi <= upper:
            return label
    return RISK_CATEGORY_ORDER[-1]


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


class OrderedLabelEncoder:
    """Label encoder whose integer codes follow AQI severity, not the alphabet.

    The sklearn LabelEncoder sorts classes with np.unique, which is
    alphabetical: Good=0, Hazardous=1, Moderate=2, ... That is harmless to a
    nominal classifier, but it puts Hazardous between Good and Moderate on
    every confusion-matrix axis and in every saved report, which is actively
    misleading to read.

    Drop-in for the LabelEncoder API this project uses (fit / transform /
    inverse_transform / classes_), and picklable by joblib so the app loads it
    exactly as before.
    """

    def __init__(self, order=None):
        self.order = list(order) if order is not None else list(RISK_CATEGORY_ORDER)
        self.classes_ = None

    def fit(self, y):
        present = set(pd.Series(y).dropna().unique())
        unknown = present - set(self.order)
        if unknown:
            raise ValueError(
                f"Labels not in the known severity order: {sorted(unknown)}. "
                "Add them to RISK_CATEGORY_BANDS in src/labeling.py."
            )
        # Keep every known class in severity order, including ones absent from
        # this particular dataset, so codes mean the same thing across cities.
        self.classes_ = np.array(self.order, dtype=object)
        return self

    def transform(self, y):
        if self.classes_ is None:
            raise RuntimeError("OrderedLabelEncoder must be fit before transform.")
        lookup = {label: i for i, label in enumerate(self.classes_)}
        out = []
        for v in pd.Series(y):
            if v not in lookup:
                raise ValueError(f"Unseen label {v!r}.")
            out.append(lookup[v])
        return np.array(out, dtype=int)

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, codes):
        if self.classes_ is None:
            raise RuntimeError("OrderedLabelEncoder must be fit before inverse_transform.")
        codes = np.asarray(codes, dtype=int)
        if codes.size and (codes.min() < 0 or codes.max() >= len(self.classes_)):
            raise ValueError(f"Code out of range for {len(self.classes_)} classes.")
        return np.array([self.classes_[c] for c in codes], dtype=object)
