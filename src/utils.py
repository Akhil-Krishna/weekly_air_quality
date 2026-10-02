"""Shared helper functions used across the pipeline."""

import sys
import time
import requests


def get_json_with_retries(url, params=None, headers=None, max_retries=6, backoff=3.0):
    """GET a URL and return parsed JSON, retrying on transient failures / rate limits.

    Client errors other than 429 (e.g. 400/401/404/422) are NOT retried --
    those mean the request itself is malformed or unauthorized, and retrying
    identical bad params 4 times just wastes time before failing anyway.

    429 (rate limited) gets its own longer, exponential backoff and honors
    the server's `Retry-After` header when present, since a few seconds is
    rarely enough to clear a per-minute rate window.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = backoff * (2 ** attempt)
                else:
                    wait = backoff * (2 ** attempt)  # 3, 6, 12, 24, 48, 96s
                wait = min(wait, 120)
                print(f"  Rate limited (429). Waiting {wait:.0f}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            if 400 <= resp.status_code < 500:
                # Non-transient client error -- fail fast with the response body,
                # which usually explains exactly what was wrong with the request.
                raise RuntimeError(
                    f"{resp.status_code} error for {resp.url}: {resp.text[:300]}"
                )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            last_err = e
            wait = backoff * (attempt + 1)
            print(f"  Request failed ({e}). Retrying in {wait:.0f}s...", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Failed after {max_retries} retries: {last_err}")


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points."""
    from math import radians, sin, cos, sqrt, atan2

    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * r * atan2(sqrt(a), sqrt(1 - a))


def chunk_date_range(start_date, end_date, chunk_days=30):
    """Yield (chunk_start, chunk_end) tuples covering [start_date, end_date]."""
    from datetime import timedelta

    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_date)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def derive_split_date(datetimes, test_months=3):
    """Chronological train/test split point derived from the DATA, not the clock.

    Returns the timestamp at which the test period begins: the most recent
    `test_months` (counted as 30 days each) of whatever data actually exists.

    This exists because the split used to be `date.today() - 7 - 90 days`.
    Nothing about it referred to the dataset, so the held-out window silently
    shrank as the data aged -- the saved Delhi model was evaluated on 461 rows
    (~19 days) while the ReadMe claimed "the last 3 months" -- and once the data
    was more than ~97 days old the test set became empty altogether.

    If the data spans less than ~4x the requested test window, the test period
    is capped at 25% of the span so there is always something left to train on.
    """
    import pandas as pd

    s = pd.to_datetime(pd.Series(list(datetimes))).dropna()
    if s.empty:
        raise ValueError("Cannot derive a split date from an empty datetime series.")
    data_min, data_max = s.min(), s.max()
    span_days = max((data_max - data_min).days, 1)
    test_days = min(test_months * 30, max(1, int(span_days * 0.25)))
    return data_max - pd.Timedelta(days=test_days)


def to_local(utc_naive_series, tz):
    """Convert a UTC-naive datetime Series to naive local wall-clock time.

    Every file in this project stores `datetime` as UTC (so weather and
    pollution join on one clock). Hour-of-day, day-of-week and season are only
    meaningful on a local clock, so they are derived through this. Going via a
    real IANA zone rather than a fixed offset is what makes DST correct, which
    matters for the NZ cities -- their offset to UTC is 12h or 13h depending on
    the date.
    """
    import pandas as pd

    s = pd.to_datetime(utc_naive_series)
    if getattr(s.dt, "tz", None) is None:
        s = s.dt.tz_localize("UTC")
    return s.dt.tz_convert(tz).dt.tz_localize(None)


# Datetime columns the pipeline writes to CSV. "datetime" is the UTC join key
# and is always present; "datetime_local" is added by feature_engineering, so
# features.csv has it and merged_clean.csv does not.
PIPELINE_DATETIME_COLS = ("datetime", "datetime_local")


def read_pipeline_csv(path):
    """Read a pipeline CSV with every datetime column it has properly typed.

    Naming datetime_local in parse_dates unconditionally raises on
    merged_clean.csv, which does not have that column; leaving it out left it
    as plain strings, so .min().to_pydatetime() raised AttributeError in the
    app's date picker. Checking per column is what makes one reader correct
    for both files.
    """
    import pandas as pd

    df = pd.read_csv(path)
    for col in PIPELINE_DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df
