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
