"""
Fetch historical air quality data (PM2.5, PM10, NO2) from the OpenAQ v3 API.

IMPORTANT: As of the OpenAQ v3 API, a free API key is REQUIRED.
Register at https://explore.openaq.org/register, then either:
    export OPENAQ_API_KEY="your-key-here"
or put it in a .env file (see .env.example) and load it before running.

Docs: https://docs.openaq.org/

Flow:
    1. Find monitoring stations ("locations") near the target city coordinates.
    2. For each location, list its sensors (one sensor per pollutant/parameter).
    3. For each relevant sensor (pm25 / pm10 / no2), pull hourly measurements
       for the requested date range.
    4. Combine everything into one long dataframe, then pivot to wide format
       (one column per pollutant) for merging with weather data.

Usage:
    python -m src.fetch_air_quality
"""

import sys
import os
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import get_json_with_retries, chunk_date_range, haversine_km


# OpenAQ v3 caps `limit` at 1000 per page. A 90-day chunk is 2160 hours, so a
# fully-populated sensor needs 3 pages; the cap is a runaway guard, not a limit
# we expect to reach.
PAGE_LIMIT = 1000
MAX_PAGES_PER_CHUNK = 20


def _headers():
    if not config.OPENAQ_API_KEY:
        raise RuntimeError(
            "OPENAQ_API_KEY is not set. Get a free key at "
            "https://explore.openaq.org/register and run:\n"
            "  export OPENAQ_API_KEY='your-key-here'"
        )
    return {"X-API-Key": config.OPENAQ_API_KEY}


def _last_report_date(location):
    """Date a station last reported, or None if OpenAQ does not say."""
    stamp = (location.get("datetimeLast") or {}).get("utc")
    if not stamp:
        return None
    try:
        return pd.to_datetime(stamp, utc=True, errors="coerce").date()
    except (TypeError, ValueError):
        return None


def find_nearby_locations(lat, lon, radius_m, max_locations=None, active_since=None):
    """Find OpenAQ monitoring stations near the given coordinates.

    Stations are capped to max_locations because a dense city returns 50-100+
    within a small radius, and pulling every one multiplies request volume (and
    rate-limit risk) for little modeling benefit over the nearest handful.

    Which ones get those slots now depends on whether the station is still
    alive, not only on how close it is. Sorting by distance alone spent three
    of Delhi's six slots on stations whose last reading was in 2016, 2018 and
    2018 -- so half the request budget produced nothing, while ~90 live
    stations slightly further out were never queried.
    """
    params = {
        "coordinates": f"{lat},{lon}",
        "radius": radius_m,
        "limit": 100,
    }
    data = get_json_with_retries(
        f"{config.OPENAQ_BASE_URL}/locations", params=params, headers=_headers()
    )
    results = data.get("results", [])
    print(f"Found {len(results)} OpenAQ location(s) within {radius_m/1000:.0f} km of "
          f"({lat}, {lon})")

    def _distance(loc):
        coords = loc.get("coordinates") or {}
        loc_lat, loc_lon = coords.get("latitude"), coords.get("longitude")
        if loc_lat is None or loc_lon is None:
            return float("inf")
        return haversine_km(lat, lon, loc_lat, loc_lon)

    if active_since is not None:
        live, undated, retired = [], [], []
        for loc in results:
            last = _last_report_date(loc)
            if last is None:
                # No metadata either way. Keep it, but behind the stations we
                # can positively confirm are reporting.
                undated.append(loc)
            elif last >= active_since:
                live.append(loc)
            else:
                retired.append((loc, last))

        if retired:
            sample = "; ".join(
                "{} (last {})".format(loc.get("name", "?"), last)
                for loc, last in retired[:3]
            )
            print(f"Skipping {len(retired)} station(s) that stopped reporting "
                  f"before {active_since}. Nearest few: {sample}")
        results = sorted(live, key=_distance) + sorted(undated, key=_distance)
        print(f"{len(live)} station(s) reported on or after {active_since}"
              + (f", plus {len(undated)} with no date metadata" if undated else ""))

    else:
        results = sorted(results, key=_distance)

    if max_locations is not None and len(results) > max_locations:
        print(f"Using nearest {max_locations} station(s) to limit request volume "
              f"(set OPENAQ_MAX_LOCATIONS in config.py to change this).")
        results = results[:max_locations]

    for loc in results:
        print(f"    -> {loc.get('name', '?')}  ({_distance(loc):.1f} km, "
              f"last reported {_last_report_date(loc)})")
    return results


def assess_ground_coverage(df, start_date, end_date):
    """Is this OpenAQ result actually usable for the requested window?

    Returns (usable, reason).

    "Does a station exist nearby" is not the same question as "is there usable
    ground data". Hamilton has two stations 0.7km and 2.7km away, both of which
    stopped on 2025-12-03 -- so the existence check passed and the city was
    trained on 2 months out of a 12-month window, with a Historical tab stuck
    in 2025 and a forecast resting on a 10-month-old pollution baseline.

    Both conditions below have to hold, because they catch different failures:
    coverage catches a station that reports sporadically, staleness catches one
    that reported well and then died.
    """
    if df is None or df.empty:
        return False, "no rows returned"

    window_hours = len(pd.date_range(pd.Timestamp(start_date),
                                     pd.Timestamp(end_date), freq="h"))
    coverage = len(df) / window_hours if window_hours else 0.0
    last = df["datetime"].max()
    lag_days = (pd.Timestamp(end_date) - last).days

    if coverage < config.OPENAQ_MIN_WINDOW_COVERAGE:
        return False, (f"covers only {coverage:.0%} of the requested window "
                       f"({len(df)} of {window_hours} hours; minimum "
                       f"{config.OPENAQ_MIN_WINDOW_COVERAGE:.0%})")
    if lag_days > config.OPENAQ_MAX_STALENESS_DAYS:
        return False, (f"last reading is {lag_days} days before the end of the "
                       f"requested window ({last.date()}; maximum "
                       f"{config.OPENAQ_MAX_STALENESS_DAYS} days)")
    return True, (f"covers {coverage:.0%} of the window, up to {last.date()}")


def get_sensors_for_location(location):
    """Extract sensor id + parameter name pairs from a location record."""
    sensors = []
    for s in location.get("sensors", []):
        param_name = (s.get("parameter") or {}).get("name", "").lower()
        sensors.append({"sensor_id": s["id"], "parameter": param_name,
                         "location_name": location.get("name"),
                         "location_id": location.get("id")})
    return sensors


def fetch_sensor_measurements(sensor_id, start_date, end_date):
    """Pull hourly aggregated measurements for one sensor, paginated.

    NOTE: the OpenAQ v3 API requires full RFC3339 datetime strings
    (e.g. "2026-07-01T00:00:00Z"), not bare dates ("2026-07-01") -- passing
    date-only strings returns a 422 Unprocessable Entity on every request.

    A short delay between requests is added proactively (rather than only
    reacting after a 429) since sustained bursts are what trip rate limits
    in the first place.

    Pagination continues until the server returns a page shorter than `limit`.
    Do NOT reintroduce a meta.found check here -- see the comment at the break.
    """
    all_rows = []
    limit = PAGE_LIMIT
    for chunk_start, chunk_end in chunk_date_range(start_date, end_date, chunk_days=90):
        page = 1
        while True:
            params = {
                "datetime_from": f"{chunk_start.isoformat()}T00:00:00Z",
                "datetime_to": f"{chunk_end.isoformat()}T23:59:59Z",
                "limit": limit,
                "page": page,
            }
            time.sleep(config.OPENAQ_REQUEST_DELAY_SECONDS)
            try:
                data = get_json_with_retries(
                    f"{config.OPENAQ_BASE_URL}/sensors/{sensor_id}/hours",
                    params=params, headers=_headers(),
                )
            except RuntimeError as e:
                print(f"    Failed fetching sensor {sensor_id} page {page}: {e}")
                break

            results = data.get("results", [])
            if not results:
                break
            for r in results:
                period = r.get("period", {})
                dt = (period.get("datetimeFrom") or {}).get("utc") or period.get("datetime_from")
                value = r.get("value")
                all_rows.append({"datetime": dt, "value": value})

            # Stop ONLY when the server returns a short page. A full page means
            # there may well be another one.
            #
            # The previous version also consulted meta.found, which looks like a
            # count but is documented to come back as the STRING ">1000" once the
            # result set exceeds one page. `isinstance(found, int)` was then False,
            # `found` defaulted to 0, and `page * 1000 >= 0` was trivially true --
            # so the loop broke after page 1 and kept only the first 1000 hours of
            # every 90-day (2160-hour) chunk. That silently discarded ~40% of
            # Delhi's history in contiguous, chunk-aligned blocks, including the
            # start of stubble-burning season and peak winter smog, which is
            # exactly where the Very Unhealthy / Hazardous hours live.
            if len(results) < limit:
                break
            if page >= MAX_PAGES_PER_CHUNK:
                print(f"    WARNING: hit the {MAX_PAGES_PER_CHUNK}-page safety cap for "
                      f"sensor {sensor_id} on chunk {chunk_start} -> {chunk_end}; "
                      "some hours in this window may be missing.")
                break
            page += 1
    return all_rows


def fetch_air_quality_history(lat, lon, radius_m, start_date, end_date, pollutants,
                               max_locations=None):
    """Returns None (instead of raising) if no stations are found nearby, so the
    caller (main()) can decide whether to fall back to Open-Meteo."""
    locations = find_nearby_locations(lat, lon, radius_m, max_locations=max_locations,
                                      active_since=start_date)
    if not locations:
        return None

    all_records = []
    for loc in locations:
        sensors = get_sensors_for_location(loc)
        relevant = [s for s in sensors if s["parameter"] in pollutants]
        for s in relevant:
            print(f"Fetching {s['parameter']} from station '{s['location_name']}' "
                  f"(sensor {s['sensor_id']}) ...")
            rows = fetch_sensor_measurements(s["sensor_id"], start_date, end_date)
            for r in rows:
                all_records.append({
                    "datetime": r["datetime"],
                    "parameter": s["parameter"],
                    "value": r["value"],
                    "location_name": s["location_name"],
                })

    if not all_records:
        return None

    df_long = pd.DataFrame(all_records)
    df_long["datetime"] = pd.to_datetime(df_long["datetime"], utc=True).dt.tz_localize(None)

    # Average across stations for each timestamp+pollutant, then pivot wide
    df_avg = df_long.groupby(["datetime", "parameter"], as_index=False)["value"].mean()
    df_wide = df_avg.pivot(index="datetime", columns="parameter", values="value").reset_index()
    df_wide = df_wide.sort_values("datetime").reset_index(drop=True)
    df_wide["source"] = "openaq_ground"
    return df_wide


def main():
    """
    Air quality collection with automatic fallback (config.AQ_SOURCE_MODE = "auto"):
      1. Try OpenAQ first -- real ground-station measurements where available.
      2. If no station is found nearby, fall back to Open-Meteo Air Quality
         (model/satellite-based, works for any coordinate) so the pipeline
         doesn't just fail for places with no ground station.
      3. Separately, pull a small Open-Meteo `us_aqi` sample to cross-check
         our own EPA labeling.py math (see config.RUN_AQI_CROSSCHECK).
    """
    print(f"Collecting historical air quality for {config.CITY_NAME} "
          f"({config.LATITUDE}, {config.LONGITUDE})")

    df = None
    if config.AQ_SOURCE_MODE in ("auto", "openaq"):
        df = fetch_air_quality_history(
            config.LATITUDE, config.LONGITUDE, config.OPENAQ_RADIUS_METERS,
            config.START_DATE, config.END_DATE, config.POLLUTANTS,
            max_locations=config.OPENAQ_MAX_LOCATIONS,
        )

        # Having found a station is not the same as having usable data from it.
        if df is not None:
            usable, reason = assess_ground_coverage(
                df, config.START_DATE, config.END_DATE
            )
            if usable:
                print(f"OpenAQ ground data accepted: {reason}")
            else:
                print(f"OpenAQ ground data REJECTED for {config.CITY_NAME}: "
                      f"{reason}.")
                if config.AQ_SOURCE_MODE == "openaq":
                    print("AQ_SOURCE_MODE is forced to 'openaq', so keeping it "
                          "anyway -- set it to 'auto' to allow the Open-Meteo "
                          "fallback for cities whose stations have gone quiet.")
                else:
                    print("Falling back to Open-Meteo Air Quality so this city "
                          "gets the full requested window.")
                    df = None

        if df is None and config.AQ_SOURCE_MODE == "openaq":
            raise RuntimeError(
                "No usable OpenAQ data for this city/radius, and AQ_SOURCE_MODE "
                "is forced to 'openaq'. Either increase OPENAQ_RADIUS_METERS in "
                "config.py, or set AQ_SOURCE_MODE = 'auto' to fall back to "
                "Open-Meteo Air Quality for this location."
            )

    if df is None:
        # Either AQ_SOURCE_MODE == "openmeteo" (forced), or "auto" with no
        # usable OpenAQ data -- fall back to the model-based source.
        from src.fetch_air_quality_openmeteo import fetch_air_quality_openmeteo_history
        print("Using Open-Meteo Air Quality (model-based, works for any "
              "coordinate). Every row is tagged source='openmeteo_model' so "
              "this is never confused with ground-sensor data.")
        df = fetch_air_quality_openmeteo_history(
            config.LATITUDE, config.LONGITUDE,
            config.START_DATE, config.END_DATE,
            config.OPEN_METEO_AQ_HOURLY_VARS,
        )
        df["source"] = "openmeteo_model"
        # Drop the us_aqi column from the training data itself -- it's kept
        # separately for the cross-check, not as a training feature (using a
        # third party's AQI to predict our own AQI category would be circular).
        if "us_aqi_openmeteo" in df.columns:
            df = df.drop(columns=["us_aqi_openmeteo"])

    df.to_csv(config.AQ_RAW_PATH, index=False)
    print(f"Saved {len(df)} rows to {config.AQ_RAW_PATH} (source: {df['source'].iloc[0]})")

    if config.RUN_AQI_CROSSCHECK:
        try:
            from src.validate_aqi_crosscheck import run_crosscheck
            run_crosscheck()
        except Exception as e:
            print(f"AQI cross-check skipped (non-fatal): {e}")


if __name__ == "__main__":
    main()
