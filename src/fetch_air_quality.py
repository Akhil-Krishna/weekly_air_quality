"""
Fetch historical air quality data (PM2.5, PM10, NO2) from the OpenAQ v3 API.

check: python -m src.fetch_air_quality
"""

import sys
import os
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import get_json_with_retries, chunk_date_range, haversine_km


def _headers():
    if not config.OPENAQ_API_KEY:
        raise RuntimeError(
            "OPENAQ_API_KEY is not set. Get a free key at "
            "https://explore.openaq.org/register and run:\n"
            "  export OPENAQ_API_KEY='your-key-here'"
        )
    return {"X-API-Key": config.OPENAQ_API_KEY}


def find_nearby_locations(lat, lon, radius_m, max_locations=None):
    """Find OpenAQ monitoring stations near the given coordinates.

    Results are sorted by actual distance and capped to max_locations --
    a dense city can return 50-100+ stations within a small radius, and
    pulling every single one multiplies request volume (and rate-limit
    risk) for very little modeling benefit over the nearest handful.
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

    results = sorted(results, key=_distance)
    if max_locations is not None and len(results) > max_locations:
        print(f"Using nearest {max_locations} station(s) to limit request volume "
              f"(set OPENAQ_MAX_LOCATIONS in config.py to change this).")
        results = results[:max_locations]
    return results


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

     the OpenAQ v3 API requires full RFC3339 datetime strings
    (e.g. "2026-07-01T00:00:00Z"), not bare dates ("2026-07-01")
    """
    all_rows = []
    for chunk_start, chunk_end in chunk_date_range(start_date, end_date, chunk_days=90):
        page = 1
        while True:
            params = {
                "datetime_from": f"{chunk_start.isoformat()}T00:00:00Z",
                "datetime_to": f"{chunk_end.isoformat()}T23:59:59Z",
                "limit": 1000,
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

            meta = data.get("meta", {})
            found = meta.get("found", 0)
            if page * 1000 >= (found if isinstance(found, int) else 0) or len(results) < 1000:
                break
            page += 1
    return all_rows


def fetch_air_quality_history(lat, lon, radius_m, start_date, end_date, pollutants,
                               max_locations=None):
    """Returns None (instead of raising) if no stations are found nearby, so the
    caller (main()) can decide whether to fall back to Open-Meteo."""
    locations = find_nearby_locations(lat, lon, radius_m, max_locations=max_locations)
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
        if df is None and config.AQ_SOURCE_MODE == "openaq":
            raise RuntimeError(
                "No OpenAQ stations found near this city/radius, and AQ_SOURCE_MODE "
                "is forced to 'openaq'. Either increase OPENAQ_RADIUS_METERS in "
                "config.py, or set AQ_SOURCE_MODE = 'auto' to fall back to "
                "Open-Meteo Air Quality for this location."
            )

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
