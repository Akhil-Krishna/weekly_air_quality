"""
Central configuration for the Air Quality Risk Predictor.

To target a different city, add it to the CITIES registry below (coordinates
plus IANA timezone) and select it with the AQ_CITY environment variable, or
just let run_pipeline.py do it:

    python run_pipeline.py --city Delhi
    python run_pipeline.py --all

Everything downstream (data collection, training, app) reads from here.
"""

import os
from datetime import date, timedelta
from dotenv import load_dotenv


load_dotenv()

# ---------------------------------------------------------------------------
# CITY REGISTRY -- every city the app can switch between.
#
# Every city carries its IANA timezone. All data is fetched, stored and joined
# in UTC (see TIMEZONE_FETCH below); the timezone here is used only to derive
# LOCAL-clock features (hour-of-day, day-of-week, season) at feature-engineering
# time. Those features are only meaningful on a local clock -- "rush hour" and
# "winter" are local concepts -- while the join key has to be a single global
# clock or weather and pollution get silently misaligned.
CITIES = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "tz": "Asia/Kolkata"},
    "Auckland": {"lat": -36.8485, "lon": 174.7633, "tz": "Pacific/Auckland"},
    # NZ cities. Ground-station availability varies and is NOT guaranteed:
    # OpenAQ currently returns no stations at all near Wellington, and
    # Hamilton's two stations stopped reporting on 2025-12-03. Both therefore
    # fall back to Open-Meteo automatically (see OPENAQ_MIN_WINDOW_COVERAGE),
    # and every row records which source was used in its "source" column.
    "Wellington": {"lat": -41.2866, "lon": 174.7756, "tz": "Pacific/Auckland"},
    "Christchurch": {"lat": -43.5333, "lon": 172.6333, "tz": "Pacific/Auckland"},
    "Hamilton": {"lat": -37.7833, "lon": 175.2833, "tz": "Pacific/Auckland"},
    "Dunedin": {"lat": -45.8742, "lon": 170.5036, "tz": "Pacific/Auckland"},
}

# Timezone requested from Open-Meteo. This MUST stay "UTC": OpenAQ reports in
# UTC, and asking Open-Meteo for "auto" (city-local) time while OpenAQ reports
# UTC made merge_clean.py inner-join two different clocks -- offsetting Delhi's
# labels from its features by 5.5h and Auckland's by 12-13h (non-constant,
# because NZ observes DST). Everything is fetched in UTC and converted to local
# only for the time-of-day/season features.
TIMEZONE_FETCH = "UTC"


def city_timezone(city_name):
    """IANA timezone for a city in CITIES; falls back to UTC if unregistered."""
    return (CITIES.get(city_name) or {}).get("tz", "UTC")


# ---------------------------------------------------------------------------
# WHICH CITY this run targets.
#
# Set it with the AQ_CITY environment variable, or let run_pipeline.py do it:
#     python run_pipeline.py --city Delhi
#     python run_pipeline.py --all
#
# DEFAULT_CITY is the fallback when AQ_CITY is unset. Coordinates are read out
# of CITIES above rather than kept in separate constants -- CITY_NAME, LATITUDE
# and LONGITUDE used to be three independent values that had to be edited
# together, so any slip pointed the fetchers at one city and wrote the results
# into another city's folder.
# ---------------------------------------------------------------------------
DEFAULT_CITY = "Delhi"
CITY_NAME = (os.environ.get("AQ_CITY") or DEFAULT_CITY).strip()

if CITY_NAME not in CITIES:
    raise SystemExit(
        f"Unknown city {CITY_NAME!r}. Add it to the CITIES dict in config.py "
        f"(with its lat/lon and IANA timezone) or pick one of: "
        f"{', '.join(CITIES)}"
    )

LATITUDE = CITIES[CITY_NAME]["lat"]
LONGITUDE = CITIES[CITY_NAME]["lon"]

OPENAQ_RADIUS_METERS = 25000  # search radius around the coordinates for OpenAQ stations
OPENAQ_MAX_LOCATIONS = 6      # only pull the N nearest stations -- pulling all ~100 in a
                              # dense city like Delhi triggers rate limits for no real benefit
OPENAQ_REQUEST_DELAY_SECONDS = 1.2  # pause between requests to stay under the rate limit

# When to stop trusting OpenAQ ground data for a city and fall back to the
# model-based Open-Meteo source. "Is there a station nearby" turned out to be
# the wrong question: Hamilton has two stations 0.7km and 2.7km away that both
# stopped reporting on 2025-12-03, so the existence check passed while the city
# got 2 months of a 12-month window -- a Historical tab stuck in 2025 and a
# forecast resting on a 10-month-old pollution baseline.
#
# Both thresholds are checked, because they catch different failures: coverage
# catches a station that reports sporadically, staleness catches one that
# reported well and then died.
OPENAQ_MIN_WINDOW_COVERAGE = 0.50  # fraction of the requested hours required
OPENAQ_MAX_STALENESS_DAYS = 30     # how far behind END_DATE the last reading may be

# ---------------------------------------------------------------------------
# DATE RANGE for historical data collection
# ---------------------------------------------------------------------------
HISTORY_MONTHS = 9  # 9 months train + we'll reserve the tail for testing
_today = date.today()
END_DATE = _today - timedelta(days=7)
START_DATE = END_DATE - timedelta(days=HISTORY_MONTHS * 30 + 90)  # ~9-12 months total

# Chronological split: the most recent TEST_MONTHS of DATA (not of the wall
# clock) are held out as the test set.
#
# There is deliberately no module-level SPLIT_DATE constant any more. One
# derived from date.today() drifts every day the pipeline isn't re-run: once
# the data was a few weeks old the "last 3 months" test window silently shrank
# to 19 days, and later became empty entirely. The split is now computed from
# df["datetime"].max() at train time (see src/utils.derive_split_date) and
# persisted into model_comparison.json, so the app always reports the split the
# saved model was actually trained with.
TEST_MONTHS = 3

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPENAQ_BASE_URL = "https://api.openaq.org/v3"

OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY", "")
# ---------------------------------------------------------------------------
# Atmospore Pollen API (Allergy tab)
# ---------------------------------------------------------------------------
ATMOSPORE_API_KEY = os.environ.get("ATMOSPORE_API_KEY", "")
ATMOSPORE_BASE_URL = "https://pollenapi.com/v1"
ATMOSPORE_FORECAST_DAYS = 7

# Free tier is 100 calls/day, shared across EVERY city this app supports
# (one API key, not one budget per city). Keep a safety margin below the
# real limit so our local count and Atmospore's server-side count can't
# drift out of sync and trigger a surprise 429.
ATMOSPORE_DAILY_CALL_BUDGET = 90


#----------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Air quality data source strategy
# ---------------------------------------------------------------------------
AQ_SOURCE_MODE = "auto"    
RUN_AQI_CROSSCHECK = True  

# Which EPA PM2.5 AQI breakpoint table labeling.py should use.
#   "2024" -- EPA's 2024 revision (Good/Moderate boundary at 9.0 ug/m3)
#   "2012" -- the pre-2024 table (boundary at 12.0 ug/m3)
# The 2024 revision is current; "2012" exists only to reproduce results
# generated before this was fixed. See src/labeling.py for both tables.
PM25_BREAKPOINT_VERSION = "2024"

# Physical-plausibility ceilings (ug/m3) used by merge_clean.py to null out
# obvious sensor faults. These replaced IQR winsorizing, which both leaked
# test-set statistics into cleaning and clipped away exactly the extreme tail
# the Very Unhealthy / Hazardous classes depend on.
POLLUTANT_SANITY_CEILINGS = {"pm25": 2000.0, "pm10": 3000.0, "no2": 1000.0}
OPEN_METEO_AQ_HOURLY_VARS = ["pm2_5", "pm10", "nitrogen_dioxide", "us_aqi"]

WEATHER_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
]

POLLUTANTS = ["pm25", "pm10", "no2"]

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))




CITY_SLUG = CITY_NAME.strip().lower().replace(" ", "_")
TIMEZONE = city_timezone(CITY_NAME)

DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw", CITY_SLUG)
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed", CITY_SLUG)
MODELS_DIR = os.path.join(BASE_DIR, "models", CITY_SLUG)
# Cache + call-ledger live here rather than under a per-city folder, since
# the budget itself is shared across all cities.
ATMOSPORE_CACHE_DIR = os.path.join(BASE_DIR, "data", "processed", "_pollen_cache")


WEATHER_RAW_PATH = os.path.join(DATA_RAW_DIR, "weather_raw.csv")
AQ_RAW_PATH = os.path.join(DATA_RAW_DIR, "air_quality_raw.csv")
AQ_OPENMETEO_RAW_PATH = os.path.join(DATA_RAW_DIR, "air_quality_openmeteo_raw.csv")
AQ_CROSSCHECK_PATH = os.path.join(DATA_PROCESSED_DIR, "aqi_crosscheck_report.json")
MERGED_PATH = os.path.join(DATA_PROCESSED_DIR, "merged_clean.csv")
FEATURES_PATH = os.path.join(DATA_PROCESSED_DIR, "features.csv")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "feature_list.json")
METRICS_PATH = os.path.join(MODELS_DIR, "model_comparison.json")

RANDOM_STATE = 42

for _d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR,ATMOSPORE_CACHE_DIR):
    os.makedirs(_d, exist_ok=True)