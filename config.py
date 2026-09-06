"""
Central configuration for the Air Quality Risk Predictor.

Change CITY_NAME / LATITUDE / LONGITUDE to target a different city.
Everything downstream (data collection, training, app) reads from here.
"""

import os
from datetime import date, timedelta
from dotenv import load_dotenv


load_dotenv()

# ---------------------------------------------------------------------------
# CITY (default assumption: Delhi, India -- change these 3 lines for another city)
# ---------------------------------------------------------------------------
CITY_NAME = "Delhi"
LATITUDE = 28.6139
LONGITUDE = 77.2090
OPENAQ_RADIUS_METERS = 25000  # search radius around the coordinates for OpenAQ stations
OPENAQ_MAX_LOCATIONS = 6      # only pull the N nearest stations -- pulling all ~100 in a
                              # dense city like Delhi triggers rate limits for no real benefit
OPENAQ_REQUEST_DELAY_SECONDS = 1.2  # pause between requests to stay under the rate limit

# ---------------------------------------------------------------------------
# DATE RANGE for historical data collection
# Open-Meteo archive data typically has a ~5 day lag before it's finalized,
# so we end the historical window a week before "today".
# ---------------------------------------------------------------------------
HISTORY_MONTHS = 9  # 9 months train + we'll reserve the tail for testing
_today = date.today()
END_DATE = _today - timedelta(days=7)
START_DATE = END_DATE - timedelta(days=HISTORY_MONTHS * 30 + 90)  # ~9-12 months total

# Chronological split point: last 3 months = test set
TEST_MONTHS = 3
SPLIT_DATE = END_DATE - timedelta(days=TEST_MONTHS * 30)

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPENAQ_BASE_URL = "https://api.openaq.org/v3"

OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY", "")

# ---------------------------------------------------------------------------
# Air quality data source strategy
# ---------------------------------------------------------------------------
AQ_SOURCE_MODE = "auto"    
RUN_AQI_CROSSCHECK = True  
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

DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw", CITY_SLUG)
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed", CITY_SLUG)
MODELS_DIR = os.path.join(BASE_DIR, "models", CITY_SLUG)

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

for _d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR):
    os.makedirs(_d, exist_ok=True)