

import sys
import os
import time
import argparse

sys.path.insert(0, ".")
import config

DEFAULT_CITIES = ["Wellington", "Christchurch", "Hamilton", "Dunedin"]


def run_step(name, fn):
    print(f"\n  -- {name} ...")
    start = time.time()
    fn()
    print(f"  -- {name} done in {time.time() - start:.1f}s")


def configure_for_city(city_name):
    """Point the shared config module at one city's coordinates and paths,
    the same pattern demo_week5.py uses for training -- extended here to
    cover the fetch/merge/feature stages too, not just training."""
    if city_name not in config.CITIES:
        raise ValueError(
            f"'{city_name}' is not in config.CITIES. Add its coordinates "
            f"there first. Known cities: {list(config.CITIES.keys())}"
        )

    lat = config.CITIES[city_name]["lat"]
    lon = config.CITIES[city_name]["lon"]
    city_slug = city_name.strip().lower().replace(" ", "_")

    config.CITY_NAME = city_name
    config.LATITUDE = lat
    config.LONGITUDE = lon
    config.CITY_SLUG = city_slug

    data_raw_dir = os.path.join(config.BASE_DIR, "data", "raw", city_slug)
    data_processed_dir = os.path.join(config.BASE_DIR, "data", "processed", city_slug)
    models_dir = os.path.join(config.BASE_DIR, "models", city_slug)
    for d in (data_raw_dir, data_processed_dir, models_dir):
        os.makedirs(d, exist_ok=True)

    config.DATA_RAW_DIR = data_raw_dir
    config.DATA_PROCESSED_DIR = data_processed_dir
    config.MODELS_DIR = models_dir

    config.WEATHER_RAW_PATH = os.path.join(data_raw_dir, "weather_raw.csv")
    config.AQ_RAW_PATH = os.path.join(data_raw_dir, "air_quality_raw.csv")
    config.AQ_OPENMETEO_RAW_PATH = os.path.join(data_raw_dir, "air_quality_openmeteo_raw.csv")
    config.AQ_CROSSCHECK_PATH = os.path.join(data_processed_dir, "aqi_crosscheck_report.json")
    config.MERGED_PATH = os.path.join(data_processed_dir, "merged_clean.csv")
    config.FEATURES_PATH = os.path.join(data_processed_dir, "features.csv")

    config.BEST_MODEL_PATH = os.path.join(models_dir, "best_model.joblib")
    config.LABEL_ENCODER_PATH = os.path.join(models_dir, "label_encoder.joblib")
    config.SCALER_PATH = os.path.join(models_dir, "scaler.joblib")
    config.FEATURE_LIST_PATH = os.path.join(models_dir, "feature_list.json")
    config.METRICS_PATH = os.path.join(models_dir, "model_comparison.json")


def run_pipeline_for_city(city_name):
    from src import fetch_weather, fetch_air_quality, merge_clean, feature_engineering, train_models

    print("\n" + "=" * 70)
    print(f"CITY: {city_name}")
    print("=" * 70)

    configure_for_city(city_name)

    run_step("Fetch weather (Open-Meteo)", fetch_weather.main)
    run_step("Fetch air quality (OpenAQ, auto-fallback to Open-Meteo AQ)", fetch_air_quality.main)
    run_step("Merge & clean", merge_clean.merge_and_clean)
    run_step("Feature engineering", feature_engineering.build_features)
    run_step("Train & compare models", train_models.main)


def main():
    parser = argparse.ArgumentParser(description="Run the full pipeline for multiple cities.")
    parser.add_argument("--cities", nargs="+", default=DEFAULT_CITIES,
                         help=f"City names from config.CITIES to process (default: {DEFAULT_CITIES})")
    args = parser.parse_args()

    if not config.OPENAQ_API_KEY:
        print("WARNING: OPENAQ_API_KEY is not set. OpenAQ requests will fail for any "
              "city that has a nearby ground station -- set it first:")
        print("  export OPENAQ_API_KEY='your-key-here'")
        print("Continuing anyway -- cities with no OpenAQ coverage will still work "
              "via the Open-Meteo fallback.\n")

    results = {}
    for city in args.cities:
        try:
            run_pipeline_for_city(city)
            results[city] = "OK"
        except Exception as e:
            print(f"\n!! {city} failed: {e}")
            print(f"!! Continuing with the next city.")
            results[city] = f"FAILED: {e}"

    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for city, status in results.items():
        print(f"  {city}: {status}")

    print("\nCheck each city's models/<city_slug>/model_comparison.json to see "
          "whether a model was actually trained, or whether that city's data "
          "turned out to be single-class like Auckland (in which case "
          "train_models.py explains why instead of producing a model).")
    print("\nAny successful cities are already in config.CITIES, so the app's "
          "city selector will pick them up automatically -- no further changes needed.")
    print("\nLaunch the app with:")
    print("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
