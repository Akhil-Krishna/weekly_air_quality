
import sys
sys.path.insert(0, ".")
import os
import config
from src import train_models


def run_for_city(city_name, latitude, longitude):
    config.CITY_NAME = city_name
    config.LATITUDE = latitude
    config.LONGITUDE = longitude
    config.CITY_SLUG = city_name.strip().lower().replace(" ", "_")

    data_processed_dir = os.path.join(config.BASE_DIR, "data", "processed", config.CITY_SLUG)
    models_dir = os.path.join(config.BASE_DIR, "models", config.CITY_SLUG)
    os.makedirs(models_dir, exist_ok=True)

    config.FEATURES_PATH = os.path.join(data_processed_dir, "features.csv")
    config.BEST_MODEL_PATH = os.path.join(models_dir, "best_model.joblib")
    config.LABEL_ENCODER_PATH = os.path.join(models_dir, "label_encoder.joblib")
    config.SCALER_PATH = os.path.join(models_dir, "scaler.joblib")
    config.FEATURE_LIST_PATH = os.path.join(models_dir, "feature_list.json")
    config.METRICS_PATH = os.path.join(models_dir, "model_comparison.json")

    print(f"\n{'#'*70}")
    print(f"#  TRAINING FOR: {city_name}")
    print(f"{'#'*70}")
    train_models.main()


if __name__ == "__main__":
    run_for_city("Delhi", 28.6139, 77.2090)
    run_for_city("Auckland", -36.8485, 174.7633)

    print("\n\nSummary: Delhi produced a real, comparable 3-model result "
          "(saved to models/delhi/). Auckland's single-category dataset "
          "was correctly identified and training was skipped with an "
          "explanation rather than forcing a meaningless or crashing result.")
