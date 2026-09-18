"""
Runs the full pipeline end to end:
    1. Fetch weather history (Open-Meteo)
    2. Fetch air quality history -- OpenAQ ground stations where available,
       automatic fallback to Open-Meteo Air Quality (model-based) where not,
       plus an AQI cross-check against Open-Meteo's own us_aqi field
    3. Merge + clean
    4. Feature engineering
    5. Train + compare models, save the best one

Usage:
    python run_pipeline.py
"""

import sys
import time


def run_step(name, fn):
    print("\n" + "=" * 70)
    print(f"STEP: {name}")
    print("=" * 70)
    start = time.time()
    fn()
    print(f"[{name}] done in {time.time() - start:.1f}s")


def main():
    from src import fetch_weather, fetch_air_quality, merge_clean, feature_engineering, train_models

    run_step("Fetch weather (Open-Meteo)", fetch_weather.main)
    run_step("Fetch air quality (OpenAQ, auto-fallback to Open-Meteo AQ + crosscheck)",
              fetch_air_quality.main)
    run_step("Merge & clean", merge_clean.merge_and_clean)
    run_step("Feature engineering", feature_engineering.build_features)
    run_step("Train & compare models", train_models.main)

    print("\nPipeline complete. Launch the app with:")
    print("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
