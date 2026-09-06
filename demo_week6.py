

import sys
sys.path.insert(0, ".")
import json
import joblib
import config
from src.explainability import global_feature_importance


def show_explainability():
    print("=" * 70)
    print("PART 1: Explainability -- why does the Delhi model predict what it does?")
    print("=" * 70)
    try:
        model = joblib.load("models/delhi/best_model.joblib")
        with open("models/delhi/feature_list.json") as f:
            feature_cols = json.load(f)
    except FileNotFoundError:
        print("No saved Delhi model found -- run demo_week5.py first.")
        return
    except ModuleNotFoundError as e:
        print(f"Could not load the saved model: {e}")
        print("This model was saved using a library (e.g. xgboost) that isn't "
              "installed in this environment. Run: pip install -r requirements.txt")
        return

    imp = global_feature_importance(model, feature_cols, top_n=10)
    print(f"\nModel: {type(model).__name__}")
    print("\nTop 10 features driving this model's predictions:")
    print(imp.to_string(index=False))
    print("\nInterpretation: the model leans heavily on pollution's own recent "
          "trend (rolling averages / lag features), not just raw weather at a "
          "single instant -- which matches how pollution actually behaves "
          "(it builds up and disperses gradually, not instantly).")


def show_hybrid_source_and_crosscheck():
    print("\n" + "=" * 70)
    print("PART 2: Hybrid data source + AQI cross-check (requires internet)")
    print("=" * 70)
    print(f"AQ_SOURCE_MODE is now: '{config.AQ_SOURCE_MODE}' "
          "(was 'openaq'")
    print("This means: try OpenAQ first, and if no station is nearby, "
          "automatically fall back to Open-Meteo Air Quality (model-based, "
          "works for any coordinate) instead of failing.")

    try:
        from src.validate_aqi_crosscheck import run_crosscheck
        run_crosscheck()
    except Exception as e:
        print(f"\nCross-check could not run here: {e}")
        print("(Expected if there's no internet connection available right now "
              "-- run this on a machine with network access before presenting.)")


if __name__ == "__main__":
    show_explainability()
    show_hybrid_source_and_crosscheck()
