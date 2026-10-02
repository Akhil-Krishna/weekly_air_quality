"""
Runs the full pipeline end to end, for one city or for all of them:

    1. Fetch weather history (Open-Meteo, in UTC)
    2. Fetch air quality history -- OpenAQ ground stations where available,
       automatic fallback to Open-Meteo Air Quality (model-based) where not,
       plus an AQI cross-check against Open-Meteo's own us_aqi field
    3. Merge + clean
    4. Feature engineering
    5. Train + compare models, save the best one

Usage:
    python run_pipeline.py                      # the city in config.DEFAULT_CITY
    python run_pipeline.py --city Delhi
    python run_pipeline.py --all                # every city in config.CITIES
    python run_pipeline.py --city Delhi --skip-fetch   # re-derive from raw CSVs

Each city runs in its own subprocess. config.py resolves the target city (and
all of its file paths) at import time, so one process cannot switch cities
halfway through without stale paths leaking across.
"""

import argparse
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Steps that re-read remote APIs, versus steps that only re-derive local files.
FETCH_STEPS = [
    ("Fetch weather (Open-Meteo, UTC)", "src.fetch_weather", "main"),
    ("Fetch air quality (OpenAQ, auto-fallback to Open-Meteo AQ + crosscheck)",
     "src.fetch_air_quality", "main"),
]
DERIVE_STEPS = [
    ("Merge & clean", "src.merge_clean", "merge_and_clean"),
    ("Feature engineering", "src.feature_engineering", "build_features"),
    ("Train & compare models", "src.train_models", "main"),
]


def run_steps(steps):
    """Run the named steps in THIS process (city already fixed by AQ_CITY)."""
    import importlib

    for name, module_path, func_name in steps:
        print("\n" + "=" * 70)
        print(f"STEP: {name}")
        print("=" * 70)
        start = time.time()
        getattr(importlib.import_module(module_path), func_name)()
        print(f"[{name}] done in {time.time() - start:.1f}s")


def run_city(city, skip_fetch):
    """Spawn a subprocess pinned to one city via the AQ_CITY env var."""
    env = dict(os.environ, AQ_CITY=city)
    # -u keeps the child unbuffered. Without it the OpenAQ fetch can run for
    # ten minutes with nothing on screen, which is indistinguishable from a
    # hang when the output is being piped to a file.
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--_worker"]
    if skip_fetch:
        cmd.append("--skip-fetch")
    print("\n" + "#" * 70)
    print(f"# CITY: {city}")
    print("#" * 70)
    return subprocess.call(cmd, cwd=BASE_DIR, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", help="City name as spelled in config.CITIES")
    parser.add_argument("--all", action="store_true",
                        help="Run every city in config.CITIES, in order")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip the two network steps and re-derive from the "
                             "existing raw CSVs. Note: the raw CSVs must have "
                             "been collected AFTER the UTC timezone fix, or the "
                             "merge will join two different clocks again.")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    steps = (DERIVE_STEPS if args.skip_fetch else FETCH_STEPS + DERIVE_STEPS)

    if args._worker:
        run_steps(steps)
        return 0

    sys.path.insert(0, BASE_DIR)

    if args.all:
        # Imported with no AQ_CITY set purely to read the city registry.
        import config
        cities = list(config.CITIES)
    elif args.city:
        cities = [args.city]
    else:
        import config
        cities = [config.CITY_NAME]

    failures = []
    for city in cities:
        if run_city(city, args.skip_fetch) != 0:
            failures.append(city)
            print(f"\n!! {city} failed -- continuing with the remaining cities.")

    print("\n" + "=" * 70)
    if failures:
        print(f"Pipeline finished with failures: {', '.join(failures)}")
        print(f"Succeeded: {', '.join(c for c in cities if c not in failures) or 'none'}")
    else:
        print(f"Pipeline complete for: {', '.join(cities)}")
    print("\nLaunch the app with:")
    print("  streamlit run app/streamlit_app.py")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
