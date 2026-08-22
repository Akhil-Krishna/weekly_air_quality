"""
Week 4 demo script -- run this live to show feature engineering working on
the real, merged & cleaned Delhi dataset from week 3.

Usage:
    python demo_week4.py
"""

import sys
sys.path.insert(0, ".")
import pandas as pd
import config
from src.feature_engineering import build_features

print("=== BEFORE: merged_clean.csv (week 3's output) ===")
before = pd.read_csv(config.MERGED_PATH)
print(f"Rows: {len(before)}  |  Columns: {before.shape[1]}")
print(f"Column names: {list(before.columns)}")

print("\n=== RUNNING feature_engineering.py ===")
after = build_features()

print("\n=== AFTER: features.csv ===")
print(f"Rows: {len(after)}  |  Columns: {after.shape[1]}")

new_cols = [c for c in after.columns if c not in before.columns]
print(f"\n{len(new_cols)} new engineered feature columns added, e.g.:")
sample_new = [c for c in new_cols if "roll_24h" in c or "lag_24h" in c or c in
              ("hour_sin", "hour_cos", "is_weekend")][:8]
for c in sample_new:
    print(f"  - {c}")

print("\n=== Sample rows showing a few engineered features ===")
show_cols = ["datetime", "risk_category", "hour", "is_weekend",
             "pm25_roll_24h_mean", "pm25_lag_24h"]
show_cols = [c for c in show_cols if c in after.columns]
print(after[show_cols].head(5).to_string(index=False))

print(f"\nNote: {len(before)} -> {len(after)} rows -- the drop is expected: "
      f"rolling/lag features need prior history, so the first ~7 days of the "
      f"series (before a full week of lookback exists) are dropped.")
print(f"\nDone -- features.csv saved to data/processed/{config.CITY_SLUG}/")
