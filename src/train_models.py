"""
Train and compare Logistic Regression, Random Forest, and XGBoost on the
chronologically-split dataset. Selects the best model by macro F1-score on
the held-out (most recent) test period, and saves the winner + supporting
artifacts (scaler, label encoder, feature list) for the Streamlit app.

Usage:
    python -m src.train_models
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


NON_FEATURE_COLS = {
    "datetime", "risk_category", "aqi", "pm25", "pm10", "no2", "hour", "day_of_week", "month",
}


def load_features():
    df = pd.read_csv(config.FEATURES_PATH, parse_dates=["datetime"])
    return df


def chronological_split(df):
    """Chronological split: train on the earlier period, test on the most
    recent period.
    """
    df = df.sort_values("datetime").reset_index(drop=True)
    data_min, data_max = df["datetime"].min(), df["datetime"].max()

    def _split_at(split_date):
        return df[df["datetime"] < split_date].copy(), df[df["datetime"] >= split_date].copy()

    split_date = pd.Timestamp(config.SPLIT_DATE)
    train, test = _split_at(split_date)

    if len(train) == 0 or len(test) == 0:
        span_days = max((data_max - data_min).days, 1)
        test_days = max(1, min(config.TEST_MONTHS * 30, int(span_days * 0.25)))
        fallback_split = data_max - pd.Timedelta(days=test_days)
        print(f"NOTE: global config.SPLIT_DATE ({config.SPLIT_DATE}) falls outside "
              f"{config.CITY_NAME}'s own data range ({data_min.date()} to "
              f"{data_max.date()}), which would leave an empty train or test set. "
              f"Falling back to a split relative to {config.CITY_NAME}'s own data: "
              f"last {test_days} days as test (split at {fallback_split.date()}).")
        split_date = fallback_split
        train, test = _split_at(split_date)

    if len(test) == 0 or len(train) == 0:
        # Last resort for very short collection windows: split by row count
        # instead of by date.
        n_test = max(1, int(len(df) * 0.2))
        train, test = df.iloc[:-n_test].copy(), df.iloc[-n_test:].copy()
        print(f"NOTE: still an empty train/test set after the data-relative "
              f"fallback -- using a plain 80/20 row-count split instead "
              f"({len(train)} train / {len(test)} test).")

    print(f"Train: {len(train)} rows ({train['datetime'].min()} to {train['datetime'].max()})")
    print(f"Test:  {len(test)} rows ({test['datetime'].min()} to {test['datetime'].max()})")
    return train, test


def get_feature_columns(df):
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    # keep only numeric columns
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return cols


def report_class_balance(train, test):
    print("\n--- Class balance: TRAIN ---")
    print(train["risk_category"].value_counts(normalize=True).round(3))
    print("\n--- Class balance: TEST ---")
    if len(test) > 0:
        print(test["risk_category"].value_counts(normalize=True).round(3))
    else:
        print("(empty)")


def train_and_evaluate(X_train, y_train, X_test, y_test, class_names):
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            random_state=config.RANDOM_STATE, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
            random_state=config.RANDOM_STATE, n_jobs=-1,
        ),
    }

    results = {}
    fitted_models = {}
    all_labels = list(range(len(class_names)))  # explicit, so metrics stay consistent
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        f1_macro = f1_score(y_test, preds, average="macro", labels=all_labels, zero_division=0)
        precision_macro = precision_score(y_test, preds, average="macro", labels=all_labels, zero_division=0)
        recall_macro = recall_score(y_test, preds, average="macro", labels=all_labels, zero_division=0)
        cm = confusion_matrix(y_test, preds, labels=all_labels).tolist()
        report = classification_report(
            y_test, preds, labels=all_labels, target_names=class_names,
            zero_division=0, output_dict=True,
        )

        results[name] = {
            "f1_macro": f1_macro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "confusion_matrix": cm,
            "classification_report": report,
        }
        fitted_models[name] = model
        print(f"  {name}: F1(macro)={f1_macro:.3f}  Precision(macro)={precision_macro:.3f}  "
              f"Recall(macro)={recall_macro:.3f}")

    return results, fitted_models


def main():
    df = load_features()
    train_df, test_df = chronological_split(df)
    report_class_balance(train_df, test_df)

    n_classes = df["risk_category"].nunique()
    if n_classes < 2:
        only_class = df["risk_category"].unique().tolist()
        print(f"\n{'='*70}")
        print(f"NOTE: {config.CITY_NAME}'s entire dataset contains only ONE risk "
              f"category ({only_class}) across the whole collection window.")
        print("Classification models require at least 2 classes to train "
              "meaningfully -- there's nothing to distinguish, so model "
              "training is skipped for this city rather than fit a model "
              "that trivially always predicts the same class.")
        print(f"\nThis is itself a real, reportable finding: {config.CITY_NAME}'s "
              "air quality stayed within a single risk band for the entire "
              "period, in contrast to a high-variance city like Delhi. It's "
              "direct evidence for why a two-city comparison is worthwhile.")
        print(f"{'='*70}")
        return

    # Also guard against a train or test split that -- after all fallbacks --
    # still ends up single-class only within that split (can happen with a
    # very rare second class near the edge of the data).
    if train_df["risk_category"].nunique() < 2:
        print(f"\nNOTE: {config.CITY_NAME}'s TRAIN split contains only one risk "
              "category even though the full dataset has more than one -- the "
              "rare class(es) fall entirely in the test period. Model training "
              "is skipped for this split rather than fit a model that can't "
              "learn to distinguish classes it never saw.")
        return

    feature_cols = get_feature_columns(df)
    print(f"\nUsing {len(feature_cols)} features.")

    le = LabelEncoder()
    le.fit(df["risk_category"])
    y_train = le.transform(train_df["risk_category"])
    y_test = le.transform(test_df["risk_category"])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    results, fitted_models = train_and_evaluate(
        X_train, y_train, X_test, y_test, class_names=list(le.classes_)
    )

    best_name = max(results, key=lambda n: results[n]["f1_macro"])
    best_model = fitted_models[best_name]
    print(f"\n>>> Best model by macro F1: {best_name} "
          f"(F1={results[best_name]['f1_macro']:.3f})")

    # Comparison table
    print("\n=== Model Comparison Table ===")
    print(f"{'Model':<20}{'F1 (macro)':<12}{'Precision':<12}{'Recall':<12}")
    for name, r in results.items():
        marker = " <-- BEST" if name == best_name else ""
        print(f"{name:<20}{r['f1_macro']:<12.3f}{r['precision_macro']:<12.3f}"
              f"{r['recall_macro']:<12.3f}{marker}")

    # Save artifacts
    joblib.dump(best_model, config.BEST_MODEL_PATH)
    joblib.dump(le, config.LABEL_ENCODER_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    with open(config.FEATURE_LIST_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(config.METRICS_PATH, "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2, default=str)

    print(f"\nSaved best model ({best_name}) to {config.BEST_MODEL_PATH}")
    print(f"Saved scaler to {config.SCALER_PATH}")
    print(f"Saved label encoder to {config.LABEL_ENCODER_PATH}")
    print(f"Saved feature list to {config.FEATURE_LIST_PATH}")
    print(f"Saved full metrics to {config.METRICS_PATH}")


if __name__ == "__main__":
    main()