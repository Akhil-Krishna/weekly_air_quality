"""
Train and compare Logistic Regression, Random Forest, and XGBoost on the
chronologically-split dataset. Selects the best model by macro F1-score on
the held-out (most recent) test period, and saves the winner + supporting
artifacts (scaler, label encoder, feature list) for the Streamlit app.

Two reporting details worth knowing before reading the numbers:

  * The split point is derived from the DATA, not from date.today(), and is
    written into model_comparison.json. The old wall-clock split silently
    shrank the "last 3 months" test window to 19 days as the data aged, and
    the app displayed a freshly-recomputed split date that did not match the
    one the saved model had been trained with.

  * f1_macro averages only over classes that are actually present in the test
    set. Averaging over all six when one has zero test rows folds a guaranteed
    0.0 into the mean, which understated every model by ~0.04. The all-six
    figure is still reported as f1_macro_all_labels for continuity.

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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.labeling import OrderedLabelEncoder
from src.model_wrappers import ContiguousLabelClassifier
from src.utils import derive_split_date


# Columns excluded from the feature set. The raw pollutants and aqi are the
# label's own ingredients; datetime_local is a display column; hour /
# day_of_week / month are superseded by their cyclical encodings.
#
# Note that excluding pm25/pm10/no2 here is NOT sufficient on its own to
# prevent leakage -- rolling averages OF those columns have to exclude the
# current hour too, which is handled in feature_engineering.py.
NON_FEATURE_COLS = {
    "datetime", "datetime_local", "risk_category", "aqi",
    "pm25", "pm10", "no2", "hour", "day_of_week", "month",
}


def load_features():
    df = pd.read_csv(config.FEATURES_PATH, parse_dates=["datetime"])
    return df


def chronological_split(df):
    """Split on time, with the boundary derived from the data itself.

    Returns (train, test, split_date). The caller persists split_date so the
    app can report the window the model was really evaluated on.
    """
    df = df.sort_values("datetime").reset_index(drop=True)
    data_min, data_max = df["datetime"].min(), df["datetime"].max()

    split_date = derive_split_date(df["datetime"], test_months=config.TEST_MONTHS)
    train = df[df["datetime"] < split_date].copy()
    test = df[df["datetime"] >= split_date].copy()

    if len(train) == 0 or len(test) == 0:
        # Only possible for a dataset too short or too clustered for a time
        # split to separate at all.
        n_test = max(1, int(len(df) * 0.2))
        train, test = df.iloc[:-n_test].copy(), df.iloc[-n_test:].copy()
        split_date = test["datetime"].min()
        print(f"NOTE: a time-based split of {config.CITY_NAME} left one side "
              f"empty (data spans {data_min} to {data_max}); fell back to an "
              f"80/20 row-count split ({len(train)} train / {len(test)} test).")

    span = (test["datetime"].max() - test["datetime"].min())
    print(f"Split at: {split_date}  (derived from the data, not the clock)")
    print(f"Train: {len(train)} rows ({train['datetime'].min()} to {train['datetime'].max()})")
    print(f"Test:  {len(test)} rows ({test['datetime'].min()} to {test['datetime'].max()}"
          f" -- {span.total_seconds() / 86400:.1f} days)")
    return train, test, split_date


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


def build_models():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            random_state=config.RANDOM_STATE, n_jobs=-1,
        ),
        # Wrapped so a risk category missing from the train split cannot crash
        # the fit -- see src/model_wrappers.py.
        "XGBoost": ContiguousLabelClassifier(XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
            random_state=config.RANDOM_STATE, n_jobs=-1,
        )),
    }


def train_and_evaluate(X_train, y_train, X_test, y_test, class_names):
    all_labels = list(range(len(class_names)))
    # Classes with at least one row in the test set. Macro-averaging over a
    # class with zero support contributes a guaranteed 0.0 to the mean, which
    # is a property of the split, not of the model.
    present_labels = sorted(set(np.unique(y_test).tolist()))
    absent = [class_names[i] for i in all_labels if i not in present_labels]
    if absent:
        print(f"\nNOTE: {len(absent)} risk categor(y/ies) have no rows in the "
              f"test period: {absent}.")
        print("      Headline macro scores average over the "
              f"{len(present_labels)} class(es) that are present; the all-six "
              "figure is still reported as f1_macro_all_labels.")

    results = {}
    fitted_models = {}
    for name, model in build_models().items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        def _scores(labels):
            return {
                "f1": f1_score(y_test, preds, average="macro", labels=labels, zero_division=0),
                "precision": precision_score(y_test, preds, average="macro", labels=labels, zero_division=0),
                "recall": recall_score(y_test, preds, average="macro", labels=labels, zero_division=0),
            }

        present = _scores(present_labels)
        overall = _scores(all_labels)

        results[name] = {
            "f1_macro": present["f1"],
            "precision_macro": present["precision"],
            "recall_macro": present["recall"],
            "f1_macro_all_labels": overall["f1"],
            "precision_macro_all_labels": overall["precision"],
            "recall_macro_all_labels": overall["recall"],
            "accuracy": float((preds == y_test).mean()),
            "n_test_classes_present": len(present_labels),
            # Confusion matrix rows/cols follow class_names, which is in AQI
            # severity order (Good -> Hazardous), not alphabetical.
            "confusion_matrix": confusion_matrix(y_test, preds, labels=all_labels).tolist(),
            "classification_report": classification_report(
                y_test, preds, labels=all_labels, target_names=class_names,
                zero_division=0, output_dict=True,
            ),
        }
        fitted_models[name] = model
        print(f"  {name}: F1(macro, {len(present_labels)} classes present)="
              f"{present['f1']:.3f}  Precision={present['precision']:.3f}  "
              f"Recall={present['recall']:.3f}  Accuracy={results[name]['accuracy']:.3f}")
        print(f"            F1(macro over all {len(all_labels)} labels)={overall['f1']:.3f}")

    return results, fitted_models


def main():
    df = load_features()
    train_df, test_df, split_date = chronological_split(df)
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

    if train_df["risk_category"].nunique() < 2:
        print(f"\nNOTE: {config.CITY_NAME}'s TRAIN split contains only one risk "
              "category even though the full dataset has more than one -- the "
              "rare class(es) fall entirely in the test period. Model training "
              "is skipped for this split rather than fit a model that can't "
              "learn to distinguish classes it never saw.")
        return

    feature_cols = get_feature_columns(df)
    print(f"\nUsing {len(feature_cols)} features.")

    # Severity-ordered codes (Good=0 ... Hazardous=5), stable across cities.
    le = OrderedLabelEncoder().fit(df["risk_category"])
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

    print("\n=== Model Comparison Table ===")
    print(f"{'Model':<20}{'F1 (macro)':<12}{'Precision':<12}{'Recall':<12}{'Accuracy':<10}")
    for name, r in results.items():
        marker = " <-- BEST" if name == best_name else ""
        print(f"{name:<20}{r['f1_macro']:<12.3f}{r['precision_macro']:<12.3f}"
              f"{r['recall_macro']:<12.3f}{r['accuracy']:<10.3f}{marker}")

    joblib.dump(best_model, config.BEST_MODEL_PATH)
    joblib.dump(le, config.LABEL_ENCODER_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    with open(config.FEATURE_LIST_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Everything the app needs to describe this model honestly, including the
    # split it was actually trained with.
    metadata = {
        "best_model": best_name,
        "city": config.CITY_NAME,
        "timezone": config.city_timezone(config.CITY_NAME),
        "pm25_breakpoint_version": config.PM25_BREAKPOINT_VERSION,
        "split_date": str(pd.Timestamp(split_date)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_start": str(train_df["datetime"].min()),
        "train_end": str(train_df["datetime"].max()),
        "test_start": str(test_df["datetime"].min()),
        "test_end": str(test_df["datetime"].max()),
        "class_order": list(le.classes_),
        "n_features": len(feature_cols),
        "results": results,
    }
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nSaved best model ({best_name}) to {config.BEST_MODEL_PATH}")
    print(f"Saved scaler to {config.SCALER_PATH}")
    print(f"Saved label encoder to {config.LABEL_ENCODER_PATH}")
    print(f"Saved feature list to {config.FEATURE_LIST_PATH}")
    print(f"Saved full metrics to {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
