"""
Explainability utilities: global feature importance and per-prediction SHAP
explanations for the best saved model.

Usage (standalone report):
    python -m src.explainability
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_artifacts():
    model = joblib.load(config.BEST_MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    le = joblib.load(config.LABEL_ENCODER_PATH)
    with open(config.FEATURE_LIST_PATH) as f:
        feature_cols = json.load(f)
    return model, scaler, le, feature_cols


def global_feature_importance(model, feature_cols, top_n=15):
    """Return a dataframe of top_n features by model-native importance.

    Works for tree models (RandomForest, XGBoost) via feature_importances_.
    For LogisticRegression, uses mean absolute coefficient across classes.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.mean(np.abs(model.coef_), axis=0)
    else:
        raise ValueError("Model type does not expose importances directly.")

    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    return imp_df


def shap_explain(model, X_background, X_instance, feature_cols, max_display=10):
    """Compute SHAP values for a single instance (or small batch).

    Uses TreeExplainer for tree models (fast, exact) and falls back to
    KernelExplainer for other model types (slower, sampled).
    """
    import shap

    model_type = type(model).__name__
    if model_type in ("RandomForestClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_instance)
    else:
        background = shap.sample(X_background, min(50, len(X_background)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_instance, nsamples=100)

    return shap_values


def main():
    model, scaler, le, feature_cols = load_artifacts()
    imp_df = global_feature_importance(model, feature_cols, top_n=15)
    print(f"Model: {type(model).__name__}")
    print("\nTop 15 features by global importance:")
    print(imp_df.to_string(index=False))


if __name__ == "__main__":
    main()
