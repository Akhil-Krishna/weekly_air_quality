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


def explain_single_prediction(model, X_instance_scaled, feature_cols, predicted_class_idx, top_n=10):
    """
    Compute a SHAP explanation for ONE prediction (one row), for the specific
    class the model predicted, and return a clean, consistent DataFrame --
    regardless of which shape the installed SHAP version happens to return.

    SHAP's return shape for multi-class tree models has genuinely changed
    across library versions:
      - older versions: a list of (n_instances, n_features) arrays, one per class
      - newer versions: a single (n_instances, n_features, n_classes) array
      - binary/regression: a single (n_instances, n_features) array, no class axis
    This function normalizes all three into one flat per-feature result so
    callers (the app) never need to know which shape they got.

    Returns a DataFrame with columns: feature, shap_value, abs_shap_value,
    sorted by impact (most influential first).

    Raises RuntimeError with a clear message if the model type isn't
    supported or the SHAP output shape is unrecognized -- callers should
    catch this and show it to the user rather than let the app crash.
    """
    import shap

    model_type = type(model).__name__
    if model_type not in ("RandomForestClassifier", "XGBClassifier", "GradientBoostingClassifier"):
        raise RuntimeError(
            f"Per-prediction SHAP explanation isn't supported for model type "
            f"'{model_type}' -- only tree-based models (Random Forest, XGBoost) "
            "are handled here."
        )

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_instance_scaled)

    if isinstance(raw, list):
        # Older SHAP: list of per-class arrays, each (n_instances, n_features)
        if predicted_class_idx >= len(raw):
            raise RuntimeError(
                f"SHAP returned {len(raw)} class arrays but predicted class "
                f"index was {predicted_class_idx} -- mismatch with the model's "
                "known classes."
            )
        row_values = np.asarray(raw[predicted_class_idx])[0]
    else:
        arr = np.asarray(raw)
        if arr.ndim == 3:
            # Newer SHAP: (n_instances, n_features, n_classes)
            if predicted_class_idx >= arr.shape[2]:
                raise RuntimeError(
                    f"SHAP output has {arr.shape[2]} classes but predicted "
                    f"class index was {predicted_class_idx} -- mismatch with "
                    "the model's known classes."
                )
            row_values = arr[0, :, predicted_class_idx]
        elif arr.ndim == 2:
            # Binary/regression: (n_instances, n_features), no class axis
            row_values = arr[0]
        else:
            raise RuntimeError(f"Unexpected SHAP output shape: {arr.shape}")

    if len(row_values) != len(feature_cols):
        raise RuntimeError(
            f"SHAP returned {len(row_values)} values but there are "
            f"{len(feature_cols)} features -- something doesn't line up."
        )

    result = pd.DataFrame({"feature": feature_cols, "shap_value": row_values})
    result["abs_shap_value"] = result["shap_value"].abs()
    result = result.sort_values("abs_shap_value", ascending=False).head(top_n).reset_index(drop=True)
    return result


def main():
    model, scaler, le, feature_cols = load_artifacts()
    imp_df = global_feature_importance(model, feature_cols, top_n=15)
    print(f"Model: {type(model).__name__}")
    print("\nTop 15 features by global importance:")
    print(imp_df.to_string(index=False))


if __name__ == "__main__":
    main()
