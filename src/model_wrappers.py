"""
Model wrappers that adapt a library model to this project's label space.

Lives in its own module (rather than inside train_models.py) because joblib
pickles by import path: the Streamlit app has to be able to import the class
to load a saved model, without pulling in the training script.
"""

import numpy as np


class ContiguousLabelClassifier:
    """Wraps a classifier that requires class labels 0..k-1 with no holes.

    XGBoost >= 1.7 rejects a non-contiguous label set with "Invalid classes
    inferred from unique values of y". That is reachable here: the label
    encoder is deliberately fit over all six risk categories in severity order
    so codes mean the same thing in every city and every report, and a rare
    category can easily fall entirely inside the test period -- leaving
    y_train as something like {0, 1, 2, 4, 5} and crashing the fit.

    This maps the training labels down to a dense range before fitting, and
    maps predictions back up afterwards, so callers and the saved artifact keep
    working in the project's canonical label space.
    """

    def __init__(self, model):
        self.model = model
        self.train_classes_ = None

    @property
    def inner_model(self):
        """The fitted library model, for tools that introspect it (e.g. SHAP)."""
        return self.model

    def fit(self, X, y, **kwargs):
        y = np.asarray(y)
        self.train_classes_ = np.unique(y)
        dense = {c: i for i, c in enumerate(self.train_classes_)}
        y_dense = np.array([dense[v] for v in y], dtype=int)
        self.model.fit(X, y_dense, **kwargs)
        return self

    def predict(self, X):
        dense_preds = np.asarray(self.model.predict(X), dtype=int)
        return self.train_classes_[dense_preds]

    def predict_proba(self, X):
        """Probabilities widened back out to the canonical label space.

        Classes the model never saw in training get probability 0 rather than
        being missing from the array, so column i always means class i.
        """
        dense = self.model.predict_proba(X)
        n_classes = int(self.train_classes_.max()) + 1
        full = np.zeros((dense.shape[0], n_classes), dtype=float)
        full[:, self.train_classes_] = dense
        return full

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

    @property
    def classes_(self):
        return self.train_classes_

    def __repr__(self):
        return f"ContiguousLabelClassifier({type(self.model).__name__})"
