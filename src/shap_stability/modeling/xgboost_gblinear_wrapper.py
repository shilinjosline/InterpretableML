"""Thin XGBoost gblinear (linear booster) training wrapper.

Leakage-safe by design: trains ONLY on the X/y passed in.
Caller must pass train-fold data (not full dataset).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# Reuse the same result container as your tree wrapper
from .xgboost_wrapper import XGBTrainResult


_TREE_ONLY_KEYS = {
    "max_depth",
    "min_child_weight",
    "gamma",
    "subsample",
    "colsample_bytree",
    "colsample_bylevel",
    "colsample_bynode",
    "tree_method",
    "max_leaves",
    "max_bin",
    "grow_policy",
}


def train_xgb_gblinear_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    params: dict[str, Any] | None = None,
    random_state: int | None = None,
) -> XGBTrainResult:
    """Train XGBoost with booster='gblinear' (logistic regression + regularization)."""
    X_frame = pd.DataFrame(X).reset_index(drop=True)
    X_frame.columns = [str(c) for c in X_frame.columns]

    y_series = pd.Series(y).reset_index(drop=True)
    if len(X_frame) != len(y_series):
        raise ValueError("X and y must have the same length")
    if not set(pd.unique(y_series)).issubset({0, 1}):
        raise ValueError("y must be binary with values 0/1")

    base_params: dict[str, Any] = {
        "booster": "gblinear",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 200,
        "learning_rate": 0.1,
        "reg_lambda": 1.0,  # L2
        "reg_alpha": 0.0,   # L1
        "n_jobs": -1,
        "random_state": random_state,
        "verbosity": 0,
    }

    if params:
        base_params.update(params)

    # Enforce gblinear and drop tree-only keys defensively
    base_params["booster"] = "gblinear"
    for k in list(base_params.keys()):
        if k in _TREE_ONLY_KEYS:
            base_params.pop(k, None)

    model = XGBClassifier(**base_params)
    model.fit(X_frame, y_series)

    return XGBTrainResult(model=model, feature_names=list(X_frame.columns))


def predict_proba(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Predict positive-class probabilities."""
    X_frame = pd.DataFrame(X)
    X_frame.columns = [str(c) for c in X_frame.columns]
    proba = model.predict_proba(X_frame)
    return proba[:, 1]