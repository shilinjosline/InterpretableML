"""Compute SHAP values and global importance summaries.

- Tree boosters (gbtree/dart): SHAP TreeExplainer (TreeSHAP)
- Linear booster (gblinear): Linear SHAP using E[X] from a background dataset:
    phi_j = w_j * (x_j - E[x_j])
    base  = b + sum_j w_j * E[x_j]

Leakage note:
- Pass background from the TRAINING fold only (e.g., X_train_enc).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier


@dataclass(frozen=True)
class ShapResult:
    values: np.ndarray
    base_values: np.ndarray
    feature_names: list[str]
    global_importance: pd.Series


def _linear_shap_gblinear(
    model: XGBClassifier,
    X: pd.DataFrame,
    *,
    background: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """Linear SHAP for gblinear using mean background E[X]."""
    X_frame = pd.DataFrame(X).reset_index(drop=True)
    X_frame.columns = [str(c) for c in X_frame.columns]

    bg = pd.DataFrame(background).reset_index(drop=True)
    bg = bg[X_frame.columns]
    bg.columns = [str(c) for c in bg.columns]

    if not hasattr(model, "coef_"):
        raise ValueError("gblinear model missing coef_; cannot compute linear SHAP safely.")

    coef = np.asarray(model.coef_).reshape(-1)
    if coef.shape[0] != X_frame.shape[1]:
        raise ValueError(f"coef_ has shape {coef.shape}, expected ({X_frame.shape[1]},)")

    intercept = 0.0
    if hasattr(model, "intercept_"):
        inter = np.asarray(model.intercept_).reshape(-1)
        if inter.size:
            intercept = float(inter[0])

    mu = bg.mean(axis=0).to_numpy(dtype=float)           # E[X]
    X_np = X_frame.to_numpy(dtype=float)

    values = (X_np - mu) * coef                           # w * (x - E[x])
    base_value = float(intercept + float(np.dot(mu, coef)))  # b + w·E[x]
    return values, base_value


def _tree_shap(
    model: XGBClassifier,
    X: pd.DataFrame,
    *,
    background: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """TreeSHAP using shap.TreeExplainer."""
    X_frame = pd.DataFrame(X).reset_index(drop=True)
    bg = pd.DataFrame(background).reset_index(drop=True)

    explainer = shap.TreeExplainer(model, data=bg, model_output="raw")
    shap_values = explainer.shap_values(X_frame)
    base_value = explainer.expected_value

    # Handle binary classifier outputs across SHAP versions
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_values = shap_values[1]          # positive class
            base_value = base_value[1] if isinstance(base_value, (list, np.ndarray)) else base_value
        elif len(shap_values) == 1:
            shap_values = shap_values[0]
            base_value = base_value[0] if isinstance(base_value, (list, np.ndarray)) else base_value
        else:
            raise ValueError(f"Unexpected SHAP list length: {len(shap_values)}")

    values = np.asarray(shap_values)
    base_value_scalar = float(np.asarray(base_value).reshape(-1)[0])
    return values, base_value_scalar


def compute_tree_shap(
    model: XGBClassifier,
    X: pd.DataFrame,
    *,
    background: Optional[pd.DataFrame] = None,
) -> ShapResult:
    """Compute SHAP values and mean(|SHAP|) importance.

    IMPORTANT: to avoid leakage, pass background from training fold (encoded).
    """
    X_frame = pd.DataFrame(X).reset_index(drop=True)
    feature_names = [str(c) for c in X_frame.columns]

    if background is None:
        # Keep backward-compatibility, but this can leak if X is test.
        background = X_frame

    booster = model.get_params().get("booster", None)

    if booster == "gblinear":
        values, base_value = _linear_shap_gblinear(model, X_frame, background=background)
    else:
        values, base_value = _tree_shap(model, X_frame, background=background)

    if values.ndim != 2:
        raise ValueError("SHAP values must be 2D (n_samples, n_features)")

    global_importance = pd.Series(
        np.mean(np.abs(values), axis=0),
        index=feature_names,
        name="mean_abs_shap",
    )

    return ShapResult(
        values=values,
        base_values=np.asarray(base_value),
        feature_names=feature_names,
        global_importance=global_importance,
    )