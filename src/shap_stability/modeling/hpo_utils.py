"""Inner CV hyperparameter search utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from ..metrics.metrics_utils import MetricsConfig, MetricsConfigError, score_metrics
from .xgboost_wrapper import predict_proba as _default_predict_proba
from .xgboost_wrapper import train_xgb_classifier as _default_train_xgb

from shap_stability.data_augmentation import NoisyCopyConfig, add_noisy_copies_train_test
from shap_stability.data import one_hot_encode_train_test

# Train function must accept (X, y, params=..., random_state=...) and return an object
# with a `.model` attribute (your XGBTrainResult fits this).
TrainFn = Callable[..., Any]
PredictProbaFn = Callable[[Any, pd.DataFrame], np.ndarray]


@dataclass(frozen=True)
class HPOResult:
    best_params: dict[str, Any]
    best_score: float
    model: object


def _metric_greater_is_better(metric_name: str) -> bool:
    # Keep your existing behavior
    return metric_name != "log_loss"


def _validate_grid(param_grid: dict[str, Iterable[Any]]) -> None:
    if not param_grid:
        raise MetricsConfigError("param_grid must be non-empty")
    if not list(ParameterGrid(param_grid)):
        raise MetricsConfigError("param_grid must yield at least one candidate")


def _evaluate_params(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    params: dict[str, Any],
    metric_name: str,
    inner_folds: int,
    seed: int,
    resample_fn: Callable[[pd.DataFrame, pd.Series, int], tuple[pd.DataFrame, pd.Series]]
    | None = None,
    preprocess_fn: Callable[[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]
    | None = None,
    train_fn: TrainFn = _default_train_xgb,
    predict_proba_fn: PredictProbaFn = _default_predict_proba,
) -> float:
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    splitter = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    scores: list[float] = []
    cfg = MetricsConfig(primary=metric_name, additional=())

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        # Split
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # Resample TRAIN ONLY (no leakage)
        if resample_fn is not None:
            X_train, y_train = resample_fn(X_train, y_train, seed + fold_id)

        # Preprocess: fit on train only, transform test (no leakage)
        if preprocess_fn is not None:
            X_train, X_test = preprocess_fn(X_train, X_test)

        # Train + score
        model_result = train_fn(
            X_train,
            y_train,
            params=params,
            random_state=seed + fold_id,
        )
        proba = predict_proba_fn(model_result.model, X_test)
        metrics = score_metrics(cfg, np.asarray(y_test), proba)
        scores.append(metrics[metric_name])

    return float(np.nanmean(scores))


def select_best_params(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    param_grid: dict[str, Iterable[Any]],
    metric_name: str,
    inner_folds: int,
    seed: int,
    base_params: dict[str, Any] | None = None,
    resample_fn: Callable[[pd.DataFrame, pd.Series, int], tuple[pd.DataFrame, pd.Series]]
    | None = None,
    preprocess_fn: Callable[[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]
    | None = None,
    train_fn: TrainFn = _default_train_xgb,
    predict_proba_fn: PredictProbaFn = _default_predict_proba,
) -> tuple[dict[str, Any], float]:
    _validate_grid(param_grid)

    candidates = list(ParameterGrid(param_grid))
    best_score: float | None = None
    best_params: dict[str, Any] | None = None
    greater_is_better = _metric_greater_is_better(metric_name)

    for candidate in candidates:
        merged = dict(base_params or {})
        merged.update(candidate)

        score = _evaluate_params(
            X,
            y,
            params=merged,
            metric_name=metric_name,
            inner_folds=inner_folds,
            seed=seed,
            resample_fn=resample_fn,
            preprocess_fn=preprocess_fn,
            train_fn=train_fn,
            predict_proba_fn=predict_proba_fn,
        )

        if best_score is None:
            best_score = score
            best_params = merged
            continue

        if greater_is_better:
            if score > best_score:
                best_score = score
                best_params = merged
        else:
            if score < best_score:
                best_score = score
                best_params = merged

    if best_score is None or best_params is None:
        raise MetricsConfigError("No valid hyperparameter candidates")

    return best_params, float(best_score)


def tune_and_train(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    param_grid: dict[str, Iterable[Any]],
    metric_name: str,
    inner_folds: int,
    seed: int,
    base_params: dict[str, Any] | None = None,
    resample_fn: Callable[[pd.DataFrame, pd.Series, int], tuple[pd.DataFrame, pd.Series]]
    | None = None,
    preprocess_fn: Callable[[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]
    | None = None,
    train_fn: TrainFn = _default_train_xgb,
    predict_proba_fn: PredictProbaFn = _default_predict_proba,
) -> HPOResult:
    best_params, best_score = select_best_params(
        X,
        y,
        param_grid=param_grid,
        metric_name=metric_name,
        inner_folds=inner_folds,
        seed=seed,
        base_params=base_params,
        resample_fn=resample_fn,
        preprocess_fn=preprocess_fn,
        train_fn=train_fn,
        predict_proba_fn=predict_proba_fn,
    )

    # Final refit on full provided training split (outer-train),
    # applying the SAME resampling policy (important!) and preprocessing (train-only).
    X_fit = pd.DataFrame(X).reset_index(drop=True)
    y_fit = pd.Series(y).reset_index(drop=True)

    if resample_fn is not None:
        X_fit, y_fit = resample_fn(X_fit, y_fit, seed)

    if preprocess_fn is not None:
        # Fit encoder on training only; passing X_fit twice avoids introducing any new data
        X_fit, _ = preprocess_fn(X_fit, X_fit)

    model_result = train_fn(
        X_fit,
        y_fit,
        params=best_params,
        random_state=seed,
    )

    return HPOResult(
        best_params=best_params,
        best_score=best_score,
        model=model_result.model,
    )

class AugmentAndEncode:
    def __init__(self, *, cfg, numeric_cols, base_seed: int):
        self.cfg = cfg
        self.numeric_cols = numeric_cols
        self.base_seed = base_seed
        self.call_id = 0

    def __call__(self, X_train, X_test):
        # different seed each time preprocess_fn is called (inner folds etc.)
        seed = self.base_seed + self.call_id
        self.call_id += 1

        Xtr_aug, Xte_aug = add_noisy_copies_train_test(
            X_train, X_test,
            numeric_cols=self.numeric_cols,
            cfg=self.cfg,
            seed=seed,
        )
        return one_hot_encode_train_test(Xtr_aug, Xte_aug)
