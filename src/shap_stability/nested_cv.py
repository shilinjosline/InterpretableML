"""Nested cross-validation harness skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .modeling.hpo_utils import HPOResult, tune_and_train
from .modeling.hpo_utils import TrainFn, PredictProbaFn  

from shap_stability.data_augmentation import NoisyCopyConfig, add_noisy_copies_train_test
from shap_stability.data import one_hot_encode_train_test


@dataclass(frozen=True)
class OuterFold:
    repeat_id: int
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    seed: int


@dataclass(frozen=True)
class OuterFoldResult:
    repeat_id: int
    fold_id: int
    seed: int
    best_params: dict[str, Any]
    best_score: float
    model: object


def iter_outer_folds(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    outer_folds: int,
    outer_repeats: int,
    seed: int,
) -> Iterator[OuterFold]:
    """Yield outer folds for nested CV with repeat metadata."""
    for repeat_id in range(outer_repeats):
        repeat_seed = seed + repeat_id
        splitter = StratifiedKFold(
            n_splits=outer_folds,
            shuffle=True,
            random_state=repeat_seed,
        )
        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
            yield OuterFold(
                repeat_id=repeat_id,
                fold_id=fold_id,
                train_idx=train_idx,
                test_idx=test_idx,
                seed=repeat_seed,
            )


def run_inner_hpo_for_outer_folds(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    outer_folds: int,
    outer_repeats: int,
    inner_folds: int,
    seed: int,
    metric_name: str,
    param_grid: dict[str, Iterable[Any]],
    base_params: dict[str, Any] | None = None,
    resample_fn: Callable[[pd.DataFrame, pd.Series, int], tuple[pd.DataFrame, pd.Series]]
    | None = None,
    preprocess_fn: Callable[[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]
    | None = None,
) -> Iterator[OuterFoldResult]:
    """Run inner CV HPO for each outer fold and refit on the full train split."""
    for outer in iter_outer_folds(
        X,
        y,
        outer_folds=outer_folds,
        outer_repeats=outer_repeats,
        seed=seed,
    ):
        X_train = X.iloc[outer.train_idx]
        y_train = y.iloc[outer.train_idx]
        hpo_result: HPOResult = tune_and_train(
            X_train,
            y_train,
            param_grid=param_grid,
            metric_name=metric_name,
            inner_folds=inner_folds,
            seed=outer.seed,
            base_params=base_params,
            resample_fn=resample_fn,
            preprocess_fn=preprocess_fn,
        )

        yield OuterFoldResult(
            repeat_id=outer.repeat_id,
            fold_id=outer.fold_id,
            seed=outer.seed,
            best_params=hpo_result.best_params,
            best_score=hpo_result.best_score,
            model=hpo_result.model,
        )

def run_inner_hpo_for_outer_folds(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    outer_folds: int,
    outer_repeats: int,
    inner_folds: int,
    seed: int,
    metric_name: str,
    param_grid: dict[str, Iterable[Any]],
    base_params: dict[str, Any] | None = None,
    resample_fn: Callable[[pd.DataFrame, pd.Series, int], tuple[pd.DataFrame, pd.Series]] | None = None,
    preprocess_fn: Callable[[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]] | None = None,
    train_fn: callable | None = None,
    predict_proba_fn: callable | None = None,
) -> Iterator[OuterFoldResult]:
        hpo_result: HPOResult = tune_and_train(
        X_train,
        y_train,
        param_grid=param_grid,
        metric_name=metric_name,
        inner_folds=inner_folds,
        seed=outer.seed,
        base_params=base_params,
        resample_fn=resample_fn,
        preprocess_fn=preprocess_fn,
        train_fn=train_fn if train_fn is not None else None,               # or omit if you prefer defaults
        predict_proba_fn=predict_proba_fn if predict_proba_fn is not None else None,
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

