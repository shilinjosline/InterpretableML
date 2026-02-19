"""Run the full MVS baseline with inner-CV HPO using XGBoost gblinear (linear booster)."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from shap_stability.metrics.agreement import write_agreement_summary  # noqa: E402
from shap_stability.experiment_utils import configure_logging  # noqa: E402
from shap_stability.experiment_utils import create_run_metadata  # noqa: E402
from shap_stability.experiment_utils import generate_run_id  # noqa: E402
from shap_stability.experiment_utils import set_global_seed  # noqa: E402
from shap_stability.data import load_german_credit, one_hot_encode_train_test  # noqa: E402
from shap_stability.explain.pfi_utils import compute_pfi_importance  # noqa: E402
from shap_stability.resampling import resample_train_fold  # noqa: E402
from shap_stability.metrics.results_io import ResultRecord, append_record_csv  # noqa: E402
from shap_stability.explain.shap_utils import compute_tree_shap  # noqa: E402
from shap_stability.metrics.stability import write_stability_summary  # noqa: E402
from shap_stability.modeling.xgboost_gblinear_wrapper import (  # noqa: E402
    predict_proba,
    train_xgb_gblinear_classifier,
)
from shap_stability.nested_cv import iter_outer_folds  # noqa: E402


PARAM_GRID = {
    "n_estimators": [100, 200, 500],
    "learning_rate": [0.05, 0.1],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [0.1, 1.0, 10.0],
}
AGREEMENT_TOP_K = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MVS gblinear with HPO")
    parser.add_argument("--output-dir", default="results", help="Base output dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--pfi-repeats", type=int, default=10)
    return parser.parse_args()


def _iter_param_configs(param_grid: dict[str, list[object]]) -> list[dict[str, object]]:
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    configs: list[dict[str, object]] = []
    for combo in itertools.product(*values):
        configs.append({k: v for k, v in zip(keys, combo)})
    return configs


def _safe_roc_auc(y_true: pd.Series, proba: np.ndarray) -> float:
    y_series = pd.Series(y_true)
    if y_series.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_series, proba))


def select_best_params_gblinear(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    *,
    param_grid: dict[str, list[object]],
    inner_folds: int,
    seed: int,
    ratio: float,
) -> tuple[dict[str, object], float]:
    """Inner-CV HPO for gblinear (leakage-safe):
    - split on raw data
    - resample ONLY the inner-train split to target ratio
    - fit encoder on inner-train, transform inner-val
    """
    configs = _iter_param_configs(param_grid)
    if not configs:
        raise ValueError("Empty param grid")

    cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    best_params: dict[str, object] | None = None
    best_score = -float("inf")

    X_raw = pd.DataFrame(X_train_raw).reset_index(drop=True)
    y = pd.Series(y_train).reset_index(drop=True)

    for cfg in configs:
        fold_scores: list[float] = []
        for fold_id, (tr_idx, va_idx) in enumerate(cv.split(X_raw, y), start=1):
            X_tr_raw = X_raw.iloc[tr_idx]
            y_tr = y.iloc[tr_idx]
            X_va_raw = X_raw.iloc[va_idx]
            y_va = y.iloc[va_idx]

            # Resample only the inner-train split
            inner_resampled = resample_train_fold(
                X_tr_raw,
                y_tr,
                target_positive_ratio=ratio,
                random_state=seed + 1000 * fold_id,
            )

            # Fit encoder on inner-train only; transform inner-val
            X_tr_enc, X_va_enc = one_hot_encode_train_test(inner_resampled.X, X_va_raw)

            train_result = train_xgb_gblinear_classifier(
                X_tr_enc,
                inner_resampled.y,
                params={k: v for k, v in cfg.items()},
                random_state=seed + 1000 * fold_id,
            )
            proba = predict_proba(train_result.model, X_va_enc)
            score = _safe_roc_auc(y_va, proba)
            if not np.isnan(score):
                fold_scores.append(score)

        mean_score = float(np.mean(fold_scores)) if fold_scores else -float("inf")
        if mean_score > best_score:
            best_score = mean_score
            best_params = cfg

    if best_params is None:
        best_params = {}
        best_score = float("nan")

    return best_params, float(best_score)


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    run_id = generate_run_id(prefix="mvs-linear-hpo")
    results_dir = Path(args.output_dir) / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "run.log"

    logger = configure_logging(
        run_id=run_id,
        seed=args.seed,
        log_file=log_path,
        force=True,
        logger_name="shap-it-like-its-hot",
    )

    X_raw, y = load_german_credit()

    ratios = [0.1, 0.3, 0.5]
    results_path = results_dir / "results.csv"

    for outer in iter_outer_folds(
        X_raw,
        y,
        outer_folds=args.outer_folds,
        outer_repeats=args.outer_repeats,
        seed=args.seed,
    ):
        logger.info(
            "Outer fold repeat=%s fold=%s seed=%s",
            outer.repeat_id,
            outer.fold_id,
            outer.seed,
        )
        X_train_raw = X_raw.iloc[outer.train_idx]
        y_train = y.iloc[outer.train_idx]
        X_test_raw = X_raw.iloc[outer.test_idx]
        y_test = y.iloc[outer.test_idx]

        for ratio in ratios:
            logger.info("Resampling ratio=%.2f", ratio)

            # Select best params using *inner CV* with resampling inside
            best_params, best_score = select_best_params_gblinear(
                X_train_raw,
                y_train,
                param_grid=PARAM_GRID,
                inner_folds=args.inner_folds,
                seed=outer.seed,
                ratio=ratio,
            )
            logger.info("Best HPO score=%.4f best_params=%s", best_score, best_params)

            # Resample the full outer-train fold to target ratio and train final model
            resampled = resample_train_fold(
                X_train_raw,
                y_train,
                target_positive_ratio=ratio,
                random_state=outer.seed,
            )
            achieved_ratio = resampled.positive_count / (
                resampled.positive_count + resampled.negative_count
            )
            logger.info("Achieved train positive ratio=%.4f", achieved_ratio)

            # Leakage-safe encoding: fit on resampled outer-train only, transform outer-test
            X_train_enc, X_test_enc = one_hot_encode_train_test(resampled.X, X_test_raw)

            train_result = train_xgb_gblinear_classifier(
                X_train_enc,
                resampled.y,
                params={k: v for k, v in best_params.items()},
                random_state=outer.seed,
            )

            proba = predict_proba(train_result.model, X_test_enc)
            # even though the name is compute_tree_shap, it works for linear models. Check Shap_utils
            shap_result = compute_tree_shap(train_result.model, X_test_enc,background=X_train_enc,)
            pfi_result = compute_pfi_importance(
                train_result.model,
                X_test_enc,
                y_test,
                metric_name="roc_auc",
                n_repeats=args.pfi_repeats,
                random_state=outer.seed,
            )

            roc_auc = _safe_roc_auc(y_test, proba)
            metrics = {
                "accuracy": float(accuracy_score(y_test, proba >= 0.5)),
                "roc_auc": float(roc_auc),
            }

            record = ResultRecord(
                fold_id=outer.fold_id,
                repeat_id=outer.repeat_id,
                seed=outer.seed,
                model_name="xgblinear",
                class_ratio=ratio,
                metrics=metrics,
                shap_importance=shap_result.global_importance.to_dict(),
                pfi_importance=pfi_result.mean.to_dict(),
                pfi_importance_std=pfi_result.std.to_dict(),
            )
            append_record_csv(results_path, record)

    frame = pd.read_csv(results_path)
    ratios_sorted = sorted(frame["class_ratio"].unique())

    write_stability_summary(
        frame,
        ratios=ratios_sorted,
        output_path=results_dir / "stability_summary.csv",
    )
    write_agreement_summary(
        frame,
        ratios=ratios_sorted,
        output_path=results_dir / "agreement_summary.csv",
        top_k=AGREEMENT_TOP_K,
    )

    metadata = create_run_metadata(
        run_id=run_id,
        seed=args.seed,
        extra={
            "outer_folds": args.outer_folds,
            "outer_repeats": args.outer_repeats,
            "inner_folds": args.inner_folds,
            "pfi_repeats": args.pfi_repeats,
            "param_grid": PARAM_GRID,
            "agreement_top_k": AGREEMENT_TOP_K,
            "booster": "gblinear",
        },
    )
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    logger.info("Results written to %s", results_dir)
    print(f"Wrote gblinear results to {results_dir}")


if __name__ == "__main__":
    main()
