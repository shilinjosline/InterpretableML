# scripts/run_mvs_linear_hpo_aug.py
"""
Run MVS with nested-CV HPO (leakage-safe) for XGBoost **gblinear** + optional noisy-copy augmentation.

What you get:
- Outer loop: folds × repeats
- Inner loop: CV HPO on *outer-train only*
- Resampling happens ONLY on training splits (outer-train + each inner-train)
- Preprocess inside folds: (optional noisy copies) -> one-hot encode (fit on train, transform test)
- Evaluation (metrics, SHAP, PFI) ONLY on outer test folds
- SHAP: uses your compute_tree_shap(), which switches to your LinearSHAP path when booster=="gblinear"
  (and we always pass background=X_train_enc to avoid leakage)

Run examples:
  PYTHONPATH=src python scripts/run_mvs_linear_hpo_aug.py
  PYTHONPATH=src python scripts/run_mvs_linear_hpo_aug.py --n-copies 1 --numeric-noise-frac 0.05 --categorical-flip-prob 0.05
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from shap_stability.data import load_german_credit, one_hot_encode_train_test
from shap_stability.data_augmentation import NoisyCopyConfig, add_noisy_copies_train_test
from shap_stability.explain.pfi_utils import compute_pfi_importance
from shap_stability.explain.shap_utils import compute_tree_shap
from shap_stability.experiment_utils import (
    configure_logging,
    create_run_metadata,
    generate_run_id,
    set_global_seed,
)
from shap_stability.metrics.agreement import write_agreement_summary
from shap_stability.metrics.results_io import ResultRecord, append_record_csv
from shap_stability.metrics.stability import write_stability_summary
from shap_stability.modeling.xgboost_gblinear_wrapper import (
    predict_proba,
    train_xgb_gblinear_classifier,
)
from shap_stability.nested_cv import iter_outer_folds
from shap_stability.resampling import resample_train_fold


# ---- schema assumptions for German Credit in this repo ----
NUMERIC = {
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age_years",
    "existing_credits",
    "people_liable",
}

# ---- HPO grid (gblinear) ----
PARAM_GRID = {
    "n_estimators": [100, 200, 500],
    "learning_rate": [0.05, 0.1],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [0.1, 1.0, 10.0],
}

RATIOS = [0.1, 0.3, 0.5]
AGREEMENT_TOP_K = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MVS (gblinear) with nested HPO + optional noisy copies")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--outer-repeats", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--pfi-repeats", type=int, default=10)

    # augmentation knobs (set --n-copies > 0 to enable)
    p.add_argument("--n-copies", type=int, default=0, help="Noisy copies per original feature (0 disables)")
    p.add_argument("--numeric-noise-frac", type=float, default=0.05)
    p.add_argument("--categorical-flip-prob", type=float, default=0.05)
    return p.parse_args()


def _safe_roc_auc(y_true: pd.Series, proba: np.ndarray) -> float:
    y_series = pd.Series(y_true)
    if y_series.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_series, proba))


def _compute_metrics(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    y_true_arr = np.asarray(y_true)
    pred = (proba >= 0.5).astype(int)
    acc = float(accuracy_score(y_true_arr, pred))
    auc = _safe_roc_auc(pd.Series(y_true_arr), proba)
    return {"accuracy": acc, "roc_auc": auc}


def _iter_param_configs(param_grid: dict[str, list[object]]) -> list[dict[str, object]]:
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    out: list[dict[str, object]] = []
    for combo in itertools.product(*values):
        out.append({k: v for k, v in zip(keys, combo)})
    return out


def select_best_params_gblinear(
    X_train_outer_raw: pd.DataFrame,
    y_train_outer: pd.Series,
    *,
    ratio: float,
    param_grid: dict[str, list[object]],
    inner_folds: int,
    seed: int,
    preprocess_fn,
) -> tuple[dict[str, object], float]:
    """
    Inner-CV HPO for gblinear (leakage-safe):
    - split on raw outer-train
    - resample ONLY inner-train split to target ratio
    - preprocess: (optional augmentation) + fit encoder on inner-train, transform inner-val
    """
    configs = _iter_param_configs(param_grid)
    if not configs:
        raise ValueError("Empty param grid")

    X_raw = pd.DataFrame(X_train_outer_raw).reset_index(drop=True)
    y = pd.Series(y_train_outer).reset_index(drop=True)

    cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    best_params: dict[str, object] | None = None
    best_score = -float("inf")

    for cfg_id, cfg in enumerate(configs):
        fold_scores: list[float] = []

        for fold_id, (tr_idx, va_idx) in enumerate(cv.split(X_raw, y), start=1):
            X_tr_raw = X_raw.iloc[tr_idx]
            y_tr = y.iloc[tr_idx]
            X_va_raw = X_raw.iloc[va_idx]
            y_va = y.iloc[va_idx]

            inner_seed = int(seed + 1000 * fold_id + 10_000 * cfg_id)

            # resample ONLY inner-train
            inner_resampled = resample_train_fold(
                X_tr_raw,
                y_tr,
                target_positive_ratio=float(ratio),
                random_state=inner_seed,
            )

            # preprocess: fit on inner-train only; transform inner-val
            X_tr_enc, X_va_enc = preprocess_fn(inner_resampled.X, X_va_raw, inner_seed)

            # train gblinear
            params = dict(cfg)
            params["booster"] = "gblinear"

            train_result = train_xgb_gblinear_classifier(
                X_tr_enc,
                inner_resampled.y,
                params=params,
                random_state=inner_seed,
            )

            proba = predict_proba(train_result.model, X_va_enc)
            score = _safe_roc_auc(y_va, proba)
            if not np.isnan(score):
                fold_scores.append(float(score))

        mean_score = float(np.mean(fold_scores)) if fold_scores else -float("inf")
        if mean_score > best_score:
            best_score = mean_score
            best_params = cfg

    if best_params is None:
        return {}, float("nan")
    return dict(best_params), float(best_score)


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    run_prefix = "mvs-linear-hpo-aug" if int(args.n_copies) > 0 else "mvs-linear-hpo"
    run_id = generate_run_id(prefix=run_prefix)

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
    results_path = results_dir / "results.csv"

    aug_cfg = NoisyCopyConfig(
        n_copies=int(args.n_copies),
        numeric_noise_frac=float(args.numeric_noise_frac),
        categorical_flip_prob=float(args.categorical_flip_prob),
    )

    def preprocess_fn(Xtr_raw: pd.DataFrame, Xte_raw: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fold-safe preprocessing: optional noisy copies in raw space, then one-hot encode (fit on train)."""
        if int(args.n_copies) > 0:
            Xtr_aug, Xte_aug = add_noisy_copies_train_test(
                Xtr_raw,
                Xte_raw,
                numeric_cols=NUMERIC,
                cfg=aug_cfg,
                seed=int(seed),
            )
            return one_hot_encode_train_test(Xtr_aug, Xte_aug)
        return one_hot_encode_train_test(Xtr_raw, Xte_raw)

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

        X_train_outer_raw = X_raw.iloc[outer.train_idx]
        y_train_outer = y.iloc[outer.train_idx]
        X_test_outer_raw = X_raw.iloc[outer.test_idx]
        y_test_outer = y.iloc[outer.test_idx]

        for ratio in RATIOS:
            logger.info("Target resampling ratio=%.2f", ratio)

            # ---- inner-CV HPO on OUTER-TRAIN only (leakage-safe) ----
            best_params, best_score = select_best_params_gblinear(
                X_train_outer_raw,
                y_train_outer,
                ratio=float(ratio),
                param_grid=PARAM_GRID,
                inner_folds=int(args.inner_folds),
                seed=int(outer.seed),
                preprocess_fn=preprocess_fn,
            )
            logger.info("Best inner-CV roc_auc=%.4f best_params=%s", best_score, best_params)

            # ---- outer training: resample OUTER-TRAIN only ----
            resampled_outer = resample_train_fold(
                X_train_outer_raw,
                y_train_outer,
                target_positive_ratio=float(ratio),
                random_state=int(outer.seed),
            )
            achieved_ratio = resampled_outer.positive_count / (
                resampled_outer.positive_count + resampled_outer.negative_count
            )
            logger.info("Achieved train positive ratio=%.4f", achieved_ratio)

            # preprocess: fit on resampled outer-train only; transform outer-test
            X_train_enc, X_test_enc = preprocess_fn(resampled_outer.X, X_test_outer_raw, int(outer.seed))

            # train final gblinear model
            final_params = dict(best_params)
            final_params["booster"] = "gblinear"

            train_result = train_xgb_gblinear_classifier(
                X_train_enc,
                resampled_outer.y,
                params=final_params,
                random_state=int(outer.seed),
            )

            proba = predict_proba(train_result.model, X_test_enc)
            metrics = _compute_metrics(y_test_outer, proba)
            metrics["train_pos_ratio"] = float(achieved_ratio)

            # ---- explanations on HELD-OUT outer test ----
            # SHAP background must be TRAIN ONLY to avoid leakage
            shap_result = compute_tree_shap(
                train_result.model,
                X_test_enc,
                background=X_train_enc,
            )
            pfi_result = compute_pfi_importance(
                train_result.model,
                X_test_enc,
                y_test_outer,
                metric_name="roc_auc",
                n_repeats=int(args.pfi_repeats),
                random_state=int(outer.seed),
            )

            record = ResultRecord(
                fold_id=int(outer.fold_id),
                repeat_id=int(outer.repeat_id),
                seed=int(outer.seed),
                model_name="xgboost_gblinear",
                class_ratio=float(ratio),  # store TARGET ratio so summaries group cleanly
                metrics=metrics,
                shap_importance=shap_result.global_importance.to_dict(),
                pfi_importance=pfi_result.mean.to_dict(),
                pfi_importance_std=pfi_result.std.to_dict(),
            )
            append_record_csv(results_path, record)

    # ---- Summaries ----
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
        seed=int(args.seed),
        extra={
            "booster": "gblinear",
            "outer_folds": int(args.outer_folds),
            "outer_repeats": int(args.outer_repeats),
            "inner_folds": int(args.inner_folds),
            "pfi_repeats": int(args.pfi_repeats),
            "param_grid": PARAM_GRID,
            "agreement_top_k": int(AGREEMENT_TOP_K),
            "ratios": RATIOS,
            "augmentation": {
                "n_copies": int(args.n_copies),
                "numeric_noise_frac": float(args.numeric_noise_frac),
                "categorical_flip_prob": float(args.categorical_flip_prob),
            },
        },
    )
    (results_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("Done. Results written to %s", results_dir)
    print(f"Wrote results to {results_dir}")


if __name__ == "__main__":
    main()