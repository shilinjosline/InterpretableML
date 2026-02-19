# scripts/run_mvs_hpo_aug.py
"""Run the full MVS baseline with nested-CV HPO (leakage-safe) + optional noisy-copy augmentation.

Key points:
- Outer loop: CV on raw data
- Inner loop: HPO on outer-train only
- Resampling happens ONLY on training splits (outer-train and each inner-train fold)
- Preprocess happens inside folds:
    (optional noisy copies, created from TRAIN and applied to TEST)
    -> one-hot encode (fit on train, transform test)
- Evaluation (metrics, SHAP, PFI) happens ONLY on outer test folds (held-out)
- IMPORTANT: SHAP uses TRAIN background (leakage-safe)
- IMPORTANT: class_ratio stored = target ratio (0.1/0.3/0.5), not achieved ratio
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

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
from shap_stability.modeling.hpo_utils import select_best_params
from shap_stability.modeling.xgboost_wrapper import predict_proba, train_xgb_classifier
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

# HPO grid (tree booster)
PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 4],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

RATIOS = [0.1, 0.3, 0.5]
AGREEMENT_TOP_K = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run MVS (tree) with nested HPO + optional noisy-copy augmentation"
    )
    p.add_argument("--output-dir", default="results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--outer-repeats", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--pfi-repeats", type=int, default=10)

    # augmentation knobs (set --n-copies > 0 to enable)
    p.add_argument(
        "--n-copies",
        type=int,
        default=0,
        help="Noisy copies per original feature (0 disables augmentation)",
    )
    p.add_argument(
        "--numeric-noise-frac",
        type=float,
        default=0.05,
        help="Noise scale for numeric copies",
    )
    p.add_argument(
        "--categorical-flip-prob",
        type=float,
        default=0.05,
        help="Flip prob for categorical copies",
    )
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
    auc = _safe_roc_auc(y_true_arr, proba)
    return {"accuracy": acc, "roc_auc": float(auc)}


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    run_prefix = "mvs-hpo-aug" if int(args.n_copies) > 0 else "mvs-hpo"
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

            # ---- inner-fold resampling (train-only) ----
            def _inner_resample(
                X_inner_raw: pd.DataFrame,
                y_inner: pd.Series,
                seed: int,
            ) -> tuple[pd.DataFrame, pd.Series]:
                inner_resampled = resample_train_fold(
                    X_inner_raw,
                    y_inner,
                    target_positive_ratio=float(ratio),
                    random_state=int(seed),
                )
                return inner_resampled.X, inner_resampled.y

            # ---- fold-safe preprocessing: (optional noisy copies) -> one-hot encode ----
            def _preprocess(
                Xtr: pd.DataFrame,
                Xte: pd.DataFrame,
                *,
                local_seed: int,
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
                if int(args.n_copies) > 0:
                    Xtr_aug, Xte_aug = add_noisy_copies_train_test(
                        Xtr,
                        Xte,
                        numeric_cols=NUMERIC,
                        cfg=aug_cfg,
                        seed=int(local_seed),
                    )
                    return one_hot_encode_train_test(Xtr_aug, Xte_aug)

                return one_hot_encode_train_test(Xtr, Xte)

            # Wrapper for select_best_params signature (Xtr, Xte) -> (Xtr_enc, Xte_enc)
            def _preprocess_for_hpo(
                Xtr: pd.DataFrame,
                Xte: pd.DataFrame,
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
                # deterministic seed per call, based on outer split + ratio + content of Xtr
                h = int(pd.util.hash_pandas_object(Xtr, index=True).sum())
                local_seed = int((outer.seed + h + int(ratio * 1000)) % (2**32 - 1))
                return _preprocess(Xtr, Xte, local_seed=local_seed)

            # ---- inner-CV HPO on outer-train only ----
            best_params, best_score = select_best_params(
                X_train_outer_raw,
                y_train_outer,
                param_grid=PARAM_GRID,
                metric_name="roc_auc",
                inner_folds=args.inner_folds,
                seed=outer.seed,
                resample_fn=_inner_resample,
                preprocess_fn=_preprocess_for_hpo,
            )
            logger.info("Best inner-CV score (roc_auc)=%.4f best_params=%s", best_score, best_params)

            # ---- outer training: resample outer-train only ----
            resampled_outer = resample_train_fold(
                X_train_outer_raw,
                y_train_outer,
                target_positive_ratio=float(ratio),
                random_state=outer.seed,
            )
            achieved_ratio = resampled_outer.positive_count / (
                resampled_outer.positive_count + resampled_outer.negative_count
            )
            logger.info("Achieved train positive ratio=%.4f", achieved_ratio)

            # preprocess for final fit/eval (use deterministic seed too)
            h2 = int(pd.util.hash_pandas_object(resampled_outer.X, index=True).sum())
            final_seed = int((outer.seed + h2 + int(ratio * 1000) + 99991) % (2**32 - 1))
            X_train_enc, X_test_enc = _preprocess(
                resampled_outer.X,
                X_test_outer_raw,
                local_seed=final_seed,
            )

            # ---- train final model on outer-train ----
            train_result = train_xgb_classifier(
                X_train_enc,
                resampled_outer.y,
                params=best_params,
                random_state=outer.seed,
            )

            # ---- evaluate on held-out outer-test ----
            proba = predict_proba(train_result.model, X_test_enc)
            metrics = _compute_metrics(y_test_outer, proba)

            # (optional) log train achieved ratio as a metric column
            metrics["train_pos_ratio"] = float(achieved_ratio)

            # ---- explanations on held-out test fold ----
            # IMPORTANT: pass TRAIN background to avoid leakage
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
                random_state=outer.seed,
            )

            record = ResultRecord(
                fold_id=outer.fold_id,
                repeat_id=outer.repeat_id,
                seed=outer.seed,
                model_name="xgboost_tree",
                class_ratio=float(ratio),  # store the TARGET ratio (stable downstream)
                metrics=metrics,
                shap_importance=shap_result.global_importance.to_dict(),
                pfi_importance=pfi_result.mean.to_dict(),
                pfi_importance_std=pfi_result.std.to_dict(),
            )
            append_record_csv(results_path, record)

    # Summaries
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