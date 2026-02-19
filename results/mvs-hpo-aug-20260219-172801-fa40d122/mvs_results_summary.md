# MVS results summary

Run ID: `mvs-hpo-aug-20260219-172801-fa40d122`
Data source: `results/mvs-hpo-aug-20260219-172801-fa40d122/`

## Setup

- Dataset: Statlog German Credit
- Model: xgboost_tree
- Outer CV: 5 folds × 5 repeats (25 folds total)
- Inner CV: 3 folds with 32-config grid
- Train-only resampling ratios: 0.1, 0.3, 0.5
- Importance methods: mean(|SHAP|) and PFI
- PFI repeats: 10 per fold

## Performance (mean across folds)

| Class ratio | Accuracy | ROC AUC |
| --- | --- | --- |
| 0.1 | 0.712 | 0.746 |
| 0.3 | 0.751 | 0.778 |
| 0.5 | 0.721 | 0.776 |

## Stability (within-method)

### Directional

**SHAP**
- Rank stability (mean Spearman): 0.1: 0.54, 0.3: 0.66, 0.5: 0.58.
- Magnitude variance: 0.1: 1.16e-04, 0.3: 4.79e-05, 0.5: 6.12e-05.

**PFI**
- Rank stability (mean Spearman): 0.1: 0.39, 0.3: 0.46, 0.5: 0.43.
- Magnitude variance: 0.1: 9.09e-05, 0.3: 5.96e-05, 0.5: 6.15e-05.

### Magnitude

**SHAP**
- Rank stability (mean Spearman): 0.1: 0.54, 0.3: 0.66, 0.5: 0.58.
- Magnitude variance: 0.1: 1.16e-04, 0.3: 4.79e-05, 0.5: 6.12e-05.

**PFI**
- Rank stability (mean Spearman): 0.1: 0.49, 0.3: 0.58, 0.5: 0.51.
- Magnitude variance: 0.1: 5.19e-05, 0.3: 2.39e-05, 0.5: 3.19e-05.

## Agreement (SHAP vs PFI)

### Directional
- Spearman agreement: 0.1: 0.30, 0.3: 0.24, 0.5: 0.26.
- Top-k overlap: 0.1: 0.50, 0.3: 0.51, 0.5: 0.59.
- Cosine similarity: 0.1: 0.73, 0.3: 0.78, 0.5: 0.76.

### Magnitude
- Spearman agreement: 0.1: 0.92, 0.3: 0.88, 0.5: 0.87.
- Top-k overlap: 0.1: 0.50, 0.3: 0.51, 0.5: 0.59.
- Cosine similarity: 0.1: 0.85, 0.3: 0.89, 0.5: 0.87.

## Paired ratio differences (outer-fold paired)

Median difference with IQR in parentheses; positive values mean higher metric at higher ratio.

| Metric | 0.3 - 0.1 | 0.5 - 0.1 | 0.5 - 0.3 |
| --- | --- | --- | --- |
| accuracy | 0.045 (0.040) | 0.015 (0.040) | -0.030 (0.035) |
| roc_auc | 0.027 (0.034) | 0.030 (0.056) | -0.005 (0.024) |
| train_pos_ratio | 0.200 (0.000) | 0.400 (0.000) | 0.200 (0.000) |

## Within-fold PFI permutation uncertainty

Mean per-feature permutation std by ratio: 0.1: 1.505e-03, 0.3: 1.475e-03, 0.5: 1.528e-03.

## Notes / limitations
- Rank-stability and agreement plots show fold-level distributions; tables include mean plus median/IQR summaries.
- Magnitude-variance plots show mean with bootstrap SD across folds (dispersion, not inferential CIs).
- Directional variant preserves sign for correlation/cosine metrics; top-k overlap uses magnitudes to track important-feature membership.
- Agreement/stability top-k uses k=5 from run metadata.
- SHAP importance = mean absolute SHAP value over test samples; PFI importance = baseline metric − permuted metric (mean decrease).

## Files generated

- `results.csv`, `stability_summary.csv`, `agreement_summary.csv`
- `stability_table.csv`, `agreement_table.csv`
- plots under `results/.../plots/`
