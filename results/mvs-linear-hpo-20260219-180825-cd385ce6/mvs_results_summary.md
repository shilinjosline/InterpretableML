# MVS results summary

Run ID: `mvs-linear-hpo-20260219-180825-cd385ce6`
Data source: `results/mvs-linear-hpo-20260219-180825-cd385ce6/`

## Setup

- Dataset: Statlog German Credit
- Model: xgboost_gblinear
- Outer CV: 5 folds × 5 repeats (25 folds total)
- Inner CV: 3 folds with 54-config grid
- Train-only resampling ratios: 0.1, 0.3, 0.5
- Importance methods: mean(|SHAP|) and PFI
- PFI repeats: 10 per fold

## Performance (mean across folds)

| Class ratio | Accuracy | ROC AUC |
| --- | --- | --- |
| 0.1 | 0.700 | 0.746 |
| 0.3 | 0.724 | 0.781 |
| 0.5 | 0.711 | 0.780 |

## Stability (within-method)

### Directional

**SHAP**
- Rank stability (mean Spearman): 0.1: 0.82, 0.3: 0.91, 0.5: 0.83.
- Magnitude variance: 0.1: 1.16e-04, 0.3: 1.45e-05, 0.5: 5.46e-05.

**PFI**
- Rank stability (mean Spearman): 0.1: 0.59, 0.3: 0.67, 0.5: 0.64.
- Magnitude variance: 0.1: 1.70e-04, 0.3: 8.61e-05, 0.5: 9.00e-05.

### Magnitude

**SHAP**
- Rank stability (mean Spearman): 0.1: 0.82, 0.3: 0.91, 0.5: 0.83.
- Magnitude variance: 0.1: 1.16e-04, 0.3: 1.45e-05, 0.5: 5.46e-05.

**PFI**
- Rank stability (mean Spearman): 0.1: 0.66, 0.3: 0.76, 0.5: 0.68.
- Magnitude variance: 0.1: 9.32e-05, 0.3: 4.11e-05, 0.5: 4.70e-05.

## Agreement (SHAP vs PFI)

### Directional
- Spearman agreement: 0.1: 0.39, 0.3: 0.41, 0.5: 0.45.
- Top-k overlap: 0.1: 0.56, 0.3: 0.58, 0.5: 0.62.
- Cosine similarity: 0.1: 0.72, 0.3: 0.75, 0.5: 0.78.

### Magnitude
- Spearman agreement: 0.1: 0.81, 0.3: 0.83, 0.5: 0.81.
- Top-k overlap: 0.1: 0.56, 0.3: 0.58, 0.5: 0.62.
- Cosine similarity: 0.1: 0.83, 0.3: 0.85, 0.5: 0.87.

## Paired ratio differences (outer-fold paired)

Median difference with IQR in parentheses; positive values mean higher metric at higher ratio.

| Metric | 0.3 - 0.1 | 0.5 - 0.1 | 0.5 - 0.3 |
| --- | --- | --- | --- |
| accuracy | 0.025 (0.015) | 0.015 (0.035) | -0.015 (0.030) |
| roc_auc | 0.037 (0.020) | 0.033 (0.025) | 0.000 (0.006) |
| train_pos_ratio | 0.200 (0.000) | 0.400 (0.000) | 0.200 (0.000) |

## Within-fold PFI permutation uncertainty

Mean per-feature permutation std by ratio: 0.1: 2.361e-03, 0.3: 2.299e-03, 0.5: 2.350e-03.

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
