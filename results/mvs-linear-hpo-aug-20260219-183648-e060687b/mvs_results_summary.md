# MVS results summary

Run ID: `mvs-linear-hpo-aug-20260219-183648-e060687b`
Data source: `results/mvs-linear-hpo-aug-20260219-183648-e060687b/`

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
| 0.1 | 0.702 | 0.752 |
| 0.3 | 0.734 | 0.783 |
| 0.5 | 0.713 | 0.782 |

## Stability (within-method)

### Directional

**SHAP**
- Rank stability (mean Spearman): 0.1: 0.79, 0.3: 0.87, 0.5: 0.79.
- Magnitude variance: 0.1: 3.26e-05, 0.3: 5.84e-06, 0.5: 1.67e-05.

**PFI**
- Rank stability (mean Spearman): 0.1: 0.52, 0.3: 0.58, 0.5: 0.53.
- Magnitude variance: 0.1: 9.02e-05, 0.3: 4.75e-05, 0.5: 4.66e-05.

### Magnitude

**SHAP**
- Rank stability (mean Spearman): 0.1: 0.79, 0.3: 0.87, 0.5: 0.79.
- Magnitude variance: 0.1: 3.26e-05, 0.3: 5.84e-06, 0.5: 1.67e-05.

**PFI**
- Rank stability (mean Spearman): 0.1: 0.59, 0.3: 0.68, 0.5: 0.60.
- Magnitude variance: 0.1: 1.94e-05, 0.3: 1.10e-05, 0.5: 1.24e-05.

## Agreement (SHAP vs PFI)

### Directional
- Spearman agreement: 0.1: 0.25, 0.3: 0.27, 0.5: 0.25.
- Top-k overlap: 0.1: 0.38, 0.3: 0.45, 0.5: 0.50.
- Cosine similarity: 0.1: 0.36, 0.3: 0.49, 0.5: 0.50.

### Magnitude
- Spearman agreement: 0.1: 0.78, 0.3: 0.79, 0.5: 0.77.
- Top-k overlap: 0.1: 0.38, 0.3: 0.45, 0.5: 0.50.
- Cosine similarity: 0.1: 0.81, 0.3: 0.82, 0.5: 0.82.

## Paired ratio differences (outer-fold paired)

Median difference with IQR in parentheses; positive values mean higher metric at higher ratio.

| Metric | 0.3 - 0.1 | 0.5 - 0.1 | 0.5 - 0.3 |
| --- | --- | --- | --- |
| accuracy | 0.030 (0.030) | 0.005 (0.035) | -0.025 (0.035) |
| roc_auc | 0.033 (0.020) | 0.028 (0.023) | -0.001 (0.015) |
| train_pos_ratio | 0.200 (0.000) | 0.400 (0.000) | 0.200 (0.000) |

## Within-fold PFI permutation uncertainty

Mean per-feature permutation std by ratio: 0.1: 1.340e-03, 0.3: 1.281e-03, 0.5: 1.323e-03.

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
