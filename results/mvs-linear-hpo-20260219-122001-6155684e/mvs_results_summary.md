# MVS results summary

Run ID: `mvs-linear-hpo-20260219-122001-6155684e`
Data source: `results/mvs-linear-hpo-20260219-122001-6155684e/`

## Setup

- Dataset: Statlog German Credit
- Model: xgblinear
- Outer CV: 5 folds × 5 repeats (25 folds total)
- Inner CV: 3 folds with 54-config grid
- Train-only resampling ratios: 0.1, 0.3, 0.5
- Importance methods: mean(|SHAP|) and PFI
- PFI repeats: 10 per fold

## Performance (mean across folds)

| Class ratio | Accuracy | ROC AUC |
| --- | --- | --- |
| 0.1 | 0.700 | 0.750 |
| 0.3 | 0.724 | 0.781 |
| 0.5 | 0.712 | 0.782 |

## Stability (within-method)

### Directional

**SHAP**
- Rank stability (mean Spearman): 0.1: 0.81, 0.3: 0.91, 0.5: 0.83.
- Magnitude variance: 0.1: 1.07e-04, 0.3: 1.45e-05, 0.5: 4.13e-05.

**PFI**
- Rank stability (mean Spearman): 0.1: 0.59, 0.3: 0.67, 0.5: 0.64.
- Magnitude variance: 0.1: 1.55e-04, 0.3: 8.78e-05, 0.5: 8.35e-05.

### Magnitude

**SHAP**
- Rank stability (mean Spearman): 0.1: 0.81, 0.3: 0.91, 0.5: 0.83.
- Magnitude variance: 0.1: 1.07e-04, 0.3: 1.45e-05, 0.5: 4.13e-05.

**PFI**
- Rank stability (mean Spearman): 0.1: 0.64, 0.3: 0.76, 0.5: 0.67.
- Magnitude variance: 0.1: 8.32e-05, 0.3: 4.11e-05, 0.5: 4.33e-05.

## Agreement (SHAP vs PFI)

### Directional
- Spearman agreement: 0.1: 0.38, 0.3: 0.40, 0.5: 0.44.
- Top-k overlap: 0.1: 0.56, 0.3: 0.58, 0.5: 0.61.
- Cosine similarity: 0.1: 0.71, 0.3: 0.75, 0.5: 0.78.

### Magnitude
- Spearman agreement: 0.1: 0.81, 0.3: 0.83, 0.5: 0.80.
- Top-k overlap: 0.1: 0.56, 0.3: 0.58, 0.5: 0.61.
- Cosine similarity: 0.1: 0.83, 0.3: 0.85, 0.5: 0.87.

## Paired ratio differences (outer-fold paired)

Median difference with IQR in parentheses; positive values mean higher metric at higher ratio.

| Metric | 0.3 - 0.1 | 0.5 - 0.1 | 0.5 - 0.3 |
| --- | --- | --- | --- |
| accuracy | 0.025 (0.015) | 0.015 (0.035) | -0.015 (0.035) |
| roc_auc | 0.025 (0.022) | 0.030 (0.023) | 0.002 (0.006) |

## Within-fold PFI permutation uncertainty

Mean per-feature permutation std by ratio: 0.1: 2.375e-03, 0.3: 2.299e-03, 0.5: 2.340e-03.

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
