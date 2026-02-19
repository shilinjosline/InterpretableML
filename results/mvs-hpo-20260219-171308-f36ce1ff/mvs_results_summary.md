# MVS results summary

Run ID: `mvs-hpo-20260219-171308-f36ce1ff`
Data source: `results/mvs-hpo-20260219-171308-f36ce1ff/`

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
| 0.1 | 0.719 | 0.758 |
| 0.3 | 0.758 | 0.791 |
| 0.5 | 0.721 | 0.778 |

## Stability (within-method)

### Directional

**SHAP**
- Rank stability (mean Spearman): 0.1: 0.65, 0.3: 0.84, 0.5: 0.71.
- Magnitude variance: 0.1: 2.03e-04, 0.3: 5.13e-05, 0.5: 1.01e-04.

**PFI**
- Rank stability (mean Spearman): 0.1: 0.49, 0.3: 0.62, 0.5: 0.56.
- Magnitude variance: 0.1: 1.58e-04, 0.3: 6.03e-05, 0.5: 9.89e-05.

### Magnitude

**SHAP**
- Rank stability (mean Spearman): 0.1: 0.65, 0.3: 0.84, 0.5: 0.71.
- Magnitude variance: 0.1: 2.03e-04, 0.3: 5.13e-05, 0.5: 1.01e-04.

**PFI**
- Rank stability (mean Spearman): 0.1: 0.59, 0.3: 0.69, 0.5: 0.60.
- Magnitude variance: 0.1: 9.75e-05, 0.3: 3.54e-05, 0.5: 6.27e-05.

## Agreement (SHAP vs PFI)

### Directional
- Spearman agreement: 0.1: 0.39, 0.3: 0.44, 0.5: 0.39.
- Top-k overlap: 0.1: 0.62, 0.3: 0.65, 0.5: 0.62.
- Cosine similarity: 0.1: 0.79, 0.3: 0.84, 0.5: 0.83.

### Magnitude
- Spearman agreement: 0.1: 0.86, 0.3: 0.79, 0.5: 0.81.
- Top-k overlap: 0.1: 0.62, 0.3: 0.65, 0.5: 0.62.
- Cosine similarity: 0.1: 0.87, 0.3: 0.89, 0.5: 0.89.

## Paired ratio differences (outer-fold paired)

Median difference with IQR in parentheses; positive values mean higher metric at higher ratio.

| Metric | 0.3 - 0.1 | 0.5 - 0.1 | 0.5 - 0.3 |
| --- | --- | --- | --- |
| accuracy | 0.040 (0.035) | 0.000 (0.030) | -0.035 (0.025) |
| roc_auc | 0.033 (0.041) | 0.027 (0.047) | -0.014 (0.025) |
| train_pos_ratio | 0.200 (0.000) | 0.400 (0.000) | 0.200 (0.000) |

## Within-fold PFI permutation uncertainty

Mean per-feature permutation std by ratio: 0.1: 2.786e-03, 0.3: 2.773e-03, 0.5: 2.785e-03.

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
