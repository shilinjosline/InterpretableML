"""Compare MVS results between a tree run and a gblinear run.

Usage:
  PYTHONPATH=src python scripts/compare_tree_vs_linear.py \
    results/mvs-hpo-... results/mvs-linear-hpo-... \
    --output results/compare_tree_vs_linear
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare tree vs linear MVS runs")
    p.add_argument("tree_dir", help="Path to tree run directory (contains results.csv)")
    p.add_argument("linear_dir", help="Path to linear run directory (contains results.csv)")
    p.add_argument("--output", default=None, help="Output directory (default: <tree_dir>/../compare_tree_vs_linear)")
    return p.parse_args()


def _load_run(run_dir: Path) -> dict:
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.csv in {run_dir}")

    stability_path = run_dir / "stability_summary.csv"
    agreement_path = run_dir / "agreement_summary.csv"
    metadata_path = run_dir / "run_metadata.json"

    results = pd.read_csv(results_path)

    stability = pd.read_csv(stability_path) if stability_path.exists() else pd.DataFrame()
    agreement = pd.read_csv(agreement_path) if agreement_path.exists() else pd.DataFrame()

    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return {
        "dir": run_dir,
        "results": results,
        "stability": stability,
        "agreement": agreement,
        "metadata": metadata,
    }


def _perf_summary(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in results.columns if c.startswith("metric_")]
    if not metric_cols:
        raise ValueError("No metric_* columns found in results.csv")

    out = (
        results.groupby("class_ratio")[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    out.columns = ["class_ratio"] + [f"{m}_{stat}" for m, stat in out.columns[1:]]
    return out.sort_values("class_ratio")


def _pick_common_variants(tree_df: pd.DataFrame, lin_df: pd.DataFrame) -> list[str]:
    if "variant" not in tree_df.columns or "variant" not in lin_df.columns:
        return ["magnitude"]
    common = sorted(set(tree_df["variant"].unique()).intersection(set(lin_df["variant"].unique())))
    if not common:
        return ["magnitude"]
    if "magnitude" in common:
        return ["magnitude"] + [v for v in common if v != "magnitude"]
    return common


def _lineplot(frame: pd.DataFrame, *, x: str, y: str, label_col: str, title: str, xlabel: str, ylabel: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for label, sub in frame.groupby(label_col):
        sub = sub.sort_values(x)
        ax.plot(sub[x], sub[y], marker="o", label=str(label))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _merge_with_delta(tree: pd.DataFrame, lin: pd.DataFrame, on: list[str], value_cols: list[str]) -> pd.DataFrame:
    t = tree[on + value_cols].copy()
    l = lin[on + value_cols].copy()
    merged = t.merge(l, on=on, suffixes=("_tree", "_linear"))
    for c in value_cols:
        merged[f"{c}_delta_linear_minus_tree"] = merged[f"{c}_linear"] - merged[f"{c}_tree"]
    return merged


def main() -> None:
    args = parse_args()
    tree_dir = Path(args.tree_dir)
    lin_dir = Path(args.linear_dir)

    out_dir = Path(args.output) if args.output else tree_dir.parent / "compare_tree_vs_linear"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    tree = _load_run(tree_dir)
    lin = _load_run(lin_dir)

    tree_perf = _perf_summary(tree["results"])
    lin_perf = _perf_summary(lin["results"])
    tree_perf["model"] = "tree"
    lin_perf["model"] = "linear"
    perf_long = pd.concat([tree_perf, lin_perf], ignore_index=True)

    perf_long.to_csv(out_dir / "performance_by_ratio.csv", index=False)

    mean_cols = [c for c in tree_perf.columns if c.endswith("_mean")]
    perf_delta = _merge_with_delta(
        tree_perf.rename(columns={"class_ratio": "ratio"}),
        lin_perf.rename(columns={"class_ratio": "ratio"}),
        on=["ratio"],
        value_cols=mean_cols,
    )
    perf_delta.to_csv(out_dir / "performance_delta.csv", index=False)

    # Plots
    if "metric_roc_auc_mean" in perf_long.columns:
        _lineplot(
            perf_long,
            x="class_ratio",
            y="metric_roc_auc_mean",
            label_col="model",
            title="Performance: ROC AUC (mean across folds)",
            xlabel="Target positive class ratio (train resampling)",
            ylabel="ROC AUC",
            output=plots_dir / "perf_roc_auc_mean.png",
        )
    if "metric_accuracy_mean" in perf_long.columns:
        _lineplot(
            perf_long,
            x="class_ratio",
            y="metric_accuracy_mean",
            label_col="model",
            title="Performance: Accuracy (mean across folds)",
            xlabel="Target positive class ratio (train resampling)",
            ylabel="Accuracy",
            output=plots_dir / "perf_accuracy_mean.png",
        )

    # ---- Stability comparison
    tree_stab = tree["stability"]
    lin_stab = lin["stability"]
    if not tree_stab.empty and not lin_stab.empty:
        variants = _pick_common_variants(tree_stab, lin_stab)
        value_cols = [c for c in ["mean_rank_corr", "mean_magnitude_var"] if c in tree_stab.columns and c in lin_stab.columns]
        if value_cols:
            tree_stab = tree_stab.rename(columns={"ratio": "class_ratio"}).copy()
            lin_stab = lin_stab.rename(columns={"ratio": "class_ratio"}).copy()

            tree_stab["model"] = "tree"
            lin_stab["model"] = "linear"
            stab_long = pd.concat([tree_stab, lin_stab], ignore_index=True)
            stab_long.to_csv(out_dir / "stability_summary_both.csv", index=False)

            stab_delta_frames = []
            for v in variants:
                tsub = tree_stab[tree_stab.get("variant", "magnitude") == v] if "variant" in tree_stab.columns else tree_stab
                lsub = lin_stab[lin_stab.get("variant", "magnitude") == v] if "variant" in lin_stab.columns else lin_stab
                tsub = tsub.copy()
                lsub = lsub.copy()
                if "variant" in tsub.columns:
                    tsub = tsub[tsub["variant"] == v]
                if "variant" in lsub.columns:
                    lsub = lsub[lsub["variant"] == v]
                if "variant" not in tsub.columns:
                    tsub["variant"] = v
                if "variant" not in lsub.columns:
                    lsub["variant"] = v
                stab_delta_frames.append(
                    _merge_with_delta(tsub, lsub, on=["class_ratio", "method", "variant"], value_cols=value_cols)
                )
            stab_delta = pd.concat(stab_delta_frames, ignore_index=True)
            stab_delta.to_csv(out_dir / "stability_delta.csv", index=False)

            # plots for SHAP & PFI
            for v in variants:
                for method in sorted(set(stab_long["method"].unique())):
                    sub = stab_long[(stab_long["method"] == method)]
                    if "variant" in sub.columns:
                        sub = sub[sub["variant"] == v]
                    if "mean_rank_corr" in sub.columns:
                        _lineplot(
                            sub,
                            x="class_ratio",
                            y="mean_rank_corr",
                            label_col="model",
                            title=f"Stability: {method.upper()} rank stability ({v})",
                            xlabel="Target positive class ratio (train resampling)",
                            ylabel="Mean rank correlation (Spearman)",
                            output=plots_dir / f"stability_{method}_rank_{v}.png",
                        )
                    if "mean_magnitude_var" in sub.columns:
                        _lineplot(
                            sub,
                            x="class_ratio",
                            y="mean_magnitude_var",
                            label_col="model",
                            title=f"Stability: {method.upper()} magnitude variance ({v})",
                            xlabel="Target positive class ratio (train resampling)",
                            ylabel="Mean magnitude variance",
                            output=plots_dir / f"stability_{method}_magnitude_var_{v}.png",
                        )

    # ---- Agreement comparison
    tree_ag = tree["agreement"]
    lin_ag = lin["agreement"]
    if not tree_ag.empty and not lin_ag.empty:
        variants = _pick_common_variants(tree_ag, lin_ag)
        value_cols = [c for c in ["mean_spearman", "mean_topk_overlap", "mean_cosine"] if c in tree_ag.columns and c in lin_ag.columns]
        if value_cols:
            tree_ag = tree_ag.rename(columns={"ratio": "class_ratio"}).copy()
            lin_ag = lin_ag.rename(columns={"ratio": "class_ratio"}).copy()

            tree_ag["model"] = "tree"
            lin_ag["model"] = "linear"
            ag_long = pd.concat([tree_ag, lin_ag], ignore_index=True)
            ag_long.to_csv(out_dir / "agreement_summary_both.csv", index=False)

            # delta table
            ag_delta_frames = []
            for v in variants:
                tsub = tree_ag.copy()
                lsub = lin_ag.copy()
                if "variant" in tsub.columns:
                    tsub = tsub[tsub["variant"] == v]
                else:
                    tsub["variant"] = v
                if "variant" in lsub.columns:
                    lsub = lsub[lsub["variant"] == v]
                else:
                    lsub["variant"] = v
                ag_delta_frames.append(
                    _merge_with_delta(tsub, lsub, on=["class_ratio", "variant"], value_cols=value_cols)
                )
            ag_delta = pd.concat(ag_delta_frames, ignore_index=True)
            ag_delta.to_csv(out_dir / "agreement_delta.csv", index=False)

            # plots
            for v in variants:
                sub = ag_long.copy()
                if "variant" in sub.columns:
                    sub = sub[sub["variant"] == v]
                if "mean_spearman" in sub.columns:
                    _lineplot(
                        sub,
                        x="class_ratio",
                        y="mean_spearman",
                        label_col="model",
                        title=f"Agreement: SHAP vs PFI Spearman ({v})",
                        xlabel="Target positive class ratio (train resampling)",
                        ylabel="Mean Spearman",
                        output=plots_dir / f"agreement_spearman_{v}.png",
                    )
                if "mean_topk_overlap" in sub.columns:
                    _lineplot(
                        sub,
                        x="class_ratio",
                        y="mean_topk_overlap",
                        label_col="model",
                        title=f"Agreement: SHAP vs PFI top-k overlap ({v})",
                        xlabel="Target positive class ratio (train resampling)",
                        ylabel="Mean top-k overlap",
                        output=plots_dir / f"agreement_topk_overlap_{v}.png",
                    )
                if "mean_cosine" in sub.columns:
                    _lineplot(
                        sub,
                        x="class_ratio",
                        y="mean_cosine",
                        label_col="model",
                        title=f"Agreement: SHAP vs PFI cosine ({v})",
                        xlabel="Target positive class ratio (train resampling)",
                        ylabel="Mean cosine similarity",
                        output=plots_dir / f"agreement_cosine_{v}.png",
                    )

    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Tree vs Linear comparison",
                "",
                f"- Tree run: `{tree_dir}`",
                f"- Linear run: `{lin_dir}`",
                "",
                "## Outputs",
                "- `performance_by_ratio.csv`, `performance_delta.csv`",
                "- `stability_summary_both.csv`, `stability_delta.csv` (if available)",
                "- `agreement_summary_both.csv`, `agreement_delta.csv` (if available)",
                "- Plots under `plots/`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote comparison outputs to: {out_dir}")


if __name__ == "__main__":
    main()