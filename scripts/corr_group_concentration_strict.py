"""
Leakage-safe concentration analysis for correlation groups (STRICT per-fold groups).

Uses:
- results/<RUN_DIR>/results.csv
- results/<RUN_DIR>/corr_groups_per_fold_tau{tau:.1f}.jsonl  (from correlation_group_study_strict.py)

Computes, for each fold × class_ratio × group × method (SHAP/PFI):
- top_share = max(member) / sum(members)
- hhi       = sum(share^2)
- entropy_norm (optional; 0=concentrated, 1=even)

Outputs:
- corr_group_concentration_per_group_tau{tau}.csv
- corr_group_concentration_summary_tau{tau}.csv
- corr_group_concentration_topshare_tau{tau}.png
- corr_group_concentration_hhi_tau{tau}.png

Recommended: run with tau=0.5 for both tree + linear run dirs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shap_stability.data import load_german_credit  # noqa: E402


def _tau_tag(tau: float) -> str:
    return f"{tau:.1f}"


def _load_groups_map(run_dir: Path, tau: float) -> Dict[Tuple[int, int], List[List[str]]]:
    jsonl_path = run_dir / f"corr_groups_per_fold_tau{_tau_tag(tau)}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing {jsonl_path} (run correlation_group_study_strict.py first)")

    groups_map: Dict[Tuple[int, int], List[List[str]]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = int(rec["repeat_id"])
            fid = int(rec["fold_id"])
            groups = rec.get("groups", [])
            if not isinstance(groups, list) or not groups:
                raise ValueError(f"Bad groups record for repeat={rid} fold={fid}")
            cleaned: List[List[str]] = []
            for g in groups:
                if isinstance(g, list) and all(isinstance(x, str) for x in g) and len(g) >= 1:
                    cleaned.append(g)
            groups_map[(rid, fid)] = cleaned

    if not groups_map:
        raise ValueError(f"No fold groups loaded from {jsonl_path}")
    return groups_map


def _encoded_feature_to_raw(raw_vars: List[str], encoded_feat: str) -> str | None:
    best = None
    for rv in raw_vars:
        if encoded_feat == rv or encoded_feat.startswith(rv + "_"):
            if best is None or len(rv) > len(best):
                best = rv
    return best


def _collect_importance_row(row: pd.Series, prefix: str) -> pd.Series:
    cols = [c for c in row.index if str(c).startswith(prefix)]
    if prefix == "pfi_":
        cols = [c for c in cols if not str(c).startswith("pfi_std_")]
    s = row[cols].astype(float)
    s.index = [str(c).removeprefix(prefix) for c in cols]
    return s


def aggregate_encoded_to_raw(imp_encoded: pd.Series, raw_vars: List[str]) -> pd.Series:
    out = {rv: 0.0 for rv in raw_vars}
    for feat, val in imp_encoded.items():
        rv = _encoded_feature_to_raw(raw_vars, str(feat))
        if rv is not None:
            out[rv] += float(val)
    return pd.Series(out, dtype=float)


def _entropy_norm(shares: np.ndarray) -> float:
    # normalized to [0,1] where 0 = fully concentrated, 1 = uniform
    shares = shares[np.isfinite(shares)]
    shares = shares[shares > 0]
    k = int(shares.size)
    if k <= 1:
        return float("nan")
    h = -float(np.sum(shares * np.log(shares)))
    return float(h / math.log(k))


def _hhi(shares: np.ndarray) -> float:
    shares = shares[np.isfinite(shares)]
    if shares.size == 0:
        return float("nan")
    return float(np.sum(shares ** 2))


def _top_share(shares: np.ndarray) -> float:
    shares = shares[np.isfinite(shares)]
    if shares.size == 0:
        return float("nan")
    return float(np.max(shares))


def _group_member_vector(imp_raw: pd.Series, members: List[str]) -> np.ndarray:
    # members are raw variable names
    vals = imp_raw.reindex(members).fillna(0.0).to_numpy(dtype=float)
    vals = np.clip(vals, 0.0, None)
    return vals


def _summarize(per_group: pd.DataFrame) -> pd.DataFrame:
    df = per_group.copy()
    df["is_singleton"] = df["group_size"] == 1

    def agg_block(sub: pd.DataFrame, label: str) -> pd.DataFrame:
        g = (
            sub.groupby(["class_ratio", "method"], as_index=False)[
                ["top_share", "hhi", "entropy_norm", "group_total"]
            ]
            .agg(["mean", "std", "median"])
        )
        g.columns = [
            c[0] if c[1] == "" else f"{c[0]}_{c[1]}" for c in g.columns.to_flat_index()
        ]
        g["subset"] = label
        return g

    all_groups = agg_block(df, "all_groups")
    non_singletons = agg_block(df[~df["is_singleton"]], "size>=2")
    return pd.concat([all_groups, non_singletons], ignore_index=True)


def _plot_metric(
    summary: pd.DataFrame,
    *,
    subset: str,
    metric_mean_col: str,
    metric_std_col: str,
    title: str,
    ylabel: str,
    outpath: Path,
) -> None:
    df = summary[summary["subset"] == subset].sort_values("class_ratio")
    if df.empty:
        return

    ratios = sorted(df["class_ratio"].unique())
    fig, ax = plt.subplots(figsize=(6, 4))

    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method].sort_values("class_ratio")
        ax.errorbar(
            sub["class_ratio"].to_numpy(dtype=float),
            sub[metric_mean_col].to_numpy(dtype=float),
            yerr=sub[metric_std_col].to_numpy(dtype=float),
            marker="o",
            capsize=4,
            label=method.upper(),
        )

    ax.set_title(title)
    ax.set_xlabel("Target positive class ratio (train resampling)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(ratios, [f"{r:g}" for r in ratios])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str)
    ap.add_argument("--tau", type=float, default=0.5)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}")

    groups_map = _load_groups_map(run_dir, args.tau)

    X_raw, _ = load_german_credit()
    raw_vars = list(X_raw.columns)

    results = pd.read_csv(results_path)
    need = {"class_ratio", "repeat_id", "fold_id"}
    missing = need - set(results.columns)
    if missing:
        raise ValueError(f"results.csv missing required columns: {sorted(missing)}")

    per_group_rows: list[dict] = []

    for _, row in results.iterrows():
        ratio = float(row["class_ratio"])
        rid = int(row["repeat_id"])
        fid = int(row["fold_id"])
        key = (rid, fid)
        if key not in groups_map:
            raise KeyError(
                f"No groups for repeat_id={rid}, fold_id={fid}. "
                f"Check corr_groups_per_fold_tau{_tau_tag(args.tau)}.jsonl."
            )
        groups = groups_map[key]

        shap_enc = _collect_importance_row(row, "shap_").abs()
        pfi_enc = _collect_importance_row(row, "pfi_").abs()

        shap_raw = aggregate_encoded_to_raw(shap_enc, raw_vars)
        pfi_raw = aggregate_encoded_to_raw(pfi_enc, raw_vars)

        for method, imp_raw in (("shap", shap_raw), ("pfi", pfi_raw)):
            for gid, members in enumerate(groups):
                vals = _group_member_vector(imp_raw, members)
                total = float(np.sum(vals))
                group_size = int(len(members))
                group_name = "+".join(members)

                if total <= 0.0:
                    shares = np.array([], dtype=float)
                else:
                    shares = vals / total

                per_group_rows.append(
                    {
                        "class_ratio": ratio,
                        "repeat_id": rid,
                        "fold_id": fid,
                        "method": method,
                        "group_id": gid,
                        "group_name": group_name,
                        "group_size": group_size,
                        "group_total": total,
                        "top_share": _top_share(shares),
                        "hhi": _hhi(shares),
                        "entropy_norm": _entropy_norm(shares),
                        "max_member_value": float(np.max(vals)) if vals.size else float("nan"),
                    }
                )

    per_group = pd.DataFrame(per_group_rows)
    out_per_group = run_dir / f"corr_group_concentration_per_group_tau{_tau_tag(args.tau)}.csv"
    per_group.to_csv(out_per_group, index=False)

    summary = _summarize(per_group)
    out_summary = run_dir / f"corr_group_concentration_summary_tau{_tau_tag(args.tau)}.csv"
    summary.to_csv(out_summary, index=False)

    out_topshare = run_dir / f"corr_group_concentration_topshare_tau{_tau_tag(args.tau)}.png"
    _plot_metric(
        summary,
        subset="size>=2",
        metric_mean_col="top_share_mean",
        metric_std_col="top_share_std",
        title=f"Within-group concentration (top-member share), τ={args.tau:g}",
        ylabel="Top-member share (higher = more concentrated)",
        outpath=out_topshare,
    )

    out_hhi = run_dir / f"corr_group_concentration_hhi_tau{_tau_tag(args.tau)}.png"
    _plot_metric(
        summary,
        subset="size>=2",
        metric_mean_col="hhi_mean",
        metric_std_col="hhi_std",
        title=f"Within-group concentration (HHI), τ={args.tau:g}",
        ylabel="HHI (higher = more concentrated)",
        outpath=out_hhi,
    )

    print(f"Wrote {out_per_group}")
    print(f"Wrote {out_summary}")
    print(f"Wrote {out_topshare}")
    print(f"Wrote {out_hhi}")


if __name__ == "__main__":
    main()
