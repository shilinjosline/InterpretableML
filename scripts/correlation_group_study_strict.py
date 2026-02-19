from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shap_stability.data import load_german_credit  # noqa: E402
from shap_stability.nested_cv import iter_outer_folds  # noqa: E402


NUMERIC = {
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age_years",
    "existing_credits",
    "people_liable",
}


def spearman_abs(a: pd.Series, b: pd.Series) -> float:
    r, _ = spearmanr(pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce"), nan_policy="omit")
    if r is None or np.isnan(r):
        return 0.0
    return float(abs(r))


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    conf = pd.crosstab(x.astype(str), y.astype(str))
    if conf.size == 0:
        return 0.0
    chi2, _, _, _ = chi2_contingency(conf, correction=False)
    n = conf.to_numpy().sum()
    if n <= 0:
        return 0.0
    phi2 = chi2 / n
    r, k = conf.shape

    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(1, n - 1))
    rcorr = r - ((r - 1) ** 2) / max(1, n - 1)
    kcorr = k - ((k - 1) ** 2) / max(1, n - 1)
    denom = max(1e-12, min(kcorr - 1, rcorr - 1))
    return float(np.sqrt(phi2corr / denom))


def correlation_ratio(cat: pd.Series, num: pd.Series) -> float:
    df = pd.DataFrame({"cat": cat.astype(str), "num": pd.to_numeric(num, errors="coerce")}).dropna()
    if df.empty:
        return 0.0
    grand = df["num"].mean()
    ss_between = 0.0
    for _, g in df.groupby("cat")["num"]:
        ss_between += len(g) * (g.mean() - grand) ** 2
    ss_total = ((df["num"] - grand) ** 2).sum()
    if ss_total <= 1e-12:
        return 0.0
    eta2 = ss_between / ss_total
    return float(np.sqrt(max(0.0, eta2)))


def association_matrix(X: pd.DataFrame, raw_vars: list[str]) -> pd.DataFrame:
    A = pd.DataFrame(np.eye(len(raw_vars)), index=raw_vars, columns=raw_vars, dtype=float)
    for i, ci in enumerate(raw_vars):
        for j in range(i + 1, len(raw_vars)):
            cj = raw_vars[j]
            xi = X[ci]
            xj = X[cj]

            ci_num = ci in NUMERIC
            cj_num = cj in NUMERIC
            if ci_num and cj_num:
                v = spearman_abs(xi, xj)
            elif (not ci_num) and (not cj_num):
                v = cramers_v(xi, xj)
            else:
                if ci_num:
                    v = correlation_ratio(xj, xi)
                else:
                    v = correlation_ratio(xi, xj)

            A.loc[ci, cj] = v
            A.loc[cj, ci] = v
    return A


def corr_groups(A: pd.DataFrame, threshold: float) -> list[list[str]]:
    feats = list(A.index)
    adj = {f: set() for f in feats}
    for i, fi in enumerate(feats):
        for j in range(i + 1, len(feats)):
            fj = feats[j]
            if float(A.loc[fi, fj]) >= threshold:
                adj[fi].add(fj)
                adj[fj].add(fi)

    visited = set()
    groups: list[list[str]] = []
    for f in feats:
        if f in visited:
            continue
        stack = [f]
        comp = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.append(cur)
            stack.extend(list(adj[cur] - visited))
        groups.append(sorted(comp))

    groups.sort(key=lambda g: (-len(g), g))
    return groups


def _encoded_feature_to_raw(raw_vars: list[str], encoded_feat: str) -> str | None:
    best = None
    for rv in raw_vars:
        if encoded_feat == rv or encoded_feat.startswith(rv + "_"):
            if best is None or len(rv) > len(best):
                best = rv
    return best


def _collect_importance_row(row: pd.Series, prefix: str) -> pd.Series:
    cols = [c for c in row.index if c.startswith(prefix)]
    s = row[cols].astype(float)
    s.index = [c.removeprefix(prefix) for c in cols]
    return s


def aggregate_encoded_to_raw(imp_encoded: pd.Series, raw_vars: list[str]) -> pd.Series:
    out = {rv: 0.0 for rv in raw_vars}
    for feat, val in imp_encoded.items():
        rv = _encoded_feature_to_raw(raw_vars, feat)
        if rv is not None:
            out[rv] += float(val)
    return pd.Series(out, dtype=float)


def aggregate_raw_to_groups(imp_raw: pd.Series, groups: list[list[str]]) -> pd.Series:
    out = {}
    for gid, members in enumerate(groups):
        out[f"group_{gid}"] = float(imp_raw.reindex(members).fillna(0.0).sum())
    return pd.Series(out, dtype=float)


# ---- metrics ----
def _normalize(v: pd.Series) -> pd.Series:
    s = float(v.sum())
    if s <= 0 or np.isnan(s):
        return v * np.nan
    return v / s


def spearman_agreement(a: pd.Series, b: pd.Series) -> float:
    common = a.index.intersection(b.index)
    if len(common) < 2:
        return float("nan")
    return float(a.loc[common].corr(b.loc[common], method="spearman"))


def topk_overlap(a: pd.Series, b: pd.Series, k: int = 5) -> float:
    a = a.dropna()
    b = b.dropna()
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    top_a = set(a.nlargest(min(k, len(a))).index)
    top_b = set(b.nlargest(min(k, len(b))).index)
    denom = min(k, len(top_a), len(top_b))
    if denom == 0:
        return float("nan")
    return float(len(top_a.intersection(top_b)) / denom)


def mean_pairwise_rank_corr(vectors: list[pd.Series]) -> float:
    if len(vectors) < 2:
        return float("nan")
    idx = sorted(set().union(*[set(v.index) for v in vectors]))
    mat = pd.DataFrame({i: vectors[i].reindex(idx) for i in range(len(vectors))})
    ranks = mat.rank(ascending=False, method="average")
    corr = ranks.corr(method="spearman").to_numpy()
    n = corr.shape[0]
    if n < 2:
        return float("nan")
    return float((corr.sum() - np.trace(corr)) / (n * (n - 1)))


def magnitude_variance(vectors: list[pd.Series]) -> float:
    if len(vectors) < 2:
        return float("nan")
    idx = sorted(set().union(*[set(v.index) for v in vectors]))
    mat = pd.DataFrame({i: _normalize(vectors[i]).reindex(idx) for i in range(len(vectors))})
    return float(mat.var(axis=1, ddof=1).mean())


def _fmt_tau(tau: float) -> str:
    return f"{tau:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str)
    ap.add_argument("--tau", type=float, default=0.5, help="dependence threshold for grouping (e.g. 0.5/0.6)")
    ap.add_argument("--k", type=int, default=5, help="top-k for overlap metrics")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}")

    meta_path = run_dir / "run_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    seed = int(meta.get("seed", 42))
    outer_folds = int(meta.get("outer_folds", 5))
    outer_repeats = int(meta.get("outer_repeats", 5))

    X_raw, y = load_german_credit()
    raw_vars = list(X_raw.columns)

    groups_map: dict[tuple[int, int], list[list[str]]] = {}
    jsonl_path = run_dir / f"corr_groups_per_fold_tau{_fmt_tau(args.tau)}.jsonl"

    lines = []
    for outer in iter_outer_folds(X_raw, y, outer_folds=outer_folds, outer_repeats=outer_repeats, seed=seed):
        Xtr = X_raw.iloc[outer.train_idx]
        A = association_matrix(Xtr, raw_vars)
        groups = corr_groups(A, threshold=float(args.tau))
        groups_map[(outer.repeat_id, outer.fold_id)] = groups

        lines.append(
            json.dumps(
                {
                    "repeat_id": int(outer.repeat_id),
                    "fold_id": int(outer.fold_id),
                    "seed": int(outer.seed),
                    "tau": float(args.tau),
                    "groups": groups,
                }
            )
        )

    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {jsonl_path}")

    results = pd.read_csv(results_path)

    required = {"class_ratio", "repeat_id", "fold_id"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(
            f"results.csv is missing columns {sorted(missing)}. "
            f"To do strict per-fold grouping you need repeat_id and fold_id in results.csv."
        )

    ratios = sorted(results["class_ratio"].unique())
    per_fold_rows = []
    summary_rows = []

    for ratio in ratios:
        subset = results[results["class_ratio"] == ratio].copy()

        shap_raw_vecs, pfi_raw_vecs = [], []
        shap_grp_vecs, pfi_grp_vecs = [], []
        agree_raw, agree_grp = [], []
        overlap_raw, overlap_grp = [], []

        for _, row in subset.iterrows():
            rid = int(row["repeat_id"])
            fid = int(row["fold_id"])
            key = (rid, fid)
            if key not in groups_map:
                continue
            groups = groups_map[key]

            shap_enc = _collect_importance_row(row, "shap_").abs()
            pfi_enc = _collect_importance_row(row, "pfi_").abs()

            shap_raw = aggregate_encoded_to_raw(shap_enc, raw_vars)
            pfi_raw = aggregate_encoded_to_raw(pfi_enc, raw_vars)

            shap_grp = aggregate_raw_to_groups(shap_raw, groups)
            pfi_grp = aggregate_raw_to_groups(pfi_raw, groups)

            shap_raw_vecs.append(shap_raw)
            pfi_raw_vecs.append(pfi_raw)
            shap_grp_vecs.append(shap_grp)
            pfi_grp_vecs.append(pfi_grp)

            ar = spearman_agreement(shap_raw, pfi_raw)
            ag = spearman_agreement(shap_grp, pfi_grp)
            oraw = topk_overlap(shap_raw, pfi_raw, k=int(args.k))
            ogrp = topk_overlap(shap_grp, pfi_grp, k=int(args.k))

            agree_raw.append(ar)
            agree_grp.append(ag)
            overlap_raw.append(oraw)
            overlap_grp.append(ogrp)

            per_fold_rows.append(
                {
                    "class_ratio": float(ratio),
                    "repeat_id": rid,
                    "fold_id": fid,
                    "tau": float(args.tau),
                    "n_groups": int(len(groups)),
                    "agreement_spearman_raw": ar,
                    "agreement_spearman_group": ag,
                    "topk_overlap_raw": oraw,
                    "topk_overlap_group": ogrp,
                }
            )

        summary_rows.append(
            {
                "class_ratio": float(ratio),
                "tau": float(args.tau),
                "n_folds": int(len(subset)),
                "n_groups_mean": float(np.mean([r["n_groups"] for r in per_fold_rows if r["class_ratio"] == float(ratio)])),
                "shap_rank_stability_raw": mean_pairwise_rank_corr(shap_raw_vecs),
                "pfi_rank_stability_raw": mean_pairwise_rank_corr(pfi_raw_vecs),
                "shap_rank_stability_group": mean_pairwise_rank_corr(shap_grp_vecs),
                "pfi_rank_stability_group": mean_pairwise_rank_corr(pfi_grp_vecs),
                "shap_magnitude_var_raw": magnitude_variance(shap_raw_vecs),
                "pfi_magnitude_var_raw": magnitude_variance(pfi_raw_vecs),
                "shap_magnitude_var_group": magnitude_variance(shap_grp_vecs),
                "pfi_magnitude_var_group": magnitude_variance(pfi_grp_vecs),
                "agreement_spearman_raw_mean": float(np.nanmean(agree_raw)) if len(agree_raw) else float("nan"),
                "agreement_spearman_raw_std": float(np.nanstd(agree_raw)) if len(agree_raw) else float("nan"),
                "agreement_spearman_group_mean": float(np.nanmean(agree_grp)) if len(agree_grp) else float("nan"),
                "agreement_spearman_group_std": float(np.nanstd(agree_grp)) if len(agree_grp) else float("nan"),
                "topk_overlap_raw_mean": float(np.nanmean(overlap_raw)) if len(overlap_raw) else float("nan"),
                "topk_overlap_group_mean": float(np.nanmean(overlap_grp)) if len(overlap_grp) else float("nan"),
            }
        )

    per_fold_df = pd.DataFrame(per_fold_rows)
    per_fold_out = run_dir / f"corr_group_per_fold_strict_tau{_fmt_tau(args.tau)}.csv"
    per_fold_df.to_csv(per_fold_out, index=False)
    print(f"Wrote {per_fold_out}")

    summary_df = pd.DataFrame(summary_rows).sort_values("class_ratio")
    summary_out = run_dir / f"corr_group_summary_strict_tau{_fmt_tau(args.tau)}.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    main()