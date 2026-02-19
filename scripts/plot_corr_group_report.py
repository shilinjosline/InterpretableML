from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import matplotlib.pyplot as plt


def _read_common_groups(jsonl_path: Path) -> pd.DataFrame:
    c = Counter()
    total_folds = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            total_folds += 1
            groups = obj.get("groups", [])
            for g in groups:
                if isinstance(g, list) and len(g) >= 2:
                    name = " + ".join(g)
                    c[name] += 1

    df = pd.DataFrame({"group": list(c.keys()), "count_folds": list(c.values())})
    df = df.sort_values(["count_folds", "group"], ascending=[False, True]).reset_index(drop=True)
    df.attrs["total_folds"] = total_folds
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=10) 
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    summ = run_dir / f"corr_group_summary_strict_tau{args.tau:.1f}.csv"
    if not summ.exists():
        raise FileNotFoundError(f"Missing: {summ}")

    df = pd.read_csv(summ)

    # --- Figure 1: stability (rank) ---
    fig1 = plt.figure(figsize=(10, 4), dpi=200)

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(df["class_ratio"], df["shap_rank_stability_raw"], marker="o", label="Raw features")
    ax1.plot(df["class_ratio"], df["shap_rank_stability_group"], marker="o", label="Grouped (train-only)")
    ax1.set_title("SHAP stability (rank)")
    ax1.set_xlabel("class_ratio")
    ax1.set_ylabel("Mean pairwise Spearman (ranks)")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left")

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(df["class_ratio"], df["pfi_rank_stability_raw"], marker="o", label="Raw features")
    ax2.plot(df["class_ratio"], df["pfi_rank_stability_group"], marker="o", label="Grouped (train-only)")
    ax2.set_title("PFI stability (rank)")
    ax2.set_xlabel("class_ratio")
    ax2.set_ylabel("Mean pairwise Spearman (ranks)")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    out1 = run_dir / f"corr_stability_tau{args.tau:.1f}.png"
    fig1.tight_layout()
    fig1.savefig(out1)
    plt.close(fig1)

    # --- Figure 2: agreement ---
    fig2 = plt.figure(figsize=(10, 4), dpi=200)

    ax3 = plt.subplot(1, 2, 1)
    ax3.errorbar(
        df["class_ratio"], df["agreement_spearman_raw_mean"], yerr=df["agreement_spearman_raw_std"],
        marker="o", capsize=3, label="Raw features"
    )
    ax3.errorbar(
        df["class_ratio"], df["agreement_spearman_group_mean"], yerr=df["agreement_spearman_group_std"],
        marker="o", capsize=3, label="Grouped (train-only)"
    )
    ax3.set_title("SHAP vs PFI agreement (Spearman)")
    ax3.set_xlabel("class_ratio")
    ax3.set_ylabel("Spearman")
    ax3.set_ylim(0.0, 1.0)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="lower left")

    ax4 = plt.subplot(1, 2, 2)
    ax4.plot(df["class_ratio"], df["topk_overlap_raw_mean"], marker="o", label="Raw features")
    ax4.plot(df["class_ratio"], df["topk_overlap_group_mean"], marker="o", label="Grouped (train-only)")
    ax4.set_title(f"Top-k overlap (k={args.k})")
    ax4.set_xlabel("class_ratio")
    ax4.set_ylabel("Overlap")
    ax4.set_ylim(0.0, 1.0)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="lower left")

    out2 = run_dir / f"corr_agreement_tau{args.tau:.1f}.png"
    fig2.tight_layout()
    fig2.savefig(out2)
    plt.close(fig2)

    # --- Common groups table ---
    jsonl = run_dir / f"corr_groups_per_fold_tau{args.tau:.1f}.jsonl"
    if jsonl.exists():
        gdf = _read_common_groups(jsonl)
        out3 = run_dir / f"corr_groups_most_common_tau{args.tau:.1f}.csv"
        gdf.to_csv(out3, index=False)
        print(f"Wrote {out3} (total_folds={gdf.attrs.get('total_folds')})")
    else:
        print(f"(Skipping common-groups table; missing {jsonl})")

    print(f"Wrote {out1}")
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()