from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NoisyCopyConfig:
    n_copies: int = 1
    numeric_noise_frac: float = 0.05   # alpha (0.01–0.10 typical)
    categorical_flip_prob: float = 0.05
    suffix: str = "__copy"


def add_noisy_copies_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    numeric_cols: Iterable[str],
    cfg: NoisyCopyConfig,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create near-duplicate features for BOTH train and test, using TRAIN-only stats.
    Leakage-safe: noise scale + categorical levels come from X_train only.
    """
    Xtr = X_train.copy()
    Xte = X_test.copy()

    numeric_cols = [c for c in numeric_cols if c in Xtr.columns]
    cat_cols = [c for c in Xtr.columns if c not in numeric_cols]

    rng = np.random.default_rng(seed)

    # --- numeric: add Gaussian noise scaled by train std ---
    train_std = Xtr[numeric_cols].apply(pd.to_numeric, errors="coerce").std(ddof=0).fillna(0.0)

    for k in range(cfg.n_copies):
        for col in numeric_cols:
            sigma = float(train_std[col])
            noise_scale = cfg.numeric_noise_frac * sigma
            # if sigma==0 (constant feature), just copy exactly
            if noise_scale == 0.0:
                Xtr[f"{col}{cfg.suffix}{k+1}"] = Xtr[col]
                Xte[f"{col}{cfg.suffix}{k+1}"] = Xte[col]
                continue

            tr_vals = pd.to_numeric(Xtr[col], errors="coerce").to_numpy(dtype=float)
            te_vals = pd.to_numeric(Xte[col], errors="coerce").to_numpy(dtype=float)

            Xtr[f"{col}{cfg.suffix}{k+1}"] = tr_vals + rng.normal(0.0, noise_scale, size=len(tr_vals))
            Xte[f"{col}{cfg.suffix}{k+1}"] = te_vals + rng.normal(0.0, noise_scale, size=len(te_vals))

    # --- categorical: random flips using train category distribution ---
    for k in range(cfg.n_copies):
        for col in cat_cols:
            tr = Xtr[col].astype(str)
            te = Xte[col].astype(str)

            # train levels + probabilities
            vc = tr.value_counts(dropna=False)
            levels = vc.index.to_numpy(dtype=str)
            probs = (vc / vc.sum()).to_numpy(dtype=float)

            def make_copy(series: pd.Series) -> pd.Series:
                base = series.to_numpy(dtype=str)
                flip = rng.random(size=len(base)) < cfg.categorical_flip_prob
                if len(levels) <= 1:
                    return pd.Series(base, index=series.index)
                sampled = rng.choice(levels, size=len(base), replace=True, p=probs)
                # ensure “different” when flipping (best-effort)
                sampled = np.where(sampled == base, rng.choice(levels, size=len(base), p=probs), sampled)
                out = np.where(flip, sampled, base)
                return pd.Series(out, index=series.index)

            Xtr[f"{col}{cfg.suffix}{k+1}"] = make_copy(tr)
            Xte[f"{col}{cfg.suffix}{k+1}"] = make_copy(te)

    return Xtr, Xte