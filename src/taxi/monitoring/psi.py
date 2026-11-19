"""Population Stability Index, as defined in the course (session 4 and 5).

PSI = sum over bins of (cur - ref) * ln(cur / ref), with 10 bins cut at the
deciles of the training data. Under 0.10 stable, 0.10-0.25 watch, above 0.25 act.
The reference profile (bin edges and shares) is computed at training time and
logged with the model, so the API and the batch check use the same bins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4  # avoids ln(0) when a bin is empty


def make_profile(values: np.ndarray, n_bins: int = 10) -> dict:
    """Bin edges at the training deciles, and the share of training rows in each bin."""
    values = np.asarray(values, dtype=float)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return {"edges": edges.tolist(), "shares": bin_shares(values, edges).tolist()}


def bin_shares(values: np.ndarray, edges) -> np.ndarray:
    idx = np.searchsorted(np.asarray(edges, dtype=float), np.asarray(values, dtype=float), side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1)
    return counts / max(counts.sum(), 1)


def psi_from_shares(ref: np.ndarray, cur: np.ndarray) -> float:
    ref = np.clip(np.asarray(ref, dtype=float), EPS, None)
    cur = np.clip(np.asarray(cur, dtype=float), EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi(profile: dict, current: np.ndarray) -> float:
    return psi_from_shares(profile["shares"], bin_shares(current, profile["edges"]))


def reference_profile(train: pd.DataFrame, features: list[str]) -> dict:
    return {f: make_profile(train[f].to_numpy()) for f in features}


def level(value: float, watch: float, act: float) -> str:
    return "act" if value > act else "watch" if value > watch else "stable"
