"""PCA process monitoring (Hotelling T² + SPE/Q) with variable contributions.

This is the classic multivariate statistical process control baseline used
throughout the process-industry literature. It is fast, explainable and needs
only normal-operation data – exactly what a pilot customer can provide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PCAMonitor:
    var_explained: float = 0.90
    alpha: float = 0.99
    persistence: int = 3

    # fitted state
    columns: list[str] | None = None
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    P_: np.ndarray | None = None
    lam_: np.ndarray | None = None
    t2_limit_: float = np.nan
    spe_limit_: float = np.nan
    t2c_ref_: np.ndarray | None = None
    spec_ref_: np.ndarray | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame) -> "PCAMonitor":
        self.columns = list(X.columns)
        x = X.to_numpy(dtype=float)
        n, m = x.shape
        self.mean_ = x.mean(axis=0)
        std = x.std(axis=0, ddof=1)
        std[std < 1e-12] = 1.0
        self.std_ = std
        z = (x - self.mean_) / self.std_

        _, s, vt = np.linalg.svd(z, full_matrices=False)
        eig = s**2 / (n - 1)
        k = int(np.searchsorted(np.cumsum(eig) / eig.sum(), self.var_explained) + 1)
        k = max(1, min(k, m - 1))
        self.P_ = vt[:k].T
        self.lam_ = eig[:k]

        # Hotelling T² limit (F distribution)
        f = stats.f.ppf(self.alpha, k, n - k)
        self.t2_limit_ = k * (n - 1) * (n + 1) / (n * (n - k)) * f

        # SPE limit (Jackson & Mudholkar)
        res = eig[k:]
        th1, th2, th3 = res.sum(), (res**2).sum(), (res**3).sum()
        h0 = 1 - 2 * th1 * th3 / (3 * th2**2)
        ca = stats.norm.ppf(self.alpha)
        self.spe_limit_ = th1 * (
            ca * np.sqrt(2 * th2 * h0**2) / th1 + 1 + th2 * h0 * (h0 - 1) / th1**2
        ) ** (1 / h0)

        t2c, spec = self._raw_contributions(z)
        self.t2c_ref_ = np.maximum(t2c.mean(axis=0), 1e-6)
        self.spec_ref_ = np.maximum(spec.mean(axis=0), 1e-6)
        return self

    @property
    def n_components(self) -> int:
        return 0 if self.P_ is None else self.P_.shape[1]

    # ---------------------------------------------------------------- score
    def _z(self, X: pd.DataFrame) -> np.ndarray:
        if self.columns is None:
            raise RuntimeError("monitor is not fitted")
        return (X[self.columns].to_numpy(dtype=float) - self.mean_) / self.std_

    def _raw_contributions(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        P, lam = self.P_, self.lam_
        # T² complete decomposition: c_j = z_j * (P Λ⁻¹ Pᵀ z)_j  (sums to T²)
        t2c = np.maximum(z * (z @ P @ np.diag(1 / lam) @ P.T), 0.0)
        resid = z - z @ P @ P.T
        spec = resid**2
        return t2c, spec

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        z = self._z(X)
        t = z @ self.P_
        t2 = np.sum(t**2 / self.lam_, axis=1)
        resid = z - t @ self.P_.T
        spe = np.sum(resid**2, axis=1)
        out = pd.DataFrame(
            {
                "T2": t2,
                "SPE": spe,
                "T2_ratio": t2 / self.t2_limit_,
                "SPE_ratio": spe / self.spe_limit_,
            },
            index=X.index,
        )
        out["score"] = out[["T2_ratio", "SPE_ratio"]].max(axis=1)
        raw = (out["score"] > 1.0).astype(int)
        out["alarm"] = (
            raw.rolling(self.persistence, min_periods=self.persistence).min().fillna(0).astype(int)
        )
        return out

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-sample relative contribution of each variable (rows sum to 1).

        Contributions are normalised by each variable's typical contribution
        under normal operation, which reduces the well-known "smearing" of
        raw contribution plots towards high-variance variables.
        """
        t2c, spec = self._raw_contributions(self._z(X))
        rel = t2c / self.t2c_ref_ + spec / self.spec_ref_
        rel = rel / np.maximum(rel.sum(axis=1, keepdims=True), 1e-12)
        return pd.DataFrame(rel, index=X.index, columns=self.columns)


def alarm_episodes(
    scores: pd.DataFrame,
    contributions: pd.DataFrame,
    max_gap: int = 5,
    rank_window: int = 20,
    top_n: int = 5,
    min_samples: int = 3,
) -> list[dict]:
    """Group consecutive alarm samples into episodes and rank suspect tags.

    Episodes shorter than ``min_samples`` are dropped as blips.
    Tags are ranked on the *first* ``rank_window`` samples of an episode:
    early contributions point at the origin, later ones at propagation.
    """
    alarm = scores["alarm"].to_numpy()
    idx = np.flatnonzero(alarm)
    episodes: list[dict] = []
    if idx.size == 0:
        return episodes
    groups: list[list[int]] = [[int(idx[0])]]
    for i in idx[1:]:
        if i - groups[-1][-1] <= max_gap:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    groups = [g for g in groups if g[-1] - g[0] + 1 >= min_samples]
    for n, g in enumerate(groups, start=1):
        start, end = g[0], g[-1]
        window = contributions.iloc[start : min(start + rank_window, end + 1)]
        mean_c = window.mean().sort_values(ascending=False)
        seg = scores.iloc[start : end + 1]
        episodes.append(
            {
                "id": n,
                "start": str(scores.index[start]),
                "end": str(scores.index[end]),
                "start_pos": start,
                "end_pos": end,
                "n_samples": end - start + 1,
                "peak_score": round(float(seg["score"].max()), 2),
                "dominant_statistic": "SPE" if seg["SPE_ratio"].mean() > seg["T2_ratio"].mean() else "T2",
                "top_tags": [
                    {"tag": t, "share": round(float(v), 3)} for t, v in mean_c.head(top_n).items()
                ],
            }
        )
    return episodes
