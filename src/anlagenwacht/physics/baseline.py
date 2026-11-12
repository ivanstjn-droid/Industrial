"""Operating-point normalisation.

Ships (and many plants) constantly change operating point: speed, load, sea
water temperature. Raw sensor values then mostly reflect *where* the asset is
operating, not *how healthy* it is. We learn, from healthy reference data, what
each sensor should read at a given operating point (small polynomial
regression per tag) and monitor the residuals instead. Expressed in units of
the healthy residual spread, so the PCA monitor sees comparable signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd


@dataclass
class OperatingPointBaseline:
    drivers: list[str]
    degree: int = 2
    ridge: float = 1e-6
    coef_: dict[str, np.ndarray] = field(default_factory=dict)
    scale_: dict[str, float] = field(default_factory=dict)
    x_mean_: np.ndarray | None = None
    x_std_: np.ndarray | None = None

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        z = (df[self.drivers].to_numpy(float) - self.x_mean_) / self.x_std_
        cols = [np.ones(len(z))]
        for d in range(1, self.degree + 1):
            for combo in combinations_with_replacement(range(z.shape[1]), d):
                cols.append(np.prod(z[:, list(combo)], axis=1))
        return np.column_stack(cols)

    def fit(self, df: pd.DataFrame, targets: list[str]) -> "OperatingPointBaseline":
        x = df[self.drivers].to_numpy(float)
        self.x_mean_ = x.mean(axis=0)
        sd = x.std(axis=0)
        sd[sd < 1e-12] = 1.0
        self.x_std_ = sd
        X = self._design(df)
        reg = self.ridge * np.eye(X.shape[1])
        for t in targets:
            y = df[t].to_numpy(float)
            coef = np.linalg.solve(X.T @ X + reg, X.T @ y)
            self.coef_[t] = coef
            self.scale_[t] = float(np.std(y - X @ coef)) or 1e-9
        return self

    def expected(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._design(df)
        return pd.DataFrame({t: X @ c for t, c in self.coef_.items()}, index=df.index)

    def residuals(self, df: pd.DataFrame, scaled: bool = False) -> pd.DataFrame:
        r = df[list(self.coef_)] - self.expected(df)
        if scaled:
            r = r / pd.Series(self.scale_)
        return r
