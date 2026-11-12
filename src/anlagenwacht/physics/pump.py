"""Physics-informed pump energy model.

Electrical input power should be explained by the hydraulic work the pump
delivers (P_hyd = Δp · Q) plus losses that depend smoothly on the operating
point. We fit that relationship on normal data with a small, physically
shaped regression (no black box) and track the residual: a positive residual
means the pump draws more power than it should for the work it delivers –
blockage, wear, cavitation, a throttled valve – i.e. wasted energy in kWh/€.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PRESSURE = "Pressure"  # bar
FLOW = "Volume Flow RateRMS"  # l/min
CURRENT = "Current"  # A
VOLTAGE = "Voltage"  # V
FLUID_T = "Thermocouple"  # °C


def add_power_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["P_el"] = out[VOLTAGE] * out[CURRENT]
    out["P_hyd"] = (out[PRESSURE] * 1e5) * (out[FLOW] / 60_000.0)
    return out


@dataclass
class PumpEnergyModel:
    coef_: np.ndarray | None = None
    resid_std_: float = np.nan

    ridge: float = 1.0
    x_mean_: np.ndarray | None = None
    x_std_: np.ndarray | None = None

    @staticmethod
    def _basis(df: pd.DataFrame) -> np.ndarray:
        q = df[FLOW].to_numpy(float)
        p = df[PRESSURE].to_numpy(float)
        p_hyd = (p * 1e5) * (q / 60_000.0)
        # Deliberately low-order: constant (no-load) losses + hydraulic work + flow-dependent losses.
        # Higher-order terms extrapolate badly outside the short normal-operation window.
        return np.column_stack([p_hyd, q])

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        z = (self._basis(df) - self.x_mean_) / self.x_std_
        return np.column_stack([np.ones(len(z)), z])

    def fit(self, df: pd.DataFrame) -> "PumpEnergyModel":
        b = self._basis(df)
        self.x_mean_ = b.mean(axis=0)
        sd = b.std(axis=0)
        sd[sd < 1e-9] = 1.0
        self.x_std_ = sd
        X = self._design(df)
        y = (df[VOLTAGE] * df[CURRENT]).to_numpy(float)
        reg = self.ridge * np.eye(X.shape[1])
        reg[0, 0] = 0.0  # do not shrink the intercept
        self.coef_ = np.linalg.solve(X.T @ X + reg, X.T @ y)
        self.resid_std_ = float(np.std(y - X @ self.coef_)) or 1.0
        return self

    def expected_power(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self._design(df) @ self.coef_, index=df.index, name="P_el_expected")

    def residual(self, df: pd.DataFrame) -> pd.Series:
        actual = df[VOLTAGE] * df[CURRENT]
        return (actual - self.expected_power(df)).rename("P_el_residual")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = add_power_features(df)
        out["P_el_residual"] = self.residual(df)
        return out

    def what_if(self, df: pd.DataFrame, flow_factor: float = 1.0, pressure_factor: float = 1.0) -> dict:
        """Predicted power if the operating point were changed (e.g. reduced pressure)."""
        base = self.expected_power(df).mean()
        mod = df.copy()
        mod[FLOW] = mod[FLOW] * flow_factor
        mod[PRESSURE] = mod[PRESSURE] * pressure_factor
        new = self.expected_power(mod).mean()
        return {"expected_power_w": float(base), "scenario_power_w": float(new), "delta_w": float(new - base)}
