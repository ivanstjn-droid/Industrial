"""UCI 'Condition Based Maintenance of Naval Propulsion Plants' (Coraddu et al., 2014).

Numerical simulator of a frigate with a CODLAG plant (gas turbine side),
validated against real plants. 11,934 steady-state points on a grid of
9 speeds × 51 compressor decay states (kMc 0.95–1.00) × 26 turbine decay states
(kMt 0.975–1.00). Licence CC BY 4.0, DOI 10.24432/C5K31K.

The simulator output is noise-free; ``noise`` adds Gaussian measurement noise
proportional to each sensor's value so results are not trivially perfect.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from anlagenwacht.datasets._download import data_dir, fetch

URL = "https://archive.ics.uci.edu/static/public/316/condition+based+maintenance+of+naval+propulsion+plants.zip"
COLUMNS = ["lp", "v", "GTT", "GTn", "GGn", "Ts", "Tp", "T48", "T1", "T2", "P48", "P1", "P2", "Pexh", "TIC", "mf",
           "kMc", "kMt"]
OPERATING_POINT = ["v"]
SENSORS = ["GTT", "GTn", "GGn", "Ts", "T48", "T2", "P48", "P2", "Pexh", "TIC", "mf"]  # T1, P1 constant; Tp == Ts


def load_naval(noise: float = 0.002, seed: int = 0) -> pd.DataFrame:
    path = fetch(URL, data_dir() / "naval" / "cbm.zip")
    if path is None:
        raise FileNotFoundError(URL)
    with zipfile.ZipFile(path) as z:
        raw = z.read("UCI CBM Dataset/data.txt").decode("latin-1")
    df = pd.DataFrame(np.loadtxt(io.StringIO(raw)), columns=COLUMNS)
    df["kMc"] = df["kMc"].round(3)
    df["kMt"] = df["kMt"].round(3)
    df["v"] = df["v"].round(0)
    if noise:
        rng = np.random.default_rng(seed)
        for c in SENSORS + ["Tp"]:
            df[c] = df[c] * (1 + rng.normal(0, noise, len(df)))
    return df


def healthy_mask(df: pd.DataFrame) -> pd.Series:
    """'Just after overhaul' reference: both decay coefficients within 0.5 % / 0.3 % of new."""
    return (df["kMc"] >= 0.995) & (df["kMt"] >= 0.997)


def degradation_voyage(df: pd.DataFrame, n: int = 1500, kmc_end: float = 0.96, kmt_end: float = 1.0,
                       seed: int = 1) -> pd.DataFrame:
    """Build a time series: varying speeds while compressor fouling grows linearly.

    Each step picks the simulator point at the requested speed and the nearest
    decay state, i.e. a plausible 'six months of operation' trajectory.
    """
    rng = np.random.default_rng(seed)
    speeds = np.sort(df["v"].unique())
    weights = np.where(speeds >= 12, 3.0, 1.0)  # mostly cruising
    kmc_grid = np.sort(df["kMc"].unique())
    kmt_grid = np.sort(df["kMt"].unique())
    rows = []
    for i in range(n):
        frac = i / (n - 1)
        kmc = kmc_grid[np.abs(kmc_grid - (1 - frac * (1 - kmc_end))).argmin()]
        kmt = kmt_grid[np.abs(kmt_grid - (1 - frac * (1 - kmt_end))).argmin()]
        v = rng.choice(speeds, p=weights / weights.sum())
        cand = df[(df["v"] == v) & (df["kMc"] == kmc) & (df["kMt"] == kmt)]
        rows.append(cand.iloc[0])
    out = pd.DataFrame(rows).reset_index(drop=True)
    out.index = pd.date_range("2026-01-01", periods=n, freq="3h", name="time")
    return out
