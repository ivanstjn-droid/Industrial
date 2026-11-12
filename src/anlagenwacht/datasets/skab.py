"""SKAB – Skoltech Anomaly Benchmark (water-circulation pump testbed).

Source: https://github.com/waico/SKAB (GPL-3.0 data). 1 Hz sampling, 8 sensors,
labelled anomalies (valve closures, cavitation, rotor imbalance, ...).
"""

from __future__ import annotations

import pandas as pd

from anlagenwacht.datasets._download import data_dir, fetch

BASE_URL = "https://raw.githubusercontent.com/waico/SKAB/master/data/"
GROUPS = {"valve1": range(0, 16), "valve2": range(0, 4), "other": range(1, 23)}


def _load_csv(rel: str) -> pd.DataFrame | None:
    path = fetch(BASE_URL + rel, data_dir() / "skab" / rel)
    if path is None:
        return None
    df = pd.read_csv(path, sep=";", parse_dates=["datetime"], index_col="datetime")
    df.index.name = "time"
    return df


def load_anomaly_free() -> pd.DataFrame:
    df = _load_csv("anomaly-free/anomaly-free.csv")
    assert df is not None
    df["anomaly"] = 0
    return df


def load_experiments(groups: tuple[str, ...] = ("valve1", "valve2", "other")) -> dict[str, pd.DataFrame]:
    """Return ``{"valve1/0": df, ...}``; each df has an ``anomaly`` label column."""
    out: dict[str, pd.DataFrame] = {}
    for group in groups:
        for i in GROUPS[group]:
            df = _load_csv(f"{group}/{i}.csv")
            if df is not None:
                df = df.drop(columns=["changepoint"], errors="ignore")
                df["anomaly"] = df["anomaly"].astype(int)
                out[f"{group}/{i}"] = df
    return out
