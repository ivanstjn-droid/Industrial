"""Tennessee Eastman Process benchmark (Braatz et al. data set).

Source: https://github.com/camaramm/tennessee-eastman-profBraatz (mirror of the
original Braatz group data). 52 variables (XMEAS1-41, XMV1-11), 3-minute sampling.

* ``d00.dat``      – normal operation, training (500 samples; stored transposed)
* ``dNN.dat``      – fault NN, training (480 samples, all faulty)
* ``d00_te.dat``   – normal operation, test (960 samples)
* ``dNN_te.dat``   – fault NN, test (960 samples, fault introduced at sample 160)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from anlagenwacht.assets import TEP_TAGS
from anlagenwacht.datasets._download import data_dir, fetch

BASE_URL = "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/"
FAULT_ONSET_TEST = 160
SAMPLE_MINUTES = 3

FAULTS: dict[int, str] = {
    1: "A/C feed ratio, B composition constant (step)",
    2: "B composition, A/C ratio constant (step)",
    3: "D feed temperature (step)",
    4: "Reactor cooling water inlet temperature (step)",
    5: "Condenser cooling water inlet temperature (step)",
    6: "A feed loss (step)",
    7: "C header pressure loss (step)",
    8: "A, B, C feed composition (random variation)",
    9: "D feed temperature (random variation)",
    10: "C feed temperature (random variation)",
    11: "Reactor cooling water inlet temperature (random variation)",
    12: "Condenser cooling water inlet temperature (random variation)",
    13: "Reaction kinetics (slow drift)",
    14: "Reactor cooling water valve (sticking)",
    15: "Condenser cooling water valve (sticking)",
    16: "Unknown",
    17: "Unknown",
    18: "Unknown",
    19: "Unknown",
    20: "Unknown",
    21: "Stream 4 valve fixed at steady-state position",
}

# Approximate root-cause variable sets used in the fault-diagnosis literature.
# They are *not* an official ground truth; use them as a sanity check for
# contribution ranking, not as a precise accuracy figure.
ROOT_CAUSE_TAGS: dict[int, set[str]] = {
    1: {"XMEAS1", "XMEAS4", "XMV3", "XMV4"},
    2: {"XMEAS10", "XMEAS24", "XMEAS30", "XMV6"},
    4: {"XMEAS9", "XMEAS21", "XMV10"},
    5: {"XMEAS11", "XMEAS22", "XMV11"},
    6: {"XMEAS1", "XMV3", "XMEAS7"},
    7: {"XMEAS4", "XMV4"},
    11: {"XMEAS9", "XMEAS21", "XMV10"},
    14: {"XMEAS9", "XMEAS21", "XMV10"},
}


def _read(name: str) -> np.ndarray:
    path = fetch(BASE_URL + name, data_dir() / "tep" / name)
    if path is None:
        raise FileNotFoundError(name)
    arr = np.loadtxt(path)
    if arr.shape[0] == len(TEP_TAGS) and arr.shape[1] != len(TEP_TAGS):
        arr = arr.T  # d00.dat is stored variables x samples
    return arr


def load_tep(fault: int = 0, split: str = "test") -> pd.DataFrame:
    """Return a DataFrame with 52 tag columns, a time index and a ``label`` column.

    ``label`` is 1 where the fault is active, 0 otherwise.
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    name = f"d{fault:02d}{'_te' if split == 'test' else ''}.dat"
    arr = _read(name)
    df = pd.DataFrame(arr, columns=TEP_TAGS)
    df.index = pd.date_range("2026-01-01", periods=len(df), freq=f"{SAMPLE_MINUTES}min", name="time")
    label = np.zeros(len(df), dtype=int)
    if fault:
        if split == "test":
            label[FAULT_ONSET_TEST:] = 1
        else:
            label[:] = 1
    df["label"] = label
    return df
