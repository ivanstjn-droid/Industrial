"""Plant workspace: data + asset model + fitted monitor + alarm episodes.

This is the single object the API and the AI assistant work against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from anlagenwacht.assets import AssetModel, skab_asset_model, tep_asset_model
from anlagenwacht.detect import PCAMonitor, alarm_episodes
from anlagenwacht.physics.energy import annualise
from anlagenwacht.physics.pump import PumpEnergyModel


@dataclass
class Plant:
    name: str
    assets: AssetModel
    reference: pd.DataFrame  # normal-operation data the monitor was fitted on
    live: pd.DataFrame  # data being monitored
    monitor: PCAMonitor
    energy_model: PumpEnergyModel | None = None
    scores: pd.DataFrame = field(init=False)
    contributions: pd.DataFrame = field(init=False)
    episodes: list[dict] = field(init=False)

    def __post_init__(self) -> None:
        self.scores = self.monitor.score(self.live)
        self.contributions = self.monitor.contributions(self.live)
        self.episodes = alarm_episodes(self.scores, self.contributions)

    # ------------------------------------------------------------ queries
    def episode(self, episode_id: int) -> dict:
        for ep in self.episodes:
            if ep["id"] == episode_id:
                return ep
        raise KeyError(f"Alarm {episode_id} does not exist")

    def describe_episode(self, episode_id: int, top_n: int = 8) -> dict:
        ep = self.episode(episode_id)
        window = self.contributions.iloc[ep["start_pos"] : min(ep["start_pos"] + 20, ep["end_pos"] + 1)]
        ranked = window.mean().sort_values(ascending=False).head(top_n)
        suspects = []
        for tag_id, share in ranked.items():
            tag = self.assets.tag(tag_id)
            suspects.append(
                {
                    "tag": tag_id,
                    "description": tag.description,
                    "asset": tag.asset,
                    "kind": tag.kind,
                    "share": round(float(share), 3),
                    **self.tag_shift(tag_id, episode_id),
                }
            )
        by_asset: dict[str, float] = {}
        for tag_id, share in window.mean().items():
            a = self.assets.tag(tag_id).asset
            by_asset[a] = by_asset.get(a, 0.0) + float(share)
        out = {
            **{k: v for k, v in ep.items() if k not in {"start_pos", "end_pos", "top_tags"}},
            "suspect_tags": suspects,
            "share_by_asset": {k: round(v, 3) for k, v in sorted(by_asset.items(), key=lambda kv: -kv[1])},
        }
        if self.energy_model is not None:
            out["energy"] = self.energy_impact(episode_id)
        return out

    def tag_shift(self, tag_id: str, episode_id: int) -> dict:
        """How far a tag moved during the episode, in units and in normal-σ."""
        ep = self.episode(episode_id)
        ref = self.reference[tag_id] if tag_id in self.reference else self.live[tag_id].iloc[: ep["start_pos"]]
        seg = self.live[tag_id].iloc[ep["start_pos"] : ep["end_pos"] + 1]
        onset = seg.iloc[:20]
        mu, sd = float(ref.mean()), float(ref.std()) or 1.0
        peak = float(((onset - mu) / sd).abs().max())
        return {
            "normal_mean": round(mu, 4),
            "alarm_mean": round(float(seg.mean()), 4),
            "mean_deviation_sigma": round((float(seg.mean()) - mu) / sd, 2),
            "max_onset_deviation_sigma": round(peak, 2),
            "unit": self.assets.tag(tag_id).unit,
        }

    def tag_trend(self, tag_id: str, episode_id: int | None = None, points: int = 24) -> dict:
        if tag_id not in self.live:
            raise KeyError(f"Unknown tag {tag_id}")
        s = self.live[tag_id]
        if episode_id is not None:
            ep = self.episode(episode_id)
            lo = max(0, ep["start_pos"] - (ep["end_pos"] - ep["start_pos"] + 1) // 2 - 10)
            s = s.iloc[lo : ep["end_pos"] + 1]
        step = max(1, len(s) // points)
        down = s.iloc[::step]
        return {
            "tag": tag_id,
            "description": self.assets.tag(tag_id).description,
            "unit": self.assets.tag(tag_id).unit,
            "values": [{"t": str(t), "v": round(float(v), 4)} for t, v in down.items()],
            **({"shift": self.tag_shift(tag_id, episode_id)} if episode_id is not None else {}),
        }

    def energy_impact(self, episode_id: int) -> dict | None:
        if self.energy_model is None or "P_el_residual" not in self.live:
            return None
        ep = self.episode(episode_id)
        excess = float(self.live["P_el_residual"].iloc[ep["start_pos"] : ep["end_pos"] + 1].mean())
        return {"excess_power_w": round(excess, 2), "annualised_if_persistent": annualise(max(excess, 0.0))}

    def summary(self) -> dict:
        return {
            "plant": self.name,
            "samples": len(self.live),
            "period": [str(self.live.index[0]), str(self.live.index[-1])],
            "pca_components": self.monitor.n_components,
            "alarms": len(self.episodes),
            "alarm_rate": round(float(self.scores["alarm"].mean()), 3),
        }


# ---------------------------------------------------------------- factories
def tep_plant(fault: int = 1) -> Plant:
    from anlagenwacht.datasets.tep import load_tep

    ref = load_tep(0, "train").drop(columns="label")
    live = load_tep(fault, "test").drop(columns="label")
    monitor = PCAMonitor().fit(ref)
    return Plant(f"TEP – test run, fault {fault}", tep_asset_model(), ref, live, monitor)


SKAB_FEATURES = ["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure", "Temperature",
                 "Thermocouple", "Voltage", "Volume Flow RateRMS", "P_el_residual"]


def skab_plant(experiment: str = "valve1/1", train_rows: int = 400) -> Plant:
    from anlagenwacht.datasets.skab import load_experiments

    group = experiment.split("/")[0]
    df = load_experiments((group,))[experiment].drop(columns="anomaly")
    ref_raw = df.iloc[:train_rows]
    energy = PumpEnergyModel().fit(ref_raw)
    full = energy.transform(df)
    monitor = PCAMonitor(var_explained=0.85, persistence=5).fit(full[SKAB_FEATURES].iloc[:train_rows])
    return Plant(f"SKAB pump – experiment {experiment}", skab_asset_model(),
                 full.iloc[:train_rows], full[SKAB_FEATURES + ["P_el", "P_hyd"]], monitor, energy)


def synthetic_plant(seed: int = 0) -> Plant:
    """Offline demo / test plant: 4 correlated tags, step fault on 'T_out' at sample 300."""
    from anlagenwacht.assets import Tag

    rng = np.random.default_rng(seed)
    n = 600
    load = rng.normal(0, 1, n).cumsum() * 0.05 + 50
    df = pd.DataFrame(
        {
            "F_in": load + rng.normal(0, 0.3, n),
            "P_out": 0.5 * load + rng.normal(0, 0.2, n),
            "T_out": 0.8 * load + 20 + rng.normal(0, 0.2, n),
            "I_motor": 0.1 * load + rng.normal(0, 0.05, n),
        },
        index=pd.date_range("2026-01-01", periods=n, freq="1min", name="time"),
    )
    df.iloc[300:, df.columns.get_loc("T_out")] += 3.0
    model = AssetModel("Synthetic test plant", {"unit": "Test unit"})
    for t, d, u in [("F_in", "Inlet flow", "m³/h"), ("P_out", "Outlet pressure", "bar"),
                    ("T_out", "Outlet temperature", "°C"), ("I_motor", "Motor current", "A")]:
        model.add(Tag(t, d, u, "unit"))
    monitor = PCAMonitor(var_explained=0.9).fit(df.iloc[:250])
    return Plant("Synthetic test plant", model, df.iloc[:250], df, monitor)
