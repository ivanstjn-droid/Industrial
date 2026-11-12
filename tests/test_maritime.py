import numpy as np
import pandas as pd
import pytest

from anlagenwacht import knowledge
from anlagenwacht.agent.investigator import execute_tool, rule_based_report
from anlagenwacht.datasets.vessel_sim import ROOT_CAUSE_TAGS, simulate_vessel
from anlagenwacht.physics.baseline import OperatingPointBaseline
from anlagenwacht.physics.marine import fuel_cost_and_emissions
from anlagenwacht.plant import vessel_plant


def test_fuel_cost_and_ets():
    r = fuel_cost_and_emissions(100, "HFO", eu_share=0.5, year=2026, eua_price_eur=80, fuel_price_eur_per_t=500)
    assert r["co2_t"] == pytest.approx(311.4)
    assert r["eu_allowances_t"] == pytest.approx(155.7)
    assert r["ets_cost_eur"] == pytest.approx(155.7 * 80)
    assert fuel_cost_and_emissions(100, "MGO", 1.0, year=2025)["assumptions"]["ets_phase_in"] == 0.7
    with pytest.raises(ValueError):
        fuel_cost_and_emissions(1, "COAL")


def test_operating_point_baseline_removes_load_effect():
    rng = np.random.default_rng(0)
    v = rng.uniform(8, 16, 500)
    df = pd.DataFrame({"v": v, "fuel": 0.01 * v**3 + rng.normal(0, 0.05, 500)})
    base = OperatingPointBaseline(["v"], degree=3).fit(df, ["fuel"])
    r = base.residuals(df)["fuel"]
    assert r.std() < 0.06 and abs(r.mean()) < 0.01


def test_vessel_simulator_physics():
    clean, fouled = simulate_vessel(None), simulate_vessel("hull_fouling")
    end = slice(-24 * 5, None)
    assert fouled["ME_fuel"].iloc[end].mean() > clean["ME_fuel"].iloc[end].mean() * 1.1
    strainer = simulate_vessel("sw_strainer")
    assert strainer["SW_flow"].iloc[end].mean() < 0.7 * clean["SW_flow"].iloc[end].mean()


def test_vessel_no_false_alarms_without_fault():
    p = vessel_plant(None)
    assert p.scores["alarm"].iloc[20 * 24:].mean() < 0.02


@pytest.mark.parametrize("fault", sorted(ROOT_CAUSE_TAGS))
def test_vessel_fault_detected_and_ranked(fault):
    p = vessel_plant(fault)
    eps = [e for e in p.episodes if e["end_pos"] >= 30 * 24]
    assert eps, "fault not detected"
    top3 = {t["tag"] for t in eps[0]["top_tags"][:3]}
    assert top3 & ROOT_CAUSE_TAGS[fault]


def test_hull_fouling_report_has_fuel_and_ets_cost():
    p = vessel_plant("hull_fouling")
    ep = max(p.episodes, key=lambda e: e["n_samples"])["id"]
    report = rule_based_report(p, ep)
    assert "EU ETS" in report and "Hull and propeller fouling" in report


def test_fuel_tool():
    p = vessel_plant("sw_strainer")
    out = execute_tool(p, "estimate_fuel_cost", {"fuel_t": 10, "fuel": "MGO", "eu_share": 1.0})
    assert "co2_t" in out


def test_knowledge_marine_ranking():
    assert knowledge.search([("Sea-water flow", 0.5), ("Sea-water pressure after strainer", 0.3)])[0]["title"].endswith(
        "Sea-water strainer clogging")
