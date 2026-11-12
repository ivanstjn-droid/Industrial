from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from anlagenwacht import knowledge
from anlagenwacht.agent import investigate, rule_based_report
from anlagenwacht.agent.investigator import TOOLS, execute_tool
from anlagenwacht.assets import tep_asset_model
from anlagenwacht.detect import PCAMonitor
from anlagenwacht.physics.energy import annualise
from anlagenwacht.physics.pump import PumpEnergyModel
from anlagenwacht.plant import synthetic_plant


@pytest.fixture(scope="module")
def plant():
    return synthetic_plant()


def test_monitor_low_false_alarms_on_normal_data():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(2000, 1))
    X = pd.DataFrame(np.hstack([base + rng.normal(0, 0.1, (2000, 1)) for _ in range(5)]), columns=list("abcde"))
    mon = PCAMonitor().fit(X.iloc[:1000])
    assert mon.score(X.iloc[1000:])["alarm"].mean() < 0.02


def test_detects_step_and_ranks_faulty_tag_first(plant):
    assert plant.scores["alarm"].iloc[:250].mean() < 0.05
    assert plant.scores["alarm"].iloc[305:].mean() > 0.9
    first = plant.episodes[-1] if plant.episodes[-1]["start_pos"] >= 290 else plant.episodes[0]
    assert first["top_tags"][0]["tag"] == "T_out"


def test_contributions_sum_to_one(plant):
    np.testing.assert_allclose(plant.contributions.sum(axis=1), 1.0, rtol=1e-6)


def test_describe_episode_reports_shift(plant):
    ep = next(e for e in plant.episodes if e["start_pos"] >= 290)
    d = plant.describe_episode(ep["id"])
    top = d["suspect_tags"][0]
    assert top["tag"] == "T_out" and top["mean_deviation_sigma"] > 3


def test_tep_asset_model_complete():
    m = tep_asset_model()
    assert len(m.tags) == 52
    assert m.tag("XMEAS9").description == "Reactor temperature"
    assert m.tag("XMV10").kind == "manipulated"


def test_pump_model_detects_excess_power():
    rng = np.random.default_rng(0)
    n = 500
    q = rng.uniform(28, 34, n)
    p = rng.uniform(0.2, 1.0, n)
    v = np.full(n, 230.0)
    i = (50 + 0.9 * p * 1e5 * q / 60000 + 2 * q) / v + rng.normal(0, 0.001, n)
    df = pd.DataFrame({"Volume Flow RateRMS": q, "Pressure": p, "Voltage": v, "Current": i, "Thermocouple": 25.0})
    model = PumpEnergyModel().fit(df)
    assert abs(model.residual(df).mean()) < 1.0
    worn = df.assign(Current=df["Current"] * 1.1)
    assert model.residual(worn).mean() > 10


def test_annualise():
    r = annualise(1000, hours_per_year=6000, price_eur_per_kwh=0.2)
    assert r["kwh_per_year"] == 6000 and r["eur_per_year"] == 1200


def test_knowledge_search():
    hits = knowledge.search("reactor temperature rising cooling water")
    assert hits and "cooling water" in hits[0]["title"]


def test_tool_schemas_are_strict():
    for t in TOOLS:
        assert t["strict"] and t["input_schema"]["additionalProperties"] is False


def test_execute_tools(plant):
    ep = plant.episodes[0]["id"]
    for name, args in [("plant_overview", {}), ("list_alarms", {}), ("get_alarm_details", {"alarm_id": ep}),
                       ("get_tag_trend", {"tag": "T_out", "alarm_id": ep}), ("search_knowledge", {"query": "pump"}),
                       ("estimate_energy_cost", {"delta_w": 500, "hours_per_year": 4000})]:
        assert execute_tool(plant, name, args)


def test_rule_based_report_without_api_key(plant, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    out = investigate(plant, plant.episodes[0]["id"])
    assert out["mode"] == "rule_based" and "## Evidence" in out["report"]


class FakeClient:
    """Mimics client.beta.messages.create: first asks for alarm details, then answers."""

    def __init__(self):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        if len(self.calls) == 1:
            block = SimpleNamespace(type="tool_use", id="tu_1", name="get_alarm_details", input={"alarm_id": 1})
            return SimpleNamespace(stop_reason="tool_use", content=[block])
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="## Summary\nT_out.")])


def test_agent_loop_with_fake_client(plant):
    client = FakeClient()
    out = investigate(plant, 1, client=client)
    assert out["mode"] == "claude" and out["tool_calls"][0]["tool"] == "get_alarm_details"
    second = client.calls[1]
    assert second["model"] == "claude-opus-5-5"
    tool_result = second["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "tu_1" and not tool_result["is_error"]


def test_api_demo_plant(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from anlagenwacht.api import app

    c = TestClient(app)
    assert c.get("/").status_code == 200
    assert c.get("/api/plants/demo").json()["alarms"] >= 1
    alarms = c.get("/api/plants/demo/alarms").json()
    assert c.get(f"/api/plants/demo/alarms/{alarms[0]['id']}").status_code == 200
    assert c.post(f"/api/plants/demo/alarms/{alarms[0]['id']}/investigate").json()["mode"] == "rule_based"
    assert c.get("/api/plants/nope").status_code == 404
