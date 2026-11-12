"""HTTP API + minimal web UI.

    uvicorn anlagenwacht.api:app --reload
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from anlagenwacht.agent import investigate
from anlagenwacht.plant import Plant, skab_plant, synthetic_plant, tep_plant

app = FastAPI(title="Anlagenwacht", version="0.1.0")

DEMO_PLANTS = ["demo"] + [f"tep-{i}" for i in (1, 2, 4, 5, 6, 7, 11, 14)] + [
    "skab-valve1-1", "skab-valve2-1", "skab-other-1"]


@lru_cache(maxsize=16)
def get_plant(pid: str) -> Plant:
    if pid == "demo":
        return synthetic_plant()
    if pid.startswith("tep-"):
        return tep_plant(int(pid.split("-")[1]))
    if pid.startswith("skab-"):
        _, group, i = pid.split("-")
        return skab_plant(f"{group}/{i}")
    raise KeyError(pid)


def _plant(pid: str) -> Plant:
    try:
        return get_plant(pid)
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, f"Unknown plant {pid}") from exc


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")


@app.get("/api/plants")
def plants() -> list[str]:
    return DEMO_PLANTS


@app.get("/api/plants/{pid}")
def summary(pid: str) -> dict:
    p = _plant(pid)
    return {**p.summary(), "struktur": p.assets.to_dict()}


@app.get("/api/plants/{pid}/scores")
def scores(pid: str, points: int = 300) -> list[dict]:
    s = _plant(pid).scores
    step = max(1, len(s) // points)
    return [{"t": str(t), "score": round(float(r.score), 3), "alarm": int(r.alarm)} for t, r in s.iloc[::step].iterrows()]


@app.get("/api/plants/{pid}/alarms")
def alarms(pid: str) -> list[dict]:
    return [{k: v for k, v in e.items() if not k.endswith("_pos")} for e in _plant(pid).episodes]


@app.get("/api/plants/{pid}/alarms/{aid}")
def alarm(pid: str, aid: int) -> dict:
    try:
        return _plant(pid).describe_episode(aid)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/plants/{pid}/alarms/{aid}/investigate")
def investigate_alarm(pid: str, aid: int) -> dict:
    p = _plant(pid)
    try:
        p.episode(aid)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return investigate(p, aid)
