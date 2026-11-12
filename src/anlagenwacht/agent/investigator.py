"""AI investigation assistant: Claude with read-only tools over the plant workspace.

Design rules (also stated to the model):
* read-only – the assistant can never write to the control system;
* every claim must cite evidence returned by a tool;
* output is advisory, for a shift engineer or energy manager.

Without an API key the module falls back to a deterministic, rule-based report
so the product still works offline and in tests.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from anlagenwacht import knowledge
from anlagenwacht.physics.energy import annualise
from anlagenwacht.plant import Plant

MODEL = "claude-opus-5-5"
MAX_TURNS = 12

SYSTEM_PROMPT = """You are Anlagenwacht, an assistant for process and energy engineers in industrial plants.
You investigate alarms from multivariate process monitoring (PCA, T²/SPE) and, where available, a physics model of energy consumption.

How to work:
- Use the tools to gather evidence: alarm details, trends of the suspect tags, the knowledge base.
- Separate cause from effect: tags that deviate early and strongly, and manipulated variables (kind = manipulated) that a controller is compensating with, are often closer to the cause than analyser values that follow later.
- Back every statement with concrete numbers from the tools (tag, deviation in σ, unit). Never invent measurements.
- You have read-only access. Recommend checks and actions, but never unilateral setpoint changes; interventions require approval by the responsible person.
- For energy impact, express it in kW, kWh/year, €/year and t CO₂/year and state the assumptions.

Answer in English with these sections:
## Summary (2–3 sentences)
## Evidence
## Likely causes (with confidence high/medium/low)
## Recommended checks
## Energy & cost (only if relevant)"""


def _tool(name: str, description: str, properties: dict[str, Any]) -> dict:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


TOOLS = [
    _tool("plant_overview", "Overview: plant, time range, number of alarms, asset hierarchy with all tags.", {}),
    _tool("list_alarms", "List all alarm episodes with start, end, peak score and top tags.", {}),
    _tool(
        "get_alarm_details",
        "Details of one alarm episode: suspect tags with contribution share, deviation in σ, share per asset, and excess energy use if available.",
        {"alarm_id": {"type": "integer", "description": "ID from list_alarms"}},
    ),
    _tool(
        "get_tag_trend",
        "Trend of one tag around an alarm episode (downsampled), including its shift versus normal operation.",
        {"tag": {"type": "string"}, "alarm_id": {"type": "integer"}},
    ),
    _tool(
        "search_knowledge",
        "Search the knowledge base (failure patterns, inspection procedures) by keywords.",
        {"query": {"type": "string"}},
    ),
    _tool(
        "estimate_energy_cost",
        "Convert an excess power draw (watts) into kWh/year, €/year and t CO₂/year.",
        {"delta_w": {"type": "number"}, "hours_per_year": {"type": "number"}},
    ),
]


def execute_tool(plant: Plant, name: str, args: dict) -> str:
    handlers: dict[str, Callable[[], Any]] = {
        "plant_overview": lambda: {**plant.summary(), "hierarchy": plant.assets.to_dict()},
        "list_alarms": lambda: [{k: v for k, v in e.items() if not k.endswith("_pos")} for e in plant.episodes],
        "get_alarm_details": lambda: plant.describe_episode(int(args["alarm_id"])),
        "get_tag_trend": lambda: plant.tag_trend(str(args["tag"]), int(args["alarm_id"])),
        "search_knowledge": lambda: knowledge.search(str(args["query"])),
        "estimate_energy_cost": lambda: annualise(float(args["delta_w"]), float(args["hours_per_year"])),
    }
    if name not in handlers:
        raise KeyError(f"Unknown tool {name}")
    return json.dumps(handlers[name](), ensure_ascii=False, default=str)


def investigate(plant: Plant, alarm_id: int, client: Any | None = None, question: str | None = None) -> dict:
    """Run an investigation. Returns ``{"report": str, "mode": "claude"|"rule_based", "tool_calls": [...]}``."""
    if client is None:
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return {"report": rule_based_report(plant, alarm_id), "mode": "rule_based", "tool_calls": []}
        import anthropic

        client = anthropic.Anthropic()

    prompt = question or (
        f"Investigate alarm {alarm_id} of plant '{plant.name}'. "
        "What is the most likely cause, and what should the shift check?"
    )
    messages: list[dict] = [{"role": "user", "content": prompt}]
    calls: list[dict] = []

    for _ in range(MAX_TURNS):
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=messages,
        )
        if response.stop_reason == "refusal":
            return {"report": "The model declined this request.\n\n" + rule_based_report(plant, alarm_id),
                    "mode": "rule_based", "tool_calls": calls}
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "pause_turn":
            continue
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason != "tool_use" or not tool_uses:
            text = "\n".join(b.text for b in response.content if b.type == "text").strip()
            if response.stop_reason == "max_tokens":
                text += "\n\n_(Answer truncated at the length limit.)_"
            return {"report": text, "mode": "claude", "tool_calls": calls}

        results = []
        for block in tool_uses:
            calls.append({"tool": block.name, "input": block.input})
            try:
                content, is_error = execute_tool(plant, block.name, dict(block.input)), False
            except (KeyError, ValueError, TypeError) as exc:
                content, is_error = f"Error: {exc}", True
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": is_error})
        messages.append({"role": "user", "content": results})

    return {"report": "Investigation stopped after the maximum number of steps.\n\n" + rule_based_report(plant, alarm_id),
            "mode": "rule_based", "tool_calls": calls}


def rule_based_report(plant: Plant, alarm_id: int) -> str:
    d = plant.describe_episode(alarm_id)
    lines = [
        "## Summary",
        f"Alarm {alarm_id} ({d['start']} to {d['end']}, {d['n_samples']} samples, peak {d['peak_score']}× limit, "
        f"dominant statistic: {d['dominant_statistic']}). Most affected asset: **{next(iter(d['share_by_asset']))}**.",
        "",
        "## Evidence",
    ]
    for s in d["suspect_tags"][:5]:
        lines.append(
            f"- {s['tag']} – {s['description']}: share {s['share']:.0%}, mean deviation {s['mean_deviation_sigma']:+.1f} σ, "
            f"max at onset {s['max_onset_deviation_sigma']:.1f} σ "
            f"({s['normal_mean']} → {s['alarm_mean']} {s['unit']})"
        )
    query = " ".join(s["description"] for s in d["suspect_tags"][:3])
    hints = knowledge.search(query, k=2)
    if hints:
        lines += ["", "## Knowledge base hints"] + [f"- **{h['title']}**: {h['text'][:300]}…" for h in hints]
    if d.get("energy"):
        e = d["energy"]
        a = e["annualised_if_persistent"]
        lines += ["", "## Energy & cost",
                  f"Excess power vs. physics model: {e['excess_power_w']} W → if persistent {a['kwh_per_year']} kWh/yr, "
                  f"€{a['eur_per_year']}/yr, {a['t_co2_per_year']} t CO₂/yr (assumptions: {a['assumptions']})."]
    lines += ["", "_Rule-based report (no AI model configured). Root-cause assessment by qualified staff required._"]
    return "\n".join(lines)
