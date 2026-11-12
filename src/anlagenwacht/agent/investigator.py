"""AI investigation assistant: Claude with read-only tools over the plant workspace.

Design rules (also stated to the model):
* read-only – the assistant can never write to the control system;
* every claim must cite evidence returned by a tool;
* output is advisory, in German, for a shift engineer / Energiemanager.

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

SYSTEM_PROMPT = """Du bist Anlagenwacht, ein Assistent für Prozess- und Energieingenieure in deutschen Industrieanlagen.
Du untersuchst Alarme aus einer multivariaten Prozessüberwachung (PCA, T²/SPE) und – falls vorhanden – einem Physikmodell für den Energieverbrauch.

Arbeitsweise:
- Nutze die Werkzeuge, um Belege zu sammeln: Alarmdetails, Verläufe der verdächtigen Tags, Wissensbasis.
- Unterscheide Ursache und Folge: Tags, die früh und stark abweichen und Stellgrößen (art = manipulated), die ein Regler nachführt, sind oft näher an der Ursache als Analysenwerte, die später folgen.
- Belege jede Aussage mit konkreten Zahlen aus den Werkzeugen (Tag, Abweichung in σ, Einheit). Erfinde keine Messwerte.
- Du hast nur Lesezugriff. Empfiehl Prüfungen und Maßnahmen, aber niemals eigenmächtige Sollwertänderungen; Eingriffe erfordern Freigabe durch die verantwortliche Person.
- Bei Energieauswirkungen: rechne in kW, kWh/Jahr, €/Jahr und t CO₂/Jahr und nenne die Annahmen.

Antworte auf Deutsch mit diesen Abschnitten:
## Zusammenfassung (2–3 Sätze)
## Belege
## Wahrscheinliche Ursachen (mit Konfidenz hoch/mittel/niedrig)
## Empfohlene Prüfungen
## Energie & Kosten (nur wenn relevant)"""


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
    _tool("plant_overview", "Überblick: Anlage, Zeitraum, Anzahl Alarme, Anlagenstruktur mit allen Tags.", {}),
    _tool("list_alarms", "Liste aller Alarm-Episoden mit Start, Ende, Spitzenwert und Top-Tags.", {}),
    _tool(
        "get_alarm_details",
        "Details einer Alarm-Episode: verdächtige Tags mit Beitragsanteil, Abweichung in σ, Anteil je Anlagenteil, ggf. Energie-Mehrverbrauch.",
        {"alarm_id": {"type": "integer", "description": "ID aus list_alarms"}},
    ),
    _tool(
        "get_tag_trend",
        "Verlauf eines Tags rund um eine Alarm-Episode (heruntergetastet) inkl. Verschiebung ggü. Normalbetrieb.",
        {"tag": {"type": "string"}, "alarm_id": {"type": "integer"}},
    ),
    _tool(
        "search_knowledge",
        "Durchsucht die Wissensbasis (Störungsbilder, Prüfanweisungen) nach Stichworten.",
        {"query": {"type": "string"}},
    ),
    _tool(
        "estimate_energy_cost",
        "Rechnet einen Leistungs-Mehrverbrauch (Watt) in kWh/Jahr, €/Jahr und t CO₂/Jahr um.",
        {"delta_w": {"type": "number"}, "hours_per_year": {"type": "number"}},
    ),
]


def execute_tool(plant: Plant, name: str, args: dict) -> str:
    handlers: dict[str, Callable[[], Any]] = {
        "plant_overview": lambda: {**plant.summary(), "struktur": plant.assets.to_dict()},
        "list_alarms": lambda: [{k: v for k, v in e.items() if not k.endswith("_pos")} for e in plant.episodes],
        "get_alarm_details": lambda: plant.describe_episode(int(args["alarm_id"])),
        "get_tag_trend": lambda: plant.tag_trend(str(args["tag"]), int(args["alarm_id"])),
        "search_knowledge": lambda: knowledge.search(str(args["query"])),
        "estimate_energy_cost": lambda: annualise(float(args["delta_w"]), float(args["hours_per_year"])),
    }
    if name not in handlers:
        raise KeyError(f"Unbekanntes Werkzeug {name}")
    return json.dumps(handlers[name](), ensure_ascii=False, default=str)


def investigate(plant: Plant, alarm_id: int, client: Any | None = None, question: str | None = None) -> dict:
    """Run an investigation. Returns ``{"report": str, "mode": "claude"|"rule_based", "tool_calls": [...]}``."""
    if client is None:
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return {"report": rule_based_report(plant, alarm_id), "mode": "rule_based", "tool_calls": []}
        import anthropic

        client = anthropic.Anthropic()

    prompt = question or f"Untersuche Alarm {alarm_id} der Anlage '{plant.name}'. Was ist die wahrscheinlichste Ursache und was soll die Schicht prüfen?"
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
            return {"report": "Die Anfrage wurde vom Modell abgelehnt.\n\n" + rule_based_report(plant, alarm_id),
                    "mode": "rule_based", "tool_calls": calls}
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "pause_turn":
            continue
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason != "tool_use" or not tool_uses:
            text = "\n".join(b.text for b in response.content if b.type == "text").strip()
            if response.stop_reason == "max_tokens":
                text += "\n\n_(Antwort wegen Längenlimit abgeschnitten.)_"
            return {"report": text, "mode": "claude", "tool_calls": calls}

        results = []
        for block in tool_uses:
            calls.append({"tool": block.name, "input": block.input})
            try:
                content, is_error = execute_tool(plant, block.name, dict(block.input)), False
            except (KeyError, ValueError, TypeError) as exc:
                content, is_error = f"Fehler: {exc}", True
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": is_error})
        messages.append({"role": "user", "content": results})

    return {"report": "Untersuchung nach maximaler Schrittzahl abgebrochen.\n\n" + rule_based_report(plant, alarm_id),
            "mode": "rule_based", "tool_calls": calls}


def rule_based_report(plant: Plant, alarm_id: int) -> str:
    d = plant.describe_episode(alarm_id)
    lines = [
        f"## Zusammenfassung",
        f"Alarm {alarm_id} ({d['start']} bis {d['end']}, {d['n_samples']} Messpunkte, Spitzenwert {d['peak_score']}× Grenzwert, "
        f"dominant: {d['dominant_statistic']}). Am stärksten betroffen: Anlagenteil **{next(iter(d['anteil_je_anlage']))}**.",
        "",
        "## Belege",
    ]
    for s in d["verdaechtige_tags"][:5]:
        lines.append(
            f"- {s['tag']} – {s['beschreibung']}: Anteil {s['anteil']:.0%}, Abweichung Ø {s['abweichung_sigma']:+.1f} σ, "
            f"max. zu Beginn {s['max_abweichung_beginn_sigma']:.1f} σ "
            f"({s['normal_mittel']} → {s['alarm_mittel']} {s['einheit']})"
        )
    query = " ".join(s["beschreibung"] for s in d["verdaechtige_tags"][:3])
    hints = knowledge.search(query, k=2)
    if hints:
        lines += ["", "## Hinweise aus der Wissensbasis"] + [f"- **{h['titel']}**: {h['text'][:300]}…" for h in hints]
    if d.get("energie"):
        e = d["energie"]
        h = e["hochrechnung_falls_dauerhaft"]
        lines += ["", "## Energie & Kosten",
                  f"Mehrverbrauch ggü. Physikmodell: {e['mehrverbrauch_w']} W → bei Dauerbetrieb {h['kwh_per_year']} kWh/a, "
                  f"{h['eur_per_year']} €/a, {h['t_co2_per_year']} t CO₂/a (Annahmen: {h['assumptions']})."]
    lines += ["", "_Regelbasierter Bericht (kein KI-Modell konfiguriert). Ursachenbewertung durch Fachpersonal erforderlich._"]
    return "\n".join(lines)
