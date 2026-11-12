# Anlagenwacht

**Physics-informed anomaly detection and AI-assisted root-cause analysis for German process and energy plants – MVP.**

Anlagenwacht watches plant sensor data, raises alarms when the plant leaves its normal operating envelope, ranks the tags that most likely caused the deviation, translates energy waste into kWh / € / t CO₂, and lets an AI assistant (Claude) investigate each alarm against the plant's asset model and knowledge base – read-only, in German, with evidence.

```
Data (historian CSV, later OPC UA/MQTT)
   │
   ▼
Asset model ──► PCA monitor (T² + SPE) ──► Alarm episodes + contribution ranking
   │                 ▲                              │
   │        physics residual (e.g. pump P_el)       ▼
   │                                       Investigation assistant (Claude, tool use)
   └──────────────► knowledge base ───────► German report: evidence · causes · checks · €/kWh
```

## Quick start

```bash
pip install -e '.[dev]'
anlagenwacht benchmark all          # downloads public datasets, writes reports/
anlagenwacht investigate tep-4      # investigate the main alarm of TEP fault 4
anlagenwacht serve                  # web UI at http://127.0.0.1:8000
pytest                              # offline unit tests
pytest -m network                   # benchmark regression guards (downloads data)
```

Set `ANTHROPIC_API_KEY` to enable the Claude investigator; without it the product falls back to a deterministic rule-based report.

## Public datasets

| Dataset | Domain | Used for |
|---|---|---|
| [Tennessee Eastman Process](https://github.com/camaramm/tennessee-eastman-profBraatz) (Braatz) | Chemical process, 52 tags, 21 faults | Fault detection, false-alarm rate, root-cause ranking |
| [SKAB](https://github.com/waico/SKAB) | Water pump testbed, 8 sensors, 34 labelled experiments | Detection on utility equipment, physics energy residual |

Data is cached under `data/raw/` (override with `ANLAGENWACHT_DATA`) and is not committed.

## Current benchmark results

See [`reports/tep_benchmark.md`](reports/tep_benchmark.md) and [`reports/skab_benchmark.md`](reports/skab_benchmark.md).

| | Result |
|---|---|
| TEP false-alarm rate (normal test run) | 3.4 % |
| TEP mean detection rate (excl. faults 3/9/15) | 80.2 % |
| TEP root-cause tag in top-5 | 7 / 8 faults with literature root-cause variables |
| SKAB point-wise F1 (sensors + physics residual) | 0.74 (FAR 0.38) |

These numbers are consistent with published PCA baselines. They are the floor to beat, not a claim of state of the art. On SKAB the physics residual only marginally improves detection, because that testbed's faults (valve closures) barely change motor power; the energy model is aimed at the energy-efficiency use case rather than fault detection.

## Layout

```
src/anlagenwacht/
  assets.py          asset model (hierarchy, tags, units, German descriptions)
  datasets/          public dataset loaders (TEP, SKAB) with download cache
  detect/pca.py      PCA monitor, control limits, contributions, alarm episodes
  physics/           pump energy model, kWh/€/CO₂ conversion
  plant.py           plant workspace used by API and agent
  agent/             Claude investigator (tool use) + rule-based fallback
  knowledge/         German troubleshooting knowledge base (markdown)
  api.py, web/       FastAPI + single-page UI
  evaluate.py        benchmark harness → reports/
```

## Roadmap to pilot

See [`ROADMAP.md`](ROADMAP.md).
