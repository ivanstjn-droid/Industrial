# Anlagenwacht

**Physics-informed anomaly detection and AI-assisted root-cause analysis for process plants, energy systems and ships – MVP.**

Anlagenwacht watches plant sensor data, raises alarms when the plant leaves its normal operating envelope, ranks the tags that most likely caused the deviation, translates energy and fuel waste into kWh / t fuel / t CO₂ / € (including EU ETS allowances for ships), and lets an AI assistant (Claude) investigate each alarm against the plant's asset model and knowledge base – read-only, with evidence.

```
Data (historian CSV, later OPC UA/MQTT)
   │
   ▼
Asset model ──► PCA monitor (T² + SPE) ──► Alarm episodes + contribution ranking
   │                 ▲                              │
   │        physics residual (e.g. pump P_el)       ▼
   │                                       Investigation assistant (Claude, tool use)
   └──────────────► knowledge base ───────► Report: evidence · causes · checks · €/kWh
```

## Quick start

```bash
pip install -e '.[dev]'
anlagenwacht benchmark all          # downloads public datasets, writes reports/
anlagenwacht investigate tep-4      # investigate the main alarm of TEP fault 4
anlagenwacht investigate vessel-hull_fouling   # ship: hull fouling → fuel, CO₂, EU ETS cost
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
| [UCI naval propulsion plant](https://doi.org/10.24432/C5K31K) (Coraddu et al.) | Frigate gas-turbine simulator, 11,934 points with known compressor/turbine wear | Wear detection and estimation, fuel penalty → CO₂ → EU ETS cost |
| Vessel simulator (synthetic, in this repo) | Merchant ship propulsion + central cooling, 4 injected faults | End-to-end ship demo, root-cause ranking where the cause is known |

Data is cached under `data/raw/` (override with `ANLAGENWACHT_DATA`) and is not committed.

## Current benchmark results

See [`reports/tep_benchmark.md`](reports/tep_benchmark.md), [`reports/skab_benchmark.md`](reports/skab_benchmark.md) and [`reports/maritime_benchmark.md`](reports/maritime_benchmark.md).

| | Result |
|---|---|
| TEP false-alarm rate (normal test run) | 3.4 % |
| TEP mean detection rate (excl. faults 3/9/15) | 80.2 % |
| TEP root-cause tag in top-5 | 7 / 8 faults with literature root-cause variables |
| SKAB point-wise F1 (sensors + physics residual) | 0.74 (FAR 0.38) |
| Naval GT: false alarms / compressor wear ≥ 4 % detected / wear estimate R² | 1.9 % / 100 % / 0.96 |
| Vessel sim: false alarms / root cause in top-3 | 0 % / 4 of 4 faults |

These numbers are consistent with published PCA baselines. They are the floor to beat, not a claim of state of the art. On SKAB the physics residual only marginally improves detection, because that testbed's faults (valve closures) barely change motor power; the energy model is aimed at the energy-efficiency use case rather than fault detection. The vessel simulator is synthetic: its results prove the pipeline, not real-world accuracy.

## Layout

```
src/anlagenwacht/
  assets.py          asset model (hierarchy, tags, units, descriptions)
  datasets/          dataset loaders (TEP, SKAB, UCI naval) + synthetic vessel simulator
  detect/pca.py      PCA monitor, control limits, contributions, alarm episodes
  physics/           pump energy model, operating-point baseline, kWh/€/CO₂, fuel → CO₂ → EU ETS
  plant.py           plant workspace used by API and agent
  agent/             Claude investigator (tool use) + rule-based fallback
  knowledge/         troubleshooting knowledge base (markdown)
  api.py, web/       FastAPI + single-page UI
  evaluate.py        benchmark harness → reports/ (evaluate_maritime.py for ships)
```

## Roadmap to pilot

See [`ROADMAP.md`](ROADMAP.md).
