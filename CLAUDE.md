# CLAUDE.md – Anlagenwacht

Physics-informed anomaly detection + AI root-cause assistant for process/energy plants (Germany first) and ships (second vertical). MVP stage, heading for a paid pilot.

## Commands
- `pip install -e '.[dev]'`
- `pytest` – offline unit tests (must stay green and fast, no network)
- `pytest -m network` – benchmark regression guards on public datasets
- `anlagenwacht benchmark all` – regenerates `reports/*.md|json`; commit updated reports with any detection change
- `anlagenwacht serve` – API + UI on :8000

## Non-negotiable product rules
- **Read-only.** Nothing in this codebase may write to a control system (no OPC UA writes, no setpoint changes). The assistant recommends; humans decide. This keeps us outside "safety component" scope under the EU AI Act.
- **Evidence over eloquence.** Every statement in an investigation report must trace to a tool result (tag, σ deviation, unit).
- **English UI, reports and code.** German localisation for German customers is on the roadmap; keep user-facing strings easy to extract.
- **Works offline.** Every Claude feature needs a deterministic fallback; tests never call the real API (use the fake client pattern in `tests/test_core.py`).
- **Data stays with the customer.** No telemetry, no uploading plant data anywhere except the configured LLM endpoint (EU region in pilots).

## Conventions
- Detection changes must be measured: run `anlagenwacht benchmark all` before/after and state the delta in the commit message. Do not lower thresholds in `tests/test_benchmarks.py` to make a change pass.
- Assets whose operating point varies (ships, variable-load machines) are monitored on residuals from `physics/baseline.py`, never on raw values.
- Synthetic simulators must be labelled as synthetic in reports; never present their numbers as real-world accuracy.
- New equipment types get: an asset-model factory in `assets.py`, a physics model in `physics/`, knowledge-base notes in `knowledge/`, tests.
- Claude API: model `claude-opus-5-5`, adaptive thinking, strict tool schemas, server-side fallbacks (see `agent/investigator.py`).
- Keep dependencies minimal (numpy/pandas/scipy/sklearn/fastapi/anthropic).
