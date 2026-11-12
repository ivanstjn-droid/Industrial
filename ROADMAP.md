# Roadmap: MVP → pilot-ready

## Done (v0.1)
- PCA monitor (T²/SPE) with persistence filter, normalised contributions, alarm episodes
- Asset models with descriptions for TEP and SKAB
- Pump energy physics model with residual → kWh/€/CO₂
- Claude investigator with read-only, strict tools + rule-based offline fallback
- FastAPI + web UI, benchmark harness, CI

## Next: detection quality (weeks 1–4)
- [ ] Dynamic PCA / lagged features (expected to lift TEP faults 10, 16, 19–21)
- [ ] Time-series foundation-model forecaster (Chronos/TimesFM) as a second detector; compare in `evaluate.py`
- [ ] Reconstruction-based contributions to reduce smearing
- [ ] MetroPT-3 (compressed air) dataset + compressor specific-power model (kW per m³/min)
- [ ] District heating / CHP synthetic plant from OpenModelica TransiEnt; German energy prices from SMARD
- [ ] LLM evaluation: root-cause accuracy of Claude reports vs. rule-based on TEP faults (engineer-graded rubric)

## Maritime track (in parallel, demand validation first)
- [x] Naval gas-turbine wear benchmark (UCI CBM), vessel cooling/hull simulator, fuel → CO₂ → EU ETS cost
- [ ] 8–10 interviews with technical superintendents (Hamburg, Bremen, Leer): current monitoring, gaps, willingness to pay
- [ ] Connectors: ship data loggers / alarm & monitoring system exports, noon reports, AIS position and speed
- [ ] Merchant diesel engine models (SFOC vs. load, turbocharger, exhaust temperatures) and ISO 19030 hull performance
- [ ] On-board edge deployment with store-and-forward over satellite; IACS UR E26/E27 cyber-resilience documentation
- [ ] FuelEU Maritime GHG-intensity reporting alongside EU ETS

## Pilot readiness (weeks 4–10)
- [ ] Connectors: historian CSV import wizard, OPC UA (read-only, asyncua), MQTT
- [ ] TimescaleDB storage, streaming scoring, alarm persistence + engineer feedback (true/false alarm, confirmed cause)
- [ ] Asset-model import from tag lists / P&ID with Claude-suggested mappings, engineer confirmation
- [ ] Savings tracker and ISO 50001 / EnEfG report export
- [ ] Docker Compose on-prem install, SSO, audit log, role model
- [ ] Claude via EU region (AWS Bedrock Frankfurt / Vertex EU), configurable
- [ ] Security pack: data-flow diagram, IEC 62443 zone design, GDPR data-processing agreement (AVV) template
- [ ] Render Markdown reports in the UI; PDF export
- [ ] German localisation (UI, reports, knowledge base, AI prompts) selectable per site

## Pilot (8–12 weeks on site)
- One system (compressed-air station, pump group or heat supply), read-only access
- Success criteria agreed upfront: identified savings ≥ 5 % on target system, or ≥ N confirmed early detections at < X false alarms/week
- Weekly review with plant engineer; confirmed cases go into the evaluation set
