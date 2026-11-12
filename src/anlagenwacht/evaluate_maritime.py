"""Maritime benchmarks → reports/maritime_benchmark.{md,json}.

    python -m anlagenwacht.evaluate_maritime
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from anlagenwacht.detect import PCAMonitor
from anlagenwacht.evaluate import _write
from anlagenwacht.physics.marine import fuel_cost_and_emissions


# --------------------------------------------------------------- naval (real)
def evaluate_naval(seed: int = 0) -> dict:
    from anlagenwacht.datasets.naval import SENSORS, healthy_mask, load_naval
    from anlagenwacht.physics.baseline import OperatingPointBaseline

    df = load_naval(seed=seed)
    healthy = df[healthy_mask(df)]
    fit_idx = healthy.sample(frac=0.5, random_state=seed).index
    holdout = healthy.drop(fit_idx)
    base = OperatingPointBaseline(["v"], degree=4).fit(df.loc[fit_idx], SENSORS)
    resid = base.residuals(df)
    mon = PCAMonitor(var_explained=0.95).fit(resid.loc[fit_idx])
    flagged = mon.score(resid)["score"] > 1.0  # static grid points: no persistence filter

    far = float(flagged.loc[holdout.index].mean())
    by_c = []
    for d in [0.01, 0.02, 0.03, 0.04, 0.05]:
        m = (df["kMt"] == 1.0) & np.isclose(1 - df["kMc"], d)
        by_c.append({"compressor_decay_pct": round(d * 100, 1), "detection_rate": round(float(flagged[m].mean()), 3)})
    by_t = []
    for d in [0.005, 0.01, 0.015, 0.02, 0.025]:
        m = (df["kMc"] == 1.0) & np.isclose(1 - df["kMt"], d)
        by_t.append({"turbine_decay_pct": round(d * 100, 1), "detection_rate": round(float(flagged[m].mean()), 3)})

    # decay-state estimation from speed + residuals
    X = pd.concat([df[["v"]], resid], axis=1)
    est = {}
    for target in ["kMc", "kMt"]:
        Xtr, Xte, ytr, yte = train_test_split(X, df[target], test_size=0.25, random_state=seed)
        model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, random_state=seed).fit(Xtr, ytr)
        pred = model.predict(Xte)
        est[target] = {"mae": round(float(mean_absolute_error(yte, pred)), 5), "r2": round(float(r2_score(yte, pred)), 4)}

    # fuel penalty: expected (healthy) vs actual fuel flow per decay state
    excess = (df["mf"] - base.expected(df)["mf"]) / base.expected(df)["mf"]
    penalty = []
    for kmc, kmt in [(0.99, 1.0), (0.98, 1.0), (0.97, 1.0), (0.96, 1.0), (0.95, 1.0), (1.0, 0.99), (1.0, 0.98),
                     (1.0, 0.975), (0.97, 0.985)]:
        m = np.isclose(df["kMc"], kmc) & np.isclose(df["kMt"], kmt) & (df["v"] >= 12)
        penalty.append({"kMc": kmc, "kMt": kmt, "fuel_penalty_pct": round(float(excess[m].mean() * 100), 2)})

    # money example: 15 kn cruising, compressor 3 % decayed, 3000 h/yr, MGO, intra-EU
    m15 = (df["v"] == 15) & np.isclose(df["kMc"], 0.97) & (df["kMt"] == 1.0)
    extra_t_h = float((df["mf"] - base.expected(df)["mf"])[m15].mean()) * 3.6
    example = fuel_cost_and_emissions(max(extra_t_h, 0) * 3000, "MGO", eu_share=1.0)
    return {"false_alarm_rate_healthy_holdout": round(far, 3), "detection_by_compressor_decay": by_c,
            "detection_by_turbine_decay": by_t, "decay_estimation": est, "fuel_penalty": penalty,
            "cost_example": {"scenario": "15 kn, compressor decay 3 %, 3000 h/yr, MGO, 100 % EU ETS",
                             "extra_fuel_t_per_h": round(extra_t_h, 4), **example}}


# -------------------------------------------------------- vessel (synthetic)
def evaluate_vessel() -> dict:
    from anlagenwacht.datasets.vessel_sim import FAULTS, ROOT_CAUSE_TAGS
    from anlagenwacht.plant import vessel_plant

    rows = []
    clean = vessel_plant(None)
    far_clean = float(clean.scores["alarm"].iloc[20 * 24:].mean())
    for fault, desc in FAULTS.items():
        p = vessel_plant(fault)
        a = p.scores["alarm"].to_numpy()
        onset = 30 * 24
        post = a[onset:]
        first = int(np.argmax(post)) if post.any() else None
        eps = [e for e in p.episodes if e["end_pos"] >= onset]
        top3 = [t["tag"] for t in eps[0]["top_tags"][:3]] if eps else []
        row = {
            "fault": fault, "description": desc,
            "false_alarm_rate_before_fault": round(float(a[20 * 24:onset].mean()), 3),
            "detection_delay_h": first,
            "detection_rate_after_ramp": round(float(a[40 * 24:].mean()), 3),
            "top3_tags": top3,
            "rca_hit_top3": bool(set(top3) & ROOT_CAUSE_TAGS[fault]),
        }
        if fault == "hull_fouling" and eps:
            row["cost_if_persistent"] = p.energy_impact(eps[-1]["id"])
        rows.append(row)
    return {"false_alarm_rate_no_fault_run": round(far_clean, 3), "faults": rows,
            "rca_hit_top3": f"{sum(r['rca_hit_top3'] for r in rows)}/{len(rows)}"}


def _md(naval: dict, vessel: dict) -> str:
    L = ["# Benchmark: maritime", "",
         "## A. Naval gas-turbine wear (real public simulator data, UCI CBM, CC BY 4.0)", "",
         "Sensors are compared with what a healthy plant reads at the same speed (operating-point baseline), then "
         "monitored with PCA. Simulator output is noise-free, so 0.2 % Gaussian measurement noise is added.", "",
         f"- False-alarm rate on held-out healthy points: {naval['false_alarm_rate_healthy_holdout']:.1%}",
         f"- Wear estimation from sensors (gradient boosting, 25 % test split): compressor kMc MAE "
         f"{naval['decay_estimation']['kMc']['mae']} (R² {naval['decay_estimation']['kMc']['r2']}), turbine kMt MAE "
         f"{naval['decay_estimation']['kMt']['mae']} (R² {naval['decay_estimation']['kMt']['r2']})", "",
         "| Compressor decay | Detection | | Turbine decay | Detection |", "|---|---|---|---|---|"]
    for c, t in zip(naval["detection_by_compressor_decay"], naval["detection_by_turbine_decay"]):
        L.append(f"| {c['compressor_decay_pct']} % | {c['detection_rate']:.0%} | | {t['turbine_decay_pct']} % | {t['detection_rate']:.0%} |")
    L += ["", "Fuel penalty at ≥ 12 kn versus a healthy plant:", "", "| kMc | kMt | Extra fuel |", "|---|---|---|"]
    L += [f"| {p['kMc']} | {p['kMt']} | {p['fuel_penalty_pct']:+.2f} % |" for p in naval["fuel_penalty"]]
    ex = naval["cost_example"]
    L += ["", f"**Cost example** ({ex['scenario']}): +{ex['extra_fuel_t_per_h']} t/h → {ex['fuel_t']} t fuel/yr, "
          f"{ex['co2_t']} t CO₂/yr, €{ex['fuel_cost_eur']:,.0f} fuel + €{ex['ets_cost_eur']:,.0f} EU ETS = "
          f"**€{ex['total_cost_eur']:,.0f}/yr** (assumptions: {ex['assumptions']}).", "",
          "## B. Merchant vessel – propulsion and central cooling (synthetic physics simulator)", "",
          "60 days hourly; baseline learned on days 0–20; fault starts day 30 and ramps over 10 days.", "",
          f"- False-alarm rate, fault-free run (days 20–60): {vessel['false_alarm_rate_no_fault_run']:.1%}",
          f"- Root cause in top-3 tags: {vessel['rca_hit_top3']}", "",
          "| Fault | FAR before | Delay (h) | Detection after ramp | Top-3 tags | Hit |", "|---|---|---|---|---|---|"]
    for r in vessel["faults"]:
        L.append(f"| {r['description']} | {r['false_alarm_rate_before_fault']:.1%} | {r['detection_delay_h']} | "
                 f"{r['detection_rate_after_ramp']:.0%} | {', '.join(r['top3_tags'])} | {'✓' if r['rca_hit_top3'] else '✗'} |")
    hull = next(r for r in vessel["faults"] if r["fault"] == "hull_fouling").get("cost_if_persistent")
    if hull:
        a = hull["annualised_if_persistent"]
        L += ["", f"**Hull fouling cost if left untreated** (+{hull['excess_fuel_t_per_h']} t/h, {hull['hours_per_year']} h/yr, "
              f"HFO, 50 % EU ETS share): {a['fuel_t']} t fuel, {a['co2_t']} t CO₂, €{a['fuel_cost_eur']:,.0f} fuel + "
              f"€{a['ets_cost_eur']:,.0f} EU ETS = **€{a['total_cost_eur']:,.0f}/yr**."]
    L += ["", "Notes: part A is a naval gas turbine, not a merchant diesel engine – it demonstrates the method on real "
          "public data. Part B is synthetic and only proves the pipeline works end to end; real vessel data is needed "
          "before any accuracy claim. Fuel and allowance prices are configurable assumptions."]
    return "\n".join(L)


def main() -> None:
    naval, vessel = evaluate_naval(), evaluate_vessel()
    _write("maritime_benchmark", {"naval": naval, "vessel": vessel}, _md(naval, vessel))
    print("naval:", naval["false_alarm_rate_healthy_holdout"], naval["decay_estimation"],
          naval["detection_by_compressor_decay"], naval["detection_by_turbine_decay"])
    print("vessel:", vessel["false_alarm_rate_no_fault_run"], vessel["rca_hit_top3"])


if __name__ == "__main__":
    main()
