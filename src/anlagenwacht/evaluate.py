"""Benchmark harness on public datasets. Writes Markdown + JSON to ``reports/``.

    python -m anlagenwacht.evaluate tep
    python -m anlagenwacht.evaluate skab
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from anlagenwacht.detect import PCAMonitor, alarm_episodes

REPORTS = Path("reports")


# ----------------------------------------------------------------------- TEP
def evaluate_tep(var_explained: float = 0.90, alpha: float = 0.99, persistence: int = 3) -> dict:
    from anlagenwacht.datasets.tep import FAULT_ONSET_TEST, FAULTS, ROOT_CAUSE_TAGS, SAMPLE_MINUTES, load_tep

    ref = load_tep(0, "train").drop(columns="label")
    mon = PCAMonitor(var_explained=var_explained, alpha=alpha, persistence=persistence).fit(ref)
    normal = mon.score(load_tep(0, "test").drop(columns="label"))
    rows = []
    for f in FAULTS:
        df = load_tep(f, "test").drop(columns="label")
        sc = mon.score(df)
        contrib = mon.contributions(df)
        alarm = sc["alarm"].to_numpy()
        post = alarm[FAULT_ONSET_TEST:]
        first = int(np.argmax(post)) if post.any() else None
        row = {
            "fault": f,
            "description": FAULTS[f],
            "detection_rate": round(float(post.mean()), 3),
            "false_alarm_rate_prefault": round(float(alarm[:FAULT_ONSET_TEST].mean()), 3),
            "detection_delay_min": None if first is None else first * SAMPLE_MINUTES,
            "rca_top5": None,
            "rca_hit": None,
        }
        if f in ROOT_CAUSE_TAGS:
            eps = [e for e in alarm_episodes(sc, contrib) if e["end_pos"] >= FAULT_ONSET_TEST]
            if eps:
                top5 = [t["tag"] for t in eps[0]["top_tags"]]
                row["rca_top5"] = top5
                row["rca_hit"] = bool(set(top5) & ROOT_CAUSE_TAGS[f])
        rows.append(row)

    detectable = [r for r in rows if r["fault"] not in (3, 9, 15)]
    rca = [r for r in rows if r["rca_hit"] is not None]
    result = {
        "dataset": "Tennessee Eastman (Braatz)",
        "config": {"var_explained": var_explained, "alpha": alpha, "persistence": persistence,
                   "n_components": mon.n_components},
        "false_alarm_rate_normal_test": round(float(normal["alarm"].mean()), 3),
        "mean_detection_rate_excl_3_9_15": round(float(np.mean([r["detection_rate"] for r in detectable])), 3),
        "rca_hit_at_5": f"{sum(r['rca_hit'] for r in rca)}/{len(rca)}",
        "faults": rows,
    }
    _write("tep_benchmark", result, _tep_md(result))
    return result


def _tep_md(r: dict) -> str:
    lines = [
        "# Benchmark: Tennessee Eastman Process",
        "",
        f"Model: PCA monitor (T² + SPE), {r['config']['n_components']} components, α={r['config']['alpha']}, "
        f"persistence {r['config']['persistence']} samples. Trained on normal operation only (d00.dat).",
        "",
        f"- **False-alarm rate, normal operation (d00_te):** {r['false_alarm_rate_normal_test']:.1%}",
        f"- **Mean detection rate** (excluding faults 3/9/15, considered practically undetectable in the literature): "
        f"{r['mean_detection_rate_excl_3_9_15']:.1%}",
        f"- **Root-cause ranking hit@5** (faults with literature root-cause variables): {r['rca_hit_at_5']}",
        "",
        "| Fault | Description | Detection | FAR before fault | Delay (min) | Top-5 tags | Hit |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in r["faults"]:
        lines.append(
            f"| {f['fault']} | {f['description']} | {f['detection_rate']:.1%} | {f['false_alarm_rate_prefault']:.1%} | "
            f"{'–' if f['detection_delay_min'] is None else f['detection_delay_min']} | "
            f"{', '.join(f['rca_top5']) if f['rca_top5'] else '–'} | "
            f"{'–' if f['rca_hit'] is None else ('✓' if f['rca_hit'] else '✗')} |"
        )
    lines += ["", "Note: the root-cause variables are derived from the literature and are not an official ground truth."]
    return "\n".join(lines)


# ---------------------------------------------------------------------- SKAB
def evaluate_skab(train_rows: int = 400) -> dict:
    from anlagenwacht.datasets.skab import load_experiments
    from anlagenwacht.physics.pump import PumpEnergyModel
    from anlagenwacht.plant import SKAB_FEATURES

    raw_cols = SKAB_FEATURES[:-1]
    variants = {"pca_sensors": raw_cols, "pca_sensors_plus_physics": SKAB_FEATURES}
    totals = {v: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for v in variants}
    energy_rows = []
    experiments = load_experiments()
    for name, df in experiments.items():
        y = df["anomaly"].to_numpy()[train_rows:]
        energy = PumpEnergyModel().fit(df.iloc[:train_rows])
        full = energy.transform(df)
        for v, cols in variants.items():
            mon = PCAMonitor(var_explained=0.85, persistence=5).fit(full[cols].iloc[:train_rows])
            pred = mon.score(full[cols])["alarm"].to_numpy()[train_rows:]
            t = totals[v]
            t["tp"] += int(((pred == 1) & (y == 1)).sum())
            t["fp"] += int(((pred == 1) & (y == 0)).sum())
            t["fn"] += int(((pred == 0) & (y == 1)).sum())
            t["tn"] += int(((pred == 0) & (y == 0)).sum())
        res = full["P_el_residual"].to_numpy()[train_rows:]
        if (y == 1).any() and (y == 0).any():
            energy_rows.append({"experiment": name, "excess_w_anomaly": float(res[y == 1].mean()),
                                "excess_w_normal": float(res[y == 0].mean())})

    metrics = {}
    for v, t in totals.items():
        p = t["tp"] / max(t["tp"] + t["fp"], 1)
        r = t["tp"] / max(t["tp"] + t["fn"], 1)
        metrics[v] = {
            "precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(p + r, 1e-12), 3),
            "far": round(t["fp"] / max(t["fp"] + t["tn"], 1), 3), "mar": round(t["fn"] / max(t["fn"] + t["tp"], 1), 3),
        }
    e = pd.DataFrame(energy_rows)
    result = {
        "dataset": "SKAB (Skoltech Anomaly Benchmark)",
        "experiments": len(experiments),
        "train_rows_per_experiment": train_rows,
        "metrics": metrics,
        "energy": {
            "mean_excess_w_during_anomaly": round(float(e["excess_w_anomaly"].mean()), 2),
            "mean_excess_w_during_normal": round(float(e["excess_w_normal"].mean()), 2),
            "experiments_with_higher_draw_in_anomaly": f"{int((e['excess_w_anomaly'] > e['excess_w_normal']).sum())}/{len(e)}",
        },
    }
    lines = [
        "# Benchmark: SKAB pump testbed",
        "",
        f"{result['experiments']} experiments; each trained on its first {train_rows} samples (normal operation), "
        "scored point-wise on the remainder.",
        "",
        "| Variant | Precision | Recall | F1 | FAR | MAR |",
        "|---|---|---|---|---|---|",
    ] + [f"| {v} | {m['precision']} | {m['recall']} | {m['f1']} | {m['far']} | {m['mar']} |" for v, m in metrics.items()] + [
        "",
        "## Energy (physics model P_el ~ f(p·Q, Q))",
        f"- Mean excess power during anomalies: {result['energy']['mean_excess_w_during_anomaly']} W",
        f"- Mean excess power during normal operation (test part): {result['energy']['mean_excess_w_during_normal']} W",
        f"- Experiments with higher power draw during anomalies: {result['energy']['experiments_with_higher_draw_in_anomaly']}",
        "",
        "Note: the testbed is small (watt range); the method matters here, not the absolute values.",
    ]
    _write("skab_benchmark", result, "\n".join(lines))
    return result


def _write(stem: str, result: dict, md: str) -> None:
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"{stem}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPORTS / f"{stem}.md").write_text(md + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    which = (argv or sys.argv[1:] or ["all"])[0]
    if which in ("tep", "all"):
        r = evaluate_tep()
        print(f"TEP: FAR={r['false_alarm_rate_normal_test']}, DR={r['mean_detection_rate_excl_3_9_15']}, RCA@5={r['rca_hit_at_5']}")
    if which in ("skab", "all"):
        r = evaluate_skab()
        print("SKAB:", json.dumps(r["metrics"]), r["energy"])
    if which in ("maritime", "all"):
        from anlagenwacht.evaluate_maritime import main as maritime

        maritime()


if __name__ == "__main__":
    main()
