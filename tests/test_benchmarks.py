"""Regression guards on public datasets (downloads data; run with ``pytest -m network``)."""

import pytest

pytestmark = pytest.mark.network


def test_tep_benchmark_quality():
    from anlagenwacht.evaluate import evaluate_tep

    r = evaluate_tep()
    assert r["false_alarm_rate_normal_test"] <= 0.05
    assert r["mean_detection_rate_excl_3_9_15"] >= 0.75
    hits, total = map(int, r["rca_hit_at_5"].split("/"))
    assert hits / total >= 0.75


def test_skab_benchmark_quality():
    from anlagenwacht.evaluate import evaluate_skab

    r = evaluate_skab()
    assert r["metrics"]["pca_sensoren_plus_physik"]["f1"] >= 0.70
