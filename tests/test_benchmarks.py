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
    assert r["metrics"]["pca_sensors_plus_physics"]["f1"] >= 0.70


def test_maritime_benchmark_quality():
    from anlagenwacht.evaluate_maritime import evaluate_naval

    r = evaluate_naval()
    assert r["false_alarm_rate_healthy_holdout"] <= 0.05
    assert r["detection_by_compressor_decay"][-1]["detection_rate"] >= 0.9  # 5 % compressor decay
    assert r["decay_estimation"]["kMc"]["r2"] >= 0.9
