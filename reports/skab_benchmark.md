# Benchmark: SKAB pump testbed

34 experiments; each trained on its first 400 samples (normal operation), scored point-wise on the remainder.

| Variant | Precision | Recall | F1 | FAR | MAR |
|---|---|---|---|---|---|
| pca_sensors | 0.699 | 0.772 | 0.733 | 0.385 | 0.228 |
| pca_sensors_plus_physics | 0.704 | 0.771 | 0.736 | 0.376 | 0.229 |

## Energy (physics model P_el ~ f(p·Q, Q))
- Mean excess power during anomalies: -0.51 W
- Mean excess power during normal operation (test part): -2.68 W
- Experiments with higher power draw during anomalies: 17/34

Note: the testbed is small (watt range); the method matters here, not the absolute values.
