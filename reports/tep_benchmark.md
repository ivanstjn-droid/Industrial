# Benchmark: Tennessee Eastman Process

Model: PCA monitor (T² + SPE), 31 components, α=0.99, persistence 3 samples. Trained on normal operation only (d00.dat).

- **False-alarm rate, normal operation (d00_te):** 3.4%
- **Mean detection rate** (excluding faults 3/9/15, considered practically undetectable in the literature): 80.2%
- **Root-cause ranking hit@5** (faults with literature root-cause variables): 7/8

| Fault | Description | Detection | FAR before fault | Delay (min) | Top-5 tags | Hit |
|---|---|---|---|---|---|---|
| 1 | A/C feed ratio, B composition constant (step) | 99.6% | 0.6% | 9 | XMEAS16, XMEAS20, XMEAS7, XMEAS13, XMV9 | ✗ |
| 2 | B composition, A/C ratio constant (step) | 98.8% | 1.9% | 30 | XMEAS30, XMEAS10, XMV6, XMEAS24, XMEAS16 | ✓ |
| 3 | D feed temperature (step) | 3.9% | 4.4% | 9 | – | – |
| 4 | Reactor cooling water inlet temperature (step) | 99.8% | 1.3% | 6 | XMV10, XMEAS9, XMEAS41, XMV7, XMEAS12 | ✓ |
| 5 | Condenser cooling water inlet temperature (step) | 29.0% | 1.3% | 6 | XMEAS18, XMEAS17, XMEAS11, XMEAS4, XMEAS19 | ✓ |
| 6 | A feed loss (step) | 99.8% | 0.6% | 6 | XMV3, XMEAS1, XMEAS20, XMV5, XMEAS16 | ✓ |
| 7 | C header pressure loss (step) | 99.8% | 0.0% | 6 | XMEAS7, XMEAS16, XMEAS13, XMV4, XMEAS20 | ✓ |
| 8 | A, B, C feed composition (random variation) | 97.9% | 0.6% | 51 | – | – |
| 9 | D feed temperature (random variation) | 3.4% | 5.0% | 12 | – | – |
| 10 | C feed temperature (random variation) | 64.9% | 1.3% | 0 | – | – |
| 11 | Reactor cooling water inlet temperature (random variation) | 73.9% | 3.7% | 21 | XMV10, XMEAS9, XMEAS21, XMEAS16, XMEAS31 | ✓ |
| 12 | Condenser cooling water inlet temperature (random variation) | 98.9% | 1.3% | 12 | – | – |
| 13 | Reaction kinetics (slow drift) | 95.3% | 0.6% | 114 | – | – |
| 14 | Reactor cooling water valve (sticking) | 99.9% | 0.6% | 3 | XMEAS21, XMV10, XMEAS9, XMEAS2, XMV1 | ✓ |
| 15 | Condenser cooling water valve (sticking) | 8.7% | 0.0% | 279 | – | – |
| 16 | Unknown | 59.1% | 1.9% | 48 | – | – |
| 17 | Unknown | 96.5% | 3.7% | 57 | – | – |
| 18 | Unknown | 90.0% | 1.3% | 231 | – | – |
| 19 | Unknown | 21.4% | 0.6% | 30 | – | – |
| 20 | Unknown | 63.5% | 0.0% | 246 | – | – |
| 21 | Stream 4 valve fixed at steady-state position | 56.4% | 3.7% | 0 | – | – |

Note: the root-cause variables are derived from the literature and are not an official ground truth.
