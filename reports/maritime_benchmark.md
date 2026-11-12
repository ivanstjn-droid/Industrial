# Benchmark: maritime

## A. Naval gas-turbine wear (real public simulator data, UCI CBM, CC BY 4.0)

Sensors are compared with what a healthy plant reads at the same speed (operating-point baseline), then monitored with PCA. Simulator output is noise-free, so 0.2 % Gaussian measurement noise is added.

- False-alarm rate on held-out healthy points: 1.9%
- Wear estimation from sensors (gradient boosting, 25 % test split): compressor kMc MAE 0.00234 (R² 0.9575), turbine kMt MAE 0.0013 (R² 0.9496)

| Compressor decay | Detection | | Turbine decay | Detection |
|---|---|---|---|---|
| 1.0 % | 11% | | 0.5 % | 11% |
| 2.0 % | 56% | | 1.0 % | 0% |
| 3.0 % | 78% | | 1.5 % | 33% |
| 4.0 % | 100% | | 2.0 % | 56% |
| 5.0 % | 100% | | 2.5 % | 56% |

Fuel penalty at ≥ 12 kn versus a healthy plant:

| kMc | kMt | Extra fuel |
|---|---|---|
| 0.99 | 1.0 | +0.02 % |
| 0.98 | 1.0 | +0.96 % |
| 0.97 | 1.0 | +1.81 % |
| 0.96 | 1.0 | +2.76 % |
| 0.95 | 1.0 | +3.67 % |
| 1.0 | 0.99 | +1.28 % |
| 1.0 | 0.98 | +3.14 % |
| 1.0 | 0.975 | +4.31 % |
| 0.97 | 0.985 | +4.75 % |

**Cost example** (15 kn, compressor decay 3 %, 3000 h/yr, MGO, 100 % EU ETS): +0.0714 t/h → 214.111 t fuel/yr, 686.44 t CO₂/yr, €149,878 fuel + €51,483 EU ETS = **€201,361/yr** (assumptions: {'fuel': 'MGO', 'co2_factor': 3.206, 'eu_share': 1.0, 'ets_phase_in': 1.0, 'eua_price_eur': 75.0, 'fuel_price_eur_per_t': 700.0}).

## B. Merchant vessel – propulsion and central cooling (synthetic physics simulator)

60 days hourly; baseline learned on days 0–20; fault starts day 30 and ramps over 10 days.

- False-alarm rate, fault-free run (days 20–60): 0.0%
- Root cause in top-3 tags: 4/4

| Fault | FAR before | Delay (h) | Detection after ramp | Top-3 tags | Hit |
|---|---|---|---|---|---|
| Sea-water strainer clogging | 0.0% | 14 | 100% | SW_press, SW_flow, SW_cooler_out | ✓ |
| Central cooler fouling | 0.0% | 59 | 86% | LT_valve, SW_press, SW_flow | ✓ |
| LT 3-way thermostatic valve stuck | 0.0% | 6 | 100% | LT_supply, LT_return, LO_cooler_out | ✓ |
| Hull fouling (added resistance) | 0.0% | 34 | 100% | ME_power, ME_fuel, LT_valve | ✓ |

**Hull fouling cost if left untreated** (+0.1669 t/h, 6500 h/yr, HFO, 50 % EU ETS share): 1084.814 t fuel, 3378.109 t CO₂, €488,166 fuel + €126,679 EU ETS = **€614,845/yr**.

Notes: part A is a naval gas turbine, not a merchant diesel engine – it demonstrates the method on real public data. Part B is synthetic and only proves the pipeline works end to end; real vessel data is needed before any accuracy claim. Fuel and allowance prices are configurable assumptions.
