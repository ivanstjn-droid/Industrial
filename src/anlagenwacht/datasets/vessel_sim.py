"""Synthetic merchant vessel: propulsion + central cooling system (physics-based).

Not real data. A transparent first-principles model used to (a) demo the
product on a ship and (b) score root-cause ranking where the true cause is known.

Model
-----
* Speed profile with sea passages and slow-steaming; propeller law
  P = MCR · (v / v_design)³ · hull_factor; fuel = P · SFOC(load).
* Heat to the low-temperature (LT) fresh-water circuit ∝ engine load + auxiliaries.
* Sea-water (SW) pump → strainer → central cooler (counter-flow, ε-NTU).
* 3-way thermostatic valve sends a fraction of LT water through the cooler to
  hold LT supply at 36 °C; when fully open the LT temperature rises.

Faults (start on day ``fault_day`` and ramp over ``ramp_days``)
---------------------------------------------------------------
* ``sw_strainer``    – clogging SW strainer: flow ↓, pressure after strainer ↓
* ``cooler_fouling`` – central cooler UA ↓ (fouling/biofilm)
* ``valve_stuck``    – 3-way valve stuck at its position on fault day
* ``hull_fouling``   – hull resistance ↑ 15 % → more power and fuel at same speed
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FAULTS = {
    "sw_strainer": "Sea-water strainer clogging",
    "cooler_fouling": "Central cooler fouling",
    "valve_stuck": "LT 3-way thermostatic valve stuck",
    "hull_fouling": "Hull fouling (added resistance)",
}
ROOT_CAUSE_TAGS = {
    "sw_strainer": {"SW_flow", "SW_press"},
    "cooler_fouling": {"LT_valve", "SW_cooler_out"},
    "valve_stuck": {"LT_valve", "LT_supply"},
    "hull_fouling": {"ME_power", "ME_fuel"},
}

MCR_KW = 12_000.0
V_DESIGN = 16.0
LT_SETPOINT = 36.0
C_PER_M3H = 1.16  # kW/K per m³/h of water (ρ·cp)


def _sfoc(load: np.ndarray) -> np.ndarray:
    return 168.0 + 30.0 * (load - 0.8) ** 2  # g/kWh, U-shaped around 80 % MCR


def _cooler(c_hot: float, c_cold: float, ua: float) -> float:
    cmin, cmax = min(c_hot, c_cold), max(c_hot, c_cold)
    cr, ntu = cmin / cmax, ua / cmin
    if abs(1 - cr) < 1e-9:
        return ntu / (1 + ntu)
    e = np.exp(-ntu * (1 - cr))
    return (1 - e) / (1 - cr * e)


def simulate_vessel(fault: str | None = None, days: int = 60, fault_day: int = 30, ramp_days: float = 10,
                    seed: int = 0) -> pd.DataFrame:
    if fault is not None and fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}")
    rng = np.random.default_rng(seed)
    n = days * 24
    t = np.arange(n) / 24.0  # days
    # speed: passages at 13–15 kn with occasional slow steaming legs
    leg = (t // 4).astype(int)
    leg_speed = rng.choice([11.0, 13.0, 14.0, 15.0], size=leg.max() + 1, p=[0.15, 0.3, 0.35, 0.2])
    leg_speed[:4] = rng.permutation([11.0, 13.0, 14.0, 15.0])  # every speed band seen early (baseline period)
    speed = leg_speed[leg] + rng.normal(0, 0.15, n)
    # sea temperature changes with trading area / weather (~8-day cycle) rather than a slow season
    sw_temp = 18 + 4 * np.sin(2 * np.pi * t / 8) + 1.0 * np.sin(2 * np.pi * t / 1.3) + rng.normal(0, 0.2, n)

    ramp = np.clip((t - fault_day) / ramp_days, 0, 1) if fault else np.zeros(n)
    hull = 1 + (0.15 * ramp if fault == "hull_fouling" else 0)
    strainer = 1 - (0.45 * ramp if fault == "sw_strainer" else 0)
    ua_factor = 1 - (0.55 * ramp if fault == "cooler_fouling" else 0)

    power = np.clip(MCR_KW * (speed / V_DESIGN) ** 3 * hull, 0.1 * MCR_KW, MCR_KW)
    load = power / MCR_KW
    fuel = power * _sfoc(load) / 1e6 * (1 + rng.normal(0, 0.005, n))  # t/h

    heat_lt = 0.22 * power + 400.0  # kW into LT circuit (jacket via HT, lube oil, charge air, aux)
    lt_flow = 300.0  # m³/h
    c_lt = lt_flow * C_PER_M3H
    sw_flow = 420.0 * strainer + rng.normal(0, 3, n)
    sw_press = 2.4 * strainer**2 + rng.normal(0, 0.02, n)  # bar after strainer
    sw_pump_kw = 38.0 * (0.75 + 0.25 * strainer) + rng.normal(0, 0.4, n)
    ua = 260.0 * ua_factor  # kW/K

    lt_supply = np.empty(n); lt_return = np.empty(n); valve = np.empty(n); sw_out = np.empty(n)
    stuck_at = None
    for i in range(n):
        t_ret = None
        c_sw = sw_flow[i] * C_PER_M3H

        def supply(x: float) -> tuple[float, float, float]:
            # fixed point: T_ret = T_sup + Q/C_lt ; cooler sees x·C_lt on hot side
            t_sup = LT_SETPOINT
            for _ in range(30):
                t_r = t_sup + heat_lt[i] / c_lt
                ch = max(x, 1e-6) * c_lt
                eps = _cooler(ch, c_sw, ua[i] if np.ndim(ua) else ua)
                q = eps * min(ch, c_sw) * (t_r - sw_temp[i])
                t_cool_out = t_r - q / ch
                t_new = x * t_cool_out + (1 - x) * t_r
                if abs(t_new - t_sup) < 1e-6:
                    break
                t_sup = t_new
            return t_sup, t_r, sw_temp[i] + q / c_sw

        if fault == "valve_stuck" and t[i] >= fault_day and stuck_at is not None:
            x = stuck_at
        else:
            lo, hi = 0.0, 1.0
            if supply(1.0)[0] > LT_SETPOINT:
                x = 1.0
            else:
                for _ in range(40):
                    mid = (lo + hi) / 2
                    if supply(mid)[0] > LT_SETPOINT:
                        lo = mid
                    else:
                        hi = mid
                x = hi
            if fault == "valve_stuck" and t[i] >= fault_day:
                stuck_at = x
        lt_supply[i], lt_return[i], sw_out[i] = supply(x)
        valve[i] = x

    df = pd.DataFrame(
        {
            "speed": speed,
            "SW_temp": sw_temp,
            "ME_power": power * (1 + rng.normal(0, 0.005, n)),
            "ME_fuel": fuel,
            "SW_flow": sw_flow,
            "SW_press": sw_press,
            "SW_pump_kw": sw_pump_kw,
            "SW_cooler_out": sw_out + rng.normal(0, 0.1, n),
            "LT_valve": 100 * valve + rng.normal(0, 0.5, n),
            "LT_supply": lt_supply + rng.normal(0, 0.1, n),
            "LT_return": lt_return + rng.normal(0, 0.1, n),
            "LO_cooler_out": lt_supply + 7 + 6 * load + rng.normal(0, 0.15, n),
        },
        index=pd.date_range("2026-03-01", periods=n, freq="1h", name="time"),
    )
    df["label"] = (ramp > 0.05).astype(int) if fault else 0
    return df
