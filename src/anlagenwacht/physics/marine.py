"""Fuel → CO₂ → EU ETS cost for shipping.

Emission factors are the IMO default carbon factors (t CO₂ per t fuel).
The EU ETS covers 100 % of emissions on voyages within the EU/EEA and at berth
in EU ports, and 50 % of emissions on voyages into or out of the EU/EEA. Shipping
companies surrender allowances for a growing share of emissions: 40 % for 2024,
70 % for 2025, 100 % from 2026. Allowance price and the voyage mix are inputs,
not facts: configure per fleet. (From 2026 methane and N₂O are also covered,
which this simple calculation ignores.)
"""

from __future__ import annotations

CO2_FACTOR_T_PER_T = {"HFO": 3.114, "LFO": 3.151, "MGO": 3.206, "LNG": 2.750}
ETS_PHASE_IN = {2024: 0.40, 2025: 0.70}
DEFAULT_EUA_PRICE_EUR = 75.0
DEFAULT_FUEL_PRICE_EUR_PER_T = {"HFO": 450.0, "LFO": 550.0, "MGO": 700.0, "LNG": 600.0}


def fuel_cost_and_emissions(
    fuel_t: float,
    fuel: str = "HFO",
    eu_share: float = 0.5,
    year: int = 2026,
    eua_price_eur: float = DEFAULT_EUA_PRICE_EUR,
    fuel_price_eur_per_t: float | None = None,
) -> dict:
    """Cost of an amount of fuel including EU ETS allowances.

    ``eu_share`` is the share of emissions covered by the EU ETS for this
    vessel's trading pattern (1.0 intra-EU, 0.5 extra-EU voyages, 0 outside).
    """
    fuel = fuel.upper()
    if fuel not in CO2_FACTOR_T_PER_T:
        raise ValueError(f"unknown fuel {fuel!r}; use one of {sorted(CO2_FACTOR_T_PER_T)}")
    co2 = fuel_t * CO2_FACTOR_T_PER_T[fuel]
    phase = ETS_PHASE_IN.get(year, 1.0 if year >= 2026 else 0.0)
    euas = co2 * eu_share * phase
    price = DEFAULT_FUEL_PRICE_EUR_PER_T[fuel] if fuel_price_eur_per_t is None else fuel_price_eur_per_t
    fuel_eur = fuel_t * price
    ets_eur = euas * eua_price_eur
    return {
        "fuel_t": round(fuel_t, 3),
        "co2_t": round(co2, 3),
        "eu_allowances_t": round(euas, 3),
        "fuel_cost_eur": round(fuel_eur, 2),
        "ets_cost_eur": round(ets_eur, 2),
        "total_cost_eur": round(fuel_eur + ets_eur, 2),
        "assumptions": {
            "fuel": fuel,
            "co2_factor": CO2_FACTOR_T_PER_T[fuel],
            "eu_share": eu_share,
            "ets_phase_in": phase,
            "eua_price_eur": eua_price_eur,
            "fuel_price_eur_per_t": price,
        },
    }
