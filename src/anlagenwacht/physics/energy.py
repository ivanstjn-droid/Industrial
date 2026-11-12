"""Energy and cost translation – the numbers an Energiemanager reports under ISO 50001 / EnEfG."""

from __future__ import annotations

DEFAULT_PRICE_EUR_PER_KWH = 0.20  # rough German industrial electricity price incl. levies; configure per site
DEFAULT_HOURS_PER_YEAR = 6000
CO2_KG_PER_KWH = 0.38  # approx. German grid mix; configure per site / reporting year


def annualise(delta_w: float, hours_per_year: float = DEFAULT_HOURS_PER_YEAR,
              price_eur_per_kwh: float = DEFAULT_PRICE_EUR_PER_KWH,
              co2_kg_per_kwh: float = CO2_KG_PER_KWH) -> dict:
    kwh = delta_w / 1000.0 * hours_per_year
    return {
        "delta_kw": round(delta_w / 1000.0, 4),
        "kwh_per_year": round(kwh, 1),
        "eur_per_year": round(kwh * price_eur_per_kwh, 2),
        "t_co2_per_year": round(kwh * co2_kg_per_kwh / 1000.0, 3),
        "assumptions": {
            "hours_per_year": hours_per_year,
            "price_eur_per_kwh": price_eur_per_kwh,
            "co2_kg_per_kwh": co2_kg_per_kwh,
        },
    }
