"""Asset model: plant hierarchy, tags, units and descriptions.

The asset model is what turns a ranked list of anonymous sensor columns into
something an operator can act on ("Reactor cooling water outlet temperature"
rather than "XMEAS21").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Tag:
    id: str
    description: str
    unit: str
    asset: str
    kind: str = "measurement"  # measurement | manipulated | derived


@dataclass
class AssetModel:
    name: str
    assets: dict[str, str]
    tags: dict[str, Tag] = field(default_factory=dict)

    def add(self, tag: Tag) -> None:
        if tag.asset not in self.assets:
            raise ValueError(f"unknown asset {tag.asset!r} for tag {tag.id}")
        self.tags[tag.id] = tag

    def tag(self, tag_id: str) -> Tag:
        return self.tags.get(tag_id) or Tag(tag_id, tag_id, "", "unknown")

    def tags_of(self, asset: str) -> list[Tag]:
        return [t for t in self.tags.values() if t.asset == asset]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "assets": [
                {"id": a, "description": d, "tags": [asdict(t) for t in self.tags_of(a)]}
                for a, d in self.assets.items()
            ],
        }


# --------------------------------------------------------------------------- TEP
_TEP_MEAS = [
    ("A feed (stream 1)", "kscmh", "feeds"),
    ("D feed (stream 2)", "kg/h", "feeds"),
    ("E feed (stream 3)", "kg/h", "feeds"),
    ("A and C feed (stream 4)", "kscmh", "feeds"),
    ("Recycle flow (stream 8)", "kscmh", "compressor"),
    ("Reactor feed rate (stream 6)", "kscmh", "reactor"),
    ("Reactor pressure", "kPa", "reactor"),
    ("Reactor level", "%", "reactor"),
    ("Reactor temperature", "°C", "reactor"),
    ("Purge rate (stream 9)", "kscmh", "separator"),
    ("Separator temperature", "°C", "separator"),
    ("Separator level", "%", "separator"),
    ("Separator pressure", "kPa", "separator"),
    ("Separator underflow (stream 10)", "m³/h", "separator"),
    ("Stripper level", "%", "stripper"),
    ("Stripper pressure", "kPa", "stripper"),
    ("Stripper underflow (stream 11)", "m³/h", "stripper"),
    ("Stripper temperature", "°C", "stripper"),
    ("Stripper steam flow", "kg/h", "stripper"),
    ("Compressor work", "kW", "compressor"),
    ("Reactor cooling water outlet temperature", "°C", "reactor"),
    ("Condenser cooling water outlet temperature", "°C", "separator"),
]
_TEP_ANALYSIS = (
    [("Reactor feed analysis " + c, "reactor") for c in "ABCDEF"]
    + [("Purge gas analysis " + c, "separator") for c in "ABCDEFGH"]
    + [("Product analysis " + c, "stripper") for c in "DEFGH"]
)
_TEP_MV = [
    ("D feed flow valve", "feeds"),
    ("E feed flow valve", "feeds"),
    ("A feed flow valve", "feeds"),
    ("A and C feed flow valve", "feeds"),
    ("Compressor recycle valve", "compressor"),
    ("Purge valve", "separator"),
    ("Separator liquid flow valve", "separator"),
    ("Stripper product flow valve", "stripper"),
    ("Stripper steam valve", "stripper"),
    ("Reactor cooling water flow", "reactor"),
    ("Condenser cooling water flow", "separator"),
]

TEP_TAGS: list[str] = [f"XMEAS{i}" for i in range(1, 42)] + [f"XMV{i}" for i in range(1, 12)]


def tep_asset_model() -> AssetModel:
    model = AssetModel(
        name="Tennessee Eastman Process (simulated chemical plant)",
        assets={
            "feeds": "Reactant feeds A, D, E, A+C",
            "reactor": "Exothermic reactor with cooling water loop",
            "separator": "Condenser and vapour-liquid separator",
            "compressor": "Recycle compressor",
            "stripper": "Stripper column (product recovery)",
        },
    )
    for i, (desc, unit, asset) in enumerate(_TEP_MEAS, start=1):
        model.add(Tag(f"XMEAS{i}", desc, unit, asset))
    for i, (desc, asset) in enumerate(_TEP_ANALYSIS, start=len(_TEP_MEAS) + 1):
        model.add(Tag(f"XMEAS{i}", desc, "mol%", asset))
    for i, (desc, asset) in enumerate(_TEP_MV, start=1):
        model.add(Tag(f"XMV{i}", desc, "%", asset, kind="manipulated"))
    return model


# -------------------------------------------------------------------------- SKAB
SKAB_TAGS = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
]


def skab_asset_model() -> AssetModel:
    model = AssetModel(
        name="SKAB pump testbed (water circulation loop)",
        assets={
            "pump": "Centrifugal pump with electric motor",
            "hydraulics": "Piping, valves, flow measurement",
            "energy": "Derived energy KPIs (physics model)",
        },
    )
    for tid, desc, unit, asset in [
        ("Accelerometer1RMS", "Pump vibration (RMS, sensor 1)", "g", "pump"),
        ("Accelerometer2RMS", "Pump vibration (RMS, sensor 2)", "g", "pump"),
        ("Current", "Motor current", "A", "pump"),
        ("Voltage", "Motor voltage", "V", "pump"),
        ("Temperature", "Motor temperature", "°C", "pump"),
        ("Thermocouple", "Fluid temperature", "°C", "hydraulics"),
        ("Pressure", "Discharge pressure", "bar", "hydraulics"),
        ("Volume Flow RateRMS", "Volume flow", "l/min", "hydraulics"),
    ]:
        model.add(Tag(tid, desc, unit, asset))
    for tid, desc, unit in [
        ("P_el", "Electrical power input (U·I)", "W"),
        ("P_hyd", "Hydraulic power (p·Q)", "W"),
        ("P_el_residual", "Excess power vs. physics/normal model", "W"),
    ]:
        model.add(Tag(tid, desc, unit, "energy", kind="derived"))
    return model


# ---------------------------------------------------------------- maritime
def naval_gt_asset_model() -> AssetModel:
    """Frigate gas-turbine propulsion (UCI CBM). Monitored values are deviations from
    what a healthy plant reads at the same speed."""
    model = AssetModel(
        name="Naval gas-turbine propulsion (CODLAG frigate, simulator)",
        assets={
            "compressor": "Gas-turbine compressor",
            "turbine": "HP / power turbine and exhaust",
            "fuel_system": "Fuel injection and fuel flow",
            "shafting": "Gearbox, shafts and propellers",
        },
    )
    note = " (vs. expected at current speed)"
    for tid, desc, unit, asset in [
        ("T2", "Compressor outlet air temperature", "°C", "compressor"),
        ("P2", "Compressor outlet air pressure", "bar", "compressor"),
        ("GGn", "Gas generator speed", "rpm", "compressor"),
        ("T48", "HP turbine exit temperature", "°C", "turbine"),
        ("P48", "HP turbine exit pressure", "bar", "turbine"),
        ("Pexh", "Exhaust gas pressure", "bar", "turbine"),
        ("GTT", "Gas turbine shaft torque", "kN·m", "turbine"),
        ("GTn", "Gas turbine speed", "rpm", "turbine"),
        ("TIC", "Turbine injection control", "%", "fuel_system"),
        ("mf", "Fuel flow", "kg/s", "fuel_system"),
        ("Ts", "Propeller torque", "kN", "shafting"),
    ]:
        model.add(Tag(tid, desc + note, unit, asset, kind="derived"))
    return model


def vessel_asset_model() -> AssetModel:
    """Merchant vessel propulsion and central cooling (synthetic simulator)."""
    model = AssetModel(
        name="Merchant vessel – propulsion and central cooling (simulated)",
        assets={
            "propulsion": "Main engine and propeller",
            "sea_water": "Sea-water pump, strainer and central cooler (SW side)",
            "lt_circuit": "Low-temperature fresh-water circuit with 3-way valve",
        },
    )
    note = " (vs. expected at current speed and sea temperature)"
    for tid, desc, unit, asset, kind in [
        ("ME_power", "Main engine shaft power", "kW", "propulsion", "measurement"),
        ("ME_fuel", "Main engine fuel consumption", "t/h", "propulsion", "measurement"),
        ("SW_flow", "Sea-water flow", "m³/h", "sea_water", "measurement"),
        ("SW_press", "Sea-water pressure after strainer", "bar", "sea_water", "measurement"),
        ("SW_pump_kw", "Sea-water pump power", "kW", "sea_water", "measurement"),
        ("SW_cooler_out", "Sea-water outlet temperature, central cooler", "°C", "sea_water", "measurement"),
        ("LT_valve", "LT 3-way valve position (to cooler)", "%", "lt_circuit", "manipulated"),
        ("LT_supply", "LT supply temperature", "°C", "lt_circuit", "measurement"),
        ("LT_return", "LT return temperature", "°C", "lt_circuit", "measurement"),
        ("LO_cooler_out", "Lube oil temperature after cooler", "°C", "lt_circuit", "measurement"),
    ]:
        model.add(Tag(tid, desc + note, unit, asset, kind=kind))
    return model
