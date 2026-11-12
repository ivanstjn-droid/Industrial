"""Asset model: plant hierarchy, tags, units and descriptions.

The asset model is what turns a ranked list of anonymous sensor columns into
something an operator can act on ("Reaktor-Kühlwasser-Austrittstemperatur"
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
        return self.tags.get(tag_id) or Tag(tag_id, tag_id, "", "unbekannt")

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
    ("A-Zulauf (Strom 1)", "kscmh", "zulauf"),
    ("D-Zulauf (Strom 2)", "kg/h", "zulauf"),
    ("E-Zulauf (Strom 3)", "kg/h", "zulauf"),
    ("A+C-Zulauf (Strom 4)", "kscmh", "zulauf"),
    ("Rückführstrom (Strom 8)", "kscmh", "kompressor"),
    ("Reaktor-Zulaufrate (Strom 6)", "kscmh", "reaktor"),
    ("Reaktordruck", "kPa", "reaktor"),
    ("Reaktorfüllstand", "%", "reaktor"),
    ("Reaktortemperatur", "°C", "reaktor"),
    ("Spülgasmenge (Strom 9)", "kscmh", "abscheider"),
    ("Abscheider-Temperatur", "°C", "abscheider"),
    ("Abscheider-Füllstand", "%", "abscheider"),
    ("Abscheider-Druck", "kPa", "abscheider"),
    ("Abscheider-Unterlauf (Strom 10)", "m³/h", "abscheider"),
    ("Stripper-Füllstand", "%", "stripper"),
    ("Stripper-Druck", "kPa", "stripper"),
    ("Stripper-Unterlauf (Strom 11)", "m³/h", "stripper"),
    ("Stripper-Temperatur", "°C", "stripper"),
    ("Stripper-Dampfmenge", "kg/h", "stripper"),
    ("Kompressorleistung", "kW", "kompressor"),
    ("Reaktor-Kühlwasser-Austrittstemperatur", "°C", "reaktor"),
    ("Kondensator-Kühlwasser-Austrittstemperatur", "°C", "abscheider"),
]
_TEP_ANALYSIS = (
    [("Reaktorzulauf-Analyse " + c, "reaktor") for c in "ABCDEF"]
    + [("Spülgas-Analyse " + c, "abscheider") for c in "ABCDEFGH"]
    + [("Produkt-Analyse " + c, "stripper") for c in "DEFGH"]
)
_TEP_MV = [
    ("D-Zulauf Stellwert", "zulauf"),
    ("E-Zulauf Stellwert", "zulauf"),
    ("A-Zulauf Stellwert", "zulauf"),
    ("A+C-Zulauf Stellwert", "zulauf"),
    ("Kompressor-Rückführventil", "kompressor"),
    ("Spülventil", "abscheider"),
    ("Abscheider-Flüssigkeitsabzug", "abscheider"),
    ("Stripper-Produktabzug", "stripper"),
    ("Stripper-Dampfventil", "stripper"),
    ("Reaktor-Kühlwassermenge", "reaktor"),
    ("Kondensator-Kühlwassermenge", "abscheider"),
]

TEP_TAGS: list[str] = [f"XMEAS{i}" for i in range(1, 42)] + [f"XMV{i}" for i in range(1, 12)]


def tep_asset_model() -> AssetModel:
    model = AssetModel(
        name="Tennessee-Eastman-Prozess (Chemieanlage, Simulation)",
        assets={
            "zulauf": "Edukt-Zuläufe A, D, E, A+C",
            "reaktor": "Exothermer Reaktor mit Kühlwasserkreis",
            "abscheider": "Kondensator und Dampf-Flüssig-Abscheider",
            "kompressor": "Rückführkompressor",
            "stripper": "Stripper-Kolonne (Produktaufbereitung)",
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
        name="SKAB Pumpenprüfstand (Wasserkreislauf)",
        assets={
            "pumpe": "Kreiselpumpe mit Elektromotor",
            "hydraulik": "Rohrleitung, Ventile, Durchflussmessung",
            "energie": "Abgeleitete Energiekennzahlen (Physikmodell)",
        },
    )
    for tid, desc, unit, asset in [
        ("Accelerometer1RMS", "Schwingung Pumpe (RMS, Sensor 1)", "g", "pumpe"),
        ("Accelerometer2RMS", "Schwingung Pumpe (RMS, Sensor 2)", "g", "pumpe"),
        ("Current", "Motorstrom", "A", "pumpe"),
        ("Voltage", "Motorspannung", "V", "pumpe"),
        ("Temperature", "Motortemperatur", "°C", "pumpe"),
        ("Thermocouple", "Fluidtemperatur", "°C", "hydraulik"),
        ("Pressure", "Förderdruck", "bar", "hydraulik"),
        ("Volume Flow RateRMS", "Volumenstrom", "l/min", "hydraulik"),
    ]:
        model.add(Tag(tid, desc, unit, asset))
    for tid, desc, unit in [
        ("P_el", "Elektrische Leistungsaufnahme (U·I)", "W"),
        ("P_hyd", "Hydraulische Leistung (p·Q)", "W"),
        ("P_el_residual", "Mehrverbrauch ggü. Physik-/Normalmodell", "W"),
    ]:
        model.add(Tag(tid, desc, unit, "energie", kind="derived"))
    return model
