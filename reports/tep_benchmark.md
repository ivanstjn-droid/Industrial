# Benchmark: Tennessee-Eastman-Prozess

Modell: PCA-Monitor (T² + SPE), 31 Komponenten, α=0.99, Persistenz 3 Samples. Trainiert nur auf Normalbetrieb (d00.dat).

- **Fehlalarmrate Normalbetrieb (d00_te):** 3.4%
- **Mittlere Erkennungsrate** (ohne 3/9/15, in der Literatur als praktisch nicht erkennbar geltend): 80.2%
- **Ursachen-Ranking Treffer@5** (Störungen mit Literatur-Ursachenvariablen): 7/8

| Störung | Beschreibung | Erkennung | FAR vor Störung | Verzögerung (min) | Top-5 Tags | Treffer |
|---|---|---|---|---|---|---|
| 1 | A/C-Zulaufverhältnis, B-Anteil konstant (Sprung) | 99.6% | 0.6% | 9 | XMEAS16, XMEAS20, XMEAS7, XMEAS13, XMV9 | ✗ |
| 2 | B-Anteil, A/C-Verhältnis konstant (Sprung) | 98.8% | 1.9% | 30 | XMEAS30, XMEAS10, XMV6, XMEAS24, XMEAS16 | ✓ |
| 3 | D-Zulauftemperatur (Sprung) | 3.9% | 4.4% | 9 | – | – |
| 4 | Reaktor-Kühlwasser-Eintrittstemperatur (Sprung) | 99.8% | 1.3% | 6 | XMV10, XMEAS9, XMEAS41, XMV7, XMEAS12 | ✓ |
| 5 | Kondensator-Kühlwasser-Eintrittstemperatur (Sprung) | 29.0% | 1.3% | 6 | XMEAS18, XMEAS17, XMEAS11, XMEAS4, XMEAS19 | ✓ |
| 6 | Verlust A-Zulauf (Sprung) | 99.8% | 0.6% | 6 | XMV3, XMEAS1, XMEAS20, XMV5, XMEAS16 | ✓ |
| 7 | C-Sammelleitung Druckverlust (Sprung) | 99.8% | 0.0% | 6 | XMEAS7, XMEAS16, XMEAS13, XMV4, XMEAS20 | ✓ |
| 8 | A,B,C-Zulaufzusammensetzung (zufällig) | 97.9% | 0.6% | 51 | – | – |
| 9 | D-Zulauftemperatur (zufällig) | 3.4% | 5.0% | 12 | – | – |
| 10 | C-Zulauftemperatur (zufällig) | 64.9% | 1.3% | 0 | – | – |
| 11 | Reaktor-Kühlwasser-Eintrittstemperatur (zufällig) | 73.9% | 3.7% | 21 | XMV10, XMEAS9, XMEAS21, XMEAS16, XMEAS31 | ✓ |
| 12 | Kondensator-Kühlwasser-Eintrittstemperatur (zufällig) | 98.9% | 1.3% | 12 | – | – |
| 13 | Reaktionskinetik (langsame Drift) | 95.3% | 0.6% | 114 | – | – |
| 14 | Reaktor-Kühlwasserventil (Klemmen) | 99.9% | 0.6% | 3 | XMEAS21, XMV10, XMEAS9, XMEAS2, XMV1 | ✓ |
| 15 | Kondensator-Kühlwasserventil (Klemmen) | 8.7% | 0.0% | 279 | – | – |
| 16 | Unbekannt | 59.1% | 1.9% | 48 | – | – |
| 17 | Unbekannt | 96.5% | 3.7% | 57 | – | – |
| 18 | Unbekannt | 90.0% | 1.3% | 231 | – | – |
| 19 | Unbekannt | 21.4% | 0.6% | 30 | – | – |
| 20 | Unbekannt | 63.5% | 0.0% | 246 | – | – |
| 21 | Ventil Strom 4 in Ruhelage fixiert | 56.4% | 3.7% | 0 | – | – |

Hinweis: Die Ursachenvariablen sind aus der Literatur abgeleitet und keine offizielle Ground Truth.
