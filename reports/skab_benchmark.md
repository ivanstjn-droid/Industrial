# Benchmark: SKAB Pumpenprüfstand

34 Versuche; je Versuch Training auf den ersten 400 Samples (Normalbetrieb), Bewertung punktweise auf dem Rest.

| Variante | Precision | Recall | F1 | FAR | MAR |
|---|---|---|---|---|---|
| pca_sensoren | 0.699 | 0.772 | 0.733 | 0.385 | 0.228 |
| pca_sensoren_plus_physik | 0.704 | 0.771 | 0.736 | 0.376 | 0.229 |

## Energie (Physikmodell P_el ~ f(p·Q, Q))
- Mittlerer Mehrverbrauch während Anomalien: -0.51 W
- Mittlerer Mehrverbrauch im Normalbetrieb (Testteil): -2.68 W
- Versuche mit höherem Verbrauch während Anomalie: 17/34

Hinweis: Der Prüfstand ist klein (Watt-Bereich); relevant ist das Verfahren, nicht die absoluten Beträge.
