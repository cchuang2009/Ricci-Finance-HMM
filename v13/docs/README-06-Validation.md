# 6. Validation

[Home](../README.md) · [Introduction](README-01-Introduction.md) · [Installation](README-02-Installation.md) · [Architecture](README-03-Architecture.md) · [Mathematics](README-04-Mathematics.md) · [Regime Engines](README-05-Regime-Engines.md) · [Validation](README-06-Validation.md) · [Developer](README-07-Developer.md) · [Future](README-08-Future.md)

## Synthetic validation

Known regimes permit accuracy, macro F1, adjusted Rand index, confusion matrix, switch rate, duration distribution, posterior entropy, and runtime.

Hidden-state numbers are arbitrary, so predicted states must be aligned before supervised scoring.

## Real Ricci-feature validation

Real markets have no ground-truth regime labels. v13 therefore measures walk-forward stability, duration statistics, Viterbi/posterior agreement, engine consensus, likelihood when available, separation of subsequent returns, and sensitivity to graph parameters.

## Avoiding leakage

Training and scaling must use only information available at the evaluation date. The included notebook is a research starting point; production validation should fit the scaler and engine separately inside every walk-forward fold.

## Export

The Streamlit app exports graph features and primary regime assignments as `v13_regime_features.csv` for external statistical testing.

---

Previous: [Regime Engines](README-05-Regime-Engines.md) · Next: [Developer Guide](README-07-Developer.md)
