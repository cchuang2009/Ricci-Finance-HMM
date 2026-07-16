# Ricci Finance v13

Ricci Finance v13 extends the v12 Ricci-capital manifold with interchangeable regime engines, decoder comparison, model consensus, validation export, and a research notebook.

## Documentation

1. [Introduction](docs/README-01-Introduction.md)
2. [Installation](docs/README-02-Installation.md)
3. [Architecture](docs/README-03-Architecture.md)
4. [Mathematics](docs/README-04-Mathematics.md)
5. [Regime Engines](docs/README-05-Regime-Engines.md)
6. [Validation](docs/README-06-Validation.md)
7. [Developer Guide](docs/README-07-Developer.md)
8. [Future Work](docs/README-08-Future.md)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_v13.py
```

To enable the optional pomegranate engine:

```bash
pip install -r requirements-pomegranate.txt
```

## Main design decision

- **hmmlearn Viterbi**: default coherent regime storyline.
- **hmmlearn posterior**: confidence and transition warning.
- **HSMM**: explicit-duration research engine.
- **pomegranate**: optional posterior-only engine; it is not treated as a Viterbi replacement.
- **Consensus**: agreement/disagreement across available engines.

## Notebook

Open [`notebooks/ricci_finance_v13_regime_benchmark.ipynb`](notebooks/ricci_finance_v13_regime_benchmark.ipynb) to compare synthetic accuracy, duration stability, runtime, HMM decoding, HSMM, and pomegranate.

## Animation labels

The v13 animation now displays ticker names directly beside every node. Edge midpoint labels can show Ricci curvature, correlation, effective distance, capital flow, or a combined `rho + d + kappa` label. The sidebar controls how many edges are labeled to prevent excessive visual overlap.
