# Ricci Finance v14 Final

Ricci Finance v14 is the consolidated end version of the project. The production
application and research notebook use **yfinance market data only**. Synthetic
data remains only in tests.

## Documentation

1. [Introduction](docs/README-01-Introduction.md)
2. [Installation](docs/README-02-Installation.md)
3. [Architecture](docs/README-03-Architecture.md)
4. [Mathematics](docs/README-04-Mathematics.md)
5. [Regime Engines](docs/README-05-Regime-Engines.md)
6. [Validation](docs/README-06-Validation.md)
7. [Developer Guide](docs/README-07-Developer.md)
8. [Research Limits and Extensions](docs/README-08-Future.md)


## Final pipeline

```text
yfinance OHLCV
    ↓
returns, volatility, dollar volume
    ↓
rolling similarity graph + capital weighting
    ↓
Ollivier–Ricci curvature
    ↓
graph observables
    ↓
hmmlearn Viterbi / posterior
duration-aware HSMM
optional pomegranate posterior
    ↓
engine benchmark and consensus
    ↓
3-D ticker/edge-labelled story animation
```

## Run the application

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_v14.py
```

Optional pomegranate engine:

```bash
pip install -r requirements-pomegranate.txt
```

## Run the research notebook

```bash
jupyter lab notebooks/ricci_finance_v14_yfinance_research.ipynb
```

## Final design rules

- Viterbi is the default story path.
- Posterior probabilities express confidence and transition uncertainty.
- HSMM tests explicit state duration.
- pomegranate remains optional and posterior-oriented when no compatible
  Viterbi path is available.
- Ricci flow is a geometric projection, not a price forecast.
- Node labels show tickers; edge labels can show curvature, correlation,
  distance, capital flow, or combined values.
