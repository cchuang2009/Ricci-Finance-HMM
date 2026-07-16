# 2. Installation

[Home](../README.md) · [Introduction](README-01-Introduction.md) · [Installation](README-02-Installation.md) · [Architecture](README-03-Architecture.md) · [Mathematics](README-04-Mathematics.md) · [Regime Engines](README-05-Regime-Engines.md) · [Validation](README-06-Validation.md) · [Developer](README-07-Developer.md) · [Future](README-08-Future.md)

## Core installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app_v13.py
```

Run tests:

```bash
python -m pytest -q
python smoke_test_v13.py
```

## Optional pomegranate installation

```bash
pip install -r requirements-pomegranate.txt
```

pomegranate is optional. The app records an unavailable-engine message rather than terminating when it is not installed or when its API is incompatible with the local environment.

## Python compatibility

The core project uses NumPy, pandas, SciPy, scikit-learn, hmmlearn, NetworkX, Streamlit, Plotly, and GraphRicciCurvature. pomegranate adds PyTorch and may have a narrower compatibility range, so isolating it in a separate environment can be useful.

---

Previous: [Introduction](README-01-Introduction.md) · Next: [Architecture](README-03-Architecture.md)
