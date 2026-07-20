# Installation

Use an isolated Python environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app_v14.py
```

For the optional pomegranate engine:

```bash
pip install -r requirements-pomegranate.txt
```

GraphRicciCurvature may require POT and networkit. The supplied requirements file
includes the project dependencies.
## Documentation map

1. [Introduction](README-01-Introduction.md)
2. [Installation](README-02-Installation.md)
3. [Architecture](README-03-Architecture.md)
4. [Mathematics](README-04-Mathematics.md)
5. [Regime Engines](README-05-Regime-Engines.md)
6. [Validation](README-06-Validation.md)
7. [Developer Guide](README-07-Developer.md)
8. [Research Limits and Extensions](README-08-Future.md)


---

[← Previous](README-01-Introduction.md) · [Home](../README.md) · [Next →](README-03-Architecture.md)

