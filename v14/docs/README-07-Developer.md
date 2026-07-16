# Developer Guide

Primary entry points:

- `app_v14.py`
- `notebooks/ricci_finance_v14_yfinance_research.ipynb`
- `ricci_finance/regime_engines/`
- `ricci_finance/visualization.py`

Use explicit imports from package modules. Keep graph construction independent
from visualization. New regime engines should implement the common engine
interface and return states, optional probabilities, runtime and availability.
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

[← Previous](README-06-Validation.md) · [Home](../README.md) · [Next →](README-08-Future.md)

