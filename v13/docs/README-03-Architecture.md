# 3. Architecture

[Home](../README.md) · [Introduction](README-01-Introduction.md) · [Installation](README-02-Installation.md) · [Architecture](README-03-Architecture.md) · [Mathematics](README-04-Mathematics.md) · [Regime Engines](README-05-Regime-Engines.md) · [Validation](README-06-Validation.md) · [Developer](README-07-Developer.md) · [Future](README-08-Future.md)

```text
Market data
  → rolling features
  → similarity graph + capital attributes
  → Ricci curvature / Ricci flow
  → graph-level feature table
  → interchangeable regime engines
  → state labels, probabilities, consensus
  → story animation and validation export
```

## Package layout

```text
ricci_finance/
├── data.py
├── features.py
├── graph.py
├── capital.py
├── ricci.py
├── rolling.py
├── visualization.py
├── story.py
├── regime_analysis.py
└── regime_engines/
    ├── base.py
    ├── factory.py
    ├── hmmlearn_engine.py
    ├── hsmm_engine.py
    ├── pomegranate_engine.py
    └── consensus.py
```

`BaseRegimeEngine` defines `fit`, `predict`, `predict_proba`, `score`, and `fit_predict`. The rest of the project therefore does not depend directly on a particular HMM package.

## Backward compatibility

`helper.py` remains a facade:

```python
from ricci_finance import *
```

Older v11/v12 notebooks can migrate gradually.

---

Previous: [Installation](README-02-Installation.md) · Next: [Mathematics](README-04-Mathematics.md)
