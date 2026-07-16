# 7. Developer Guide

[Home](../README.md) · [Introduction](README-01-Introduction.md) · [Installation](README-02-Installation.md) · [Architecture](README-03-Architecture.md) · [Mathematics](README-04-Mathematics.md) · [Regime Engines](README-05-Regime-Engines.md) · [Validation](README-06-Validation.md) · [Developer](README-07-Developer.md) · [Future](README-08-Future.md)

## Add a new engine

1. Subclass `BaseRegimeEngine`.
2. Implement `fit` and `predict`.
3. Implement `predict_proba` and `score` when available.
4. Register the engine in `factory.py`.
5. Add tests for output length, state type, missing dependency behavior, and deterministic seeds.

## Required semantics

- Never label posterior argmax as Viterbi.
- Never fabricate probabilities for an engine that cannot calculate them.
- Preserve arbitrary state IDs internally; derive economic labels from observed feature summaries.
- Treat pomegranate as optional.
- Keep Ricci flow separate from forecasting.

## Main test commands

```bash
python -m pytest -q
python smoke_test_v13.py
```

## Streamlit API

v13 uses `width="stretch"` rather than deprecated `use_container_width=True`.

---

Previous: [Validation](README-06-Validation.md) · Next: [Future Work](README-08-Future.md)
