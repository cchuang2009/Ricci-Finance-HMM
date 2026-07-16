# 1. Introduction

[Home](../README.md) · [Introduction](README-01-Introduction.md) · [Installation](README-02-Installation.md) · [Architecture](README-03-Architecture.md) · [Mathematics](README-04-Mathematics.md) · [Regime Engines](README-05-Regime-Engines.md) · [Validation](README-06-Validation.md) · [Developer](README-07-Developer.md) · [Future](README-08-Future.md)

Ricci Finance v13 models the market as a sequence of rolling weighted graphs. Each frame contains return relationships, capital mass, capital transport, Ollivier-Ricci curvature, connected components, and network diagnostics.

The regime layer asks which latent market condition most plausibly generated the current sequence of graph statistics. v13 does not assume that one library or one decoder is universally best. It compares several engines under a common interface.

## Research questions

1. Does the market graph move between cohesive, transitional, and fragmented regimes?
2. Does explicit state duration reduce implausible one-frame regime changes?
3. When do Viterbi and posterior decoding disagree?
4. Do independent engines agree on the same market story?
5. Are regime assignments stable under walk-forward validation?

## Application entry points

- `app_v13.py`: interactive Streamlit application.
- `notebooks/ricci_finance_v13_regime_benchmark.ipynb`: controlled comparison and validation.
- `ricci_finance/regime_engines/`: replaceable engine implementations.

---

Previous: [Home](../README.md) · Next: [Installation](README-02-Installation.md)
