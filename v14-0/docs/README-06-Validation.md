# Validation

Validation must not rely on the visual attractiveness of the network.

The project checks:

- input completeness and overlap;
- graph edge stability;
- HMM convergence and posterior confidence;
- Viterbi/posterior agreement;
- regime run-length distribution;
- HSMM reduction of one-frame flicker;
- cross-engine consensus;
- reproducible configuration;
- notebook and smoke-test execution.

Real market regimes have no perfect ground-truth labels. Therefore stability,
out-of-sample separation and economic plausibility must be considered together.
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

[← Previous](README-05-Regime-Engines.md) · [Home](../README.md) · [Next →](README-07-Developer.md)

