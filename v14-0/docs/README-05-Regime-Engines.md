# Regime Engines

v14 supports:

- `hmmlearn-viterbi`: default story path;
- `hmmlearn-posterior`: marginal state comparison;
- `hsmm`: explicit-duration regime path;
- `pomegranate`: optional posterior engine.

All engines return a common output structure. The application reports switch
rate, run length, availability, runtime and consensus. pomegranate is skipped
cleanly if it is not installed or its installed API is incompatible.
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

[← Previous](README-04-Mathematics.md) · [Home](../README.md) · [Next →](README-06-Validation.md)

