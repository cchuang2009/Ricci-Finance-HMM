# Architecture

The package is divided into data, features, graph, capital, Ricci, rolling,
regime engines, story, diagnostics and visualization modules.

`helper.py` is only a compatibility facade. New development should import from
the owning module.

The formal application uses yfinance. Synthetic generation is retained only for
tests so production and notebook results follow one data path.
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

[← Previous](README-02-Installation.md) · [Home](../README.md) · [Next →](README-04-Mathematics.md)

