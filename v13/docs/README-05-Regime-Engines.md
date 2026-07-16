# 5. Regime Engines

[Home](../README.md) · [Introduction](README-01-Introduction.md) · [Installation](README-02-Installation.md) · [Architecture](README-03-Architecture.md) · [Mathematics](README-04-Mathematics.md) · [Regime Engines](README-05-Regime-Engines.md) · [Validation](README-06-Validation.md) · [Developer](README-07-Developer.md) · [Future](README-08-Future.md)

## hmmlearn Viterbi

Default production storyline. `predict()` returns the globally most likely state path, while `predict_proba()` returns marginal posterior probabilities.

## hmmlearn posterior

Uses `predict_proba(X).argmax(axis=1)`. This may react faster but can switch more often because each frame is classified separately.

## Explicit-duration HSMM

The v13 HSMM initializes Gaussian emissions and transitions from a GaussianHMM, estimates run-duration means, and then performs explicit-duration dynamic programming. It currently returns hard states but not posterior confidence.

## pomegranate DenseHMM

Optional PyTorch engine. In v13 it is deliberately treated as posterior-only:

```python
probabilities = model.predict_proba(X)
states = probabilities.argmax(axis=1)
```

It is not presented as a Viterbi replacement. Failure to import or fit pomegranate does not disable the other engines.

## Engine selection

```python
engine = create_regime_engine(
    "hmmlearn",
    n_components=3,
    decoding="viterbi",
)
output = engine.fit_predict(X)
```

---

Previous: [Mathematics](README-04-Mathematics.md) · Next: [Validation](README-06-Validation.md)
