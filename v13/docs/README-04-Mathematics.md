# 4. Mathematics

[Home](../README.md) · [Introduction](README-01-Introduction.md) · [Installation](README-02-Installation.md) · [Architecture](README-03-Architecture.md) · [Mathematics](README-04-Mathematics.md) · [Regime Engines](README-05-Regime-Engines.md) · [Validation](README-06-Validation.md) · [Developer](README-07-Developer.md) · [Future](README-08-Future.md)

## Correlation distance

For return correlation $\rho_{ij}$, the base graph distance is

$$
d_{ij}=\sqrt{2(1-\rho_{ij})}.
$$

Capital similarity can contract this distance without changing the observed correlation.

## Ollivier-Ricci curvature

An edge with positive curvature lies inside a locally cohesive region. Strongly negative curvature often behaves like a bridge between otherwise separated market basins. Curvature is descriptive geometry, not a future-price forecast.

## HMM

For observations $X_t$ and hidden states $S_t$,

$$
P(S_{1:T},X_{1:T})=P(S_1)\prod_{t=2}^T P(S_t\mid S_{t-1})\prod_{t=1}^T P(X_t\mid S_t).
$$

Viterbi decoding selects the single path maximizing the joint probability. Posterior decoding chooses the most probable state independently at each frame.

## HSMM

The HSMM adds a duration variable $D$, allowing each state to have an explicit duration distribution. This can suppress economically implausible one-frame regimes.

## Consensus

For multiple state sequences, v13 uses majority agreement per frame. Consensus does not prove correctness; disagreement is itself a diagnostic that the regime is uncertain or model-dependent.

---

Previous: [Architecture](README-03-Architecture.md) · Next: [Regime Engines](README-05-Regime-Engines.md)
