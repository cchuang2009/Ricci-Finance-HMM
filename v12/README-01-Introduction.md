# Introduction

Ricci Finance v12 models the market as a sequence of rolling weighted graphs.
Each graph frame represents one observation window and contains tickers as nodes,
statistical relationships as edges, Ricci curvature, capital-flow attributes,
clusters, and graph-level regime features.

The HMM assigns a latent market regime to every rolling frame. The animation then
shows how the graph geometry, capital concentration, and inferred regime evolve
together.

## Main questions

- How cohesive or fragmented is the current market network?
- Which edges are fragile bridges between market basins?
- Where is capital concentrated or transported?
- Is the current frame consistent with stress, transition, or coherence?
- How did the present frame change from the preceding frame?

---

**Documentation:** [Documentation Home](README.md) · [Introduction](README-01-Introduction.md) · [Installation](README-02-Installation.md) · [Architecture](README-03-Architecture.md) · [Mathematics](README-04-Mathematics.md) · [Modules](README-05-Modules.md) · [Lecture Guide](README-06-Lecture.md) · [Developer Guide](README-07-Developer.md) · [Future Development](README-08-Future.md)

← [Documentation Home](README.md) | [Documentation Home](README.md) | [Installation](README-02-Installation.md) →
