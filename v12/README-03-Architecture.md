# Architecture

```text
Market data
    ↓
Returns and market features
    ↓
Rolling graph construction
    ↓
Capital-weighted effective distance
    ↓
Ollivier–Ricci curvature
    ↓
Graph statistics and edge stability
    ↓
Hidden Markov regime estimation
    ↓
Frame-to-frame story generation
    ↓
Static and animated 3-D visualization
```

The architecture separates computation from visualization. `helper.py` remains a
compatibility facade, while new code should import functions from their owning
modules under `ricci_finance/`.

---

**Documentation:** [Documentation Home](README.md) · [Introduction](README-01-Introduction.md) · [Installation](README-02-Installation.md) · [Architecture](README-03-Architecture.md) · [Mathematics](README-04-Mathematics.md) · [Modules](README-05-Modules.md) · [Lecture Guide](README-06-Lecture.md) · [Developer Guide](README-07-Developer.md) · [Future Development](README-08-Future.md)

← [Installation](README-02-Installation.md) | [Documentation Home](README.md) | [Mathematics](README-04-Mathematics.md) →
