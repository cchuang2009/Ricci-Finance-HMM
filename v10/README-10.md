# Ricci Finance v10

**v10 = v8 topology + optional v9 flow layer.**

This version is designed for market interpretation first:

- v8-style sparse rolling correlation topology is the default view.
- IPO / late-start tickers appear only after enough valid observations.
- Edge labels show financial distance weights: `w=...`.
- HMM regime detection remains available.
- Capital-flow proxy is shown separately from Ricci topology.
- v9 Ricci flow and financial surgery are optional diagnostics on the selected frame.

## Why v10?

v8 was better for market visualization because sparse threshold graphs show clusters, isolated names, and sector rotation clearly.
v9 was better for research because Ricci flow and surgery describe stress, geometric deformation, and singular bridges.

v10 combines both:

```text
Raw v8 topology
    -> optional Ricci flow
    -> optional surgery
    -> compare before / after
```

## Install

```bash
pip install -r requirements.txt
```

or manually:

```bash
pip install streamlit yfinance pandas numpy networkx plotly matplotlib scikit-learn hmmlearn GraphRicciCurvature pot networkit
```

`GraphRicciCurvature`, `pot`, and `networkit` are optional but recommended. If unavailable, `helper.py` uses a didactic fallback curvature.

## Run

```bash
streamlit run app.py
```

## Recommended market settings

For v8-style cluster visualization:

```text
Edge mode: threshold
Positive correlation only: ON
Minimum correlation threshold: 0.30–0.45
Max financial distance: 0.90–1.10
Keep top-N: 0
```

For smoother map display:

```text
Edge mode: hybrid
kNN neighbors: 2–3
```

For Perelman-style flow/surgery diagnostics:

```text
Enable Ricci flow layer: ON
Apply financial surgery after flow: ON
Singular curvature threshold: -0.25 to -0.40
Long-edge quantile: 0.70–0.85
Require bridge or long edge: ON
```

## Important interpretation

Ricci topology and capital flow are not the same object.

- Ricci network = correlation geometry.
- Capital-flow proxy = recent theme return.

So AI or Quantum can receive capital but remain separate clusters if the stocks do not move together strongly inside the selected rolling window.

## Files

```text
app.py                         Streamlit v10 app
helper.py                      Reusable Ricci finance helper module
Ricci_Finance_v10_Lecture.ipynb Graduate lecture notebook
README-10.md                   This file
requirements.txt               Python dependencies
```
