# Ricci Finance v10 — Streamlit app

This package upgrades the v8/v9 Ricci Finance lecture app with a more sensible market-map graph:

- `threshold`: strong-relation graph only.
- `knn`: each ticker keeps its nearest positive-correlation neighbors.
- `knn+bridges`: kNN plus a few weak positive bridge edges to prevent themes from drifting too far apart.
- Tunable layout spacing `layout_k`; lower values pull disconnected clusters closer.
- Ricci curvature, Ricci flow, HMM regime diagnostics, edge tables, and CSV exports.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

Put `app.py`, `helper.py`, `requirements.txt`, and this README in the same GitHub repository root. In Streamlit Cloud, set the main file path to:

```text
app.py
```

## Suggested market-map settings

```text
graph_mode = knn+bridges
kNN neighbors = 3
minimum positive correlation = 0.05
bridge edges = 3
cluster spacing k = 0.45
```

Use `threshold` when you want only strong relations; use `knn+bridges` when the goal is visualizing market relations without excessive component separation.
