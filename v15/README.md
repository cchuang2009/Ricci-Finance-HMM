# Ricci Finance V15 Final — SciPy LP, No TensorFlow, No POT

This build removes TensorFlow, Keras, POT (`ot`), and GraphRicciCurvature completely.

Pipeline:

`yfinance → rolling correlation graphs → capital-aware edge lengths → curvature → graph features → Gaussian HMM → sector flow / graph surgery → pure-PyTorch GCN`

## Curvature engines

### Ollivier LP

For every edge `(x,y)`, the program constructs idleness measures on the two one-hop neighborhoods and solves the discrete optimal-transport problem with `scipy.optimize.linprog(method="highs")`:

`κ(x,y) = 1 − W₁(mₓ,mᵧ) / d(x,y)`

This preserves the central Ollivier-Ricci definition without POT. It is computationally heavier because it solves one LP per edge per rolling frame.

### Forman

A weighted Forman-Ricci implementation uses only NetworkX, NumPy, and Python math. It is recommended for rapid Streamlit exploration and large frame counts.

Both engines write curvature to the same edge attribute:

```python
edge["ricciCurvature"]
```

Therefore HMM, graph surgery, visualizations, and the GNN share one downstream data format.

## Installation

Use a clean Python 3.13 virtual environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Do not install these packages for this project:

```text
tensorflow
keras
POT / ot
GraphRicciCurvature
```

Check the environment:

```bash
python check_install.py
```

Run Streamlit:

```bash
streamlit run app.py
```

Run the notebook:

```bash
jupyter lab ricci_finance_v15_lecture.ipynb
```

Run tests:

```bash
pytest -q
```

## Practical settings

For interactive work:

```text
Curvature engine: Forman
Frames: 40–80
Step: 3–5
```

For mathematical Ollivier-Ricci analysis:

```text
Curvature engine: Ollivier LP
Frames: 10–40 initially
Nodes: about 10–20
```

Increase frame count only after measuring runtime on the local machine.

## GNN

The GNN is pure PyTorch and does not use PyTorch Geometric. Its signature is:

```python
train_gcn_regime(graphs, labels, sectors, epochs=120, hidden=24, random_state=42)
```

The HMM `valid_index` is used to select the exact corresponding graph snapshots. This prevents graph/label mismatches.


## Final visualization edition

The final V15 UI replaces the earlier Matplotlib network with Plotly:

- pastel sector colors and adjustable node opacity
- labels drawn separately from node markers for improved readability
- interactive 2D network zoom and hover details
- a stable sector-radial 3D Ricci Galaxy
- Plotly edge-curvature bars and sector-flow heatmaps
- side-by-side before/after graph-surgery views
- GNN loss and HMM/GCN comparison panel

Galaxy coordinates are deterministic across reruns. Sector controls orbital direction, capital share affects radius, and node Ricci curvature controls vertical position.

## V15 ECharts UI update

All 2D charts in `app.py` now use `streamlit-echarts` with ECharts option dictionaries:

- regime feature history
- 2D market network
- edge-curvature bars
- sector momentum
- sector-flow heatmap
- surgery before/after networks
- GCN training loss

The existing Plotly 3D Ricci Galaxy is intentionally unchanged. Plotly remains in
`requirements.txt` only for that Galaxy view.

The former button variable `go` was renamed to `run_analysis`. This prevents it
from overwriting the common `plotly.graph_objects as go` module alias and fixes:

```text
AttributeError: 'bool' object has no attribute 'Figure'
```
