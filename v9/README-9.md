# Ricci Finance v9: Ricci Flow, Singularity, and Financial Surgery

This package converts Perelman-style Ricci ideas into a graduate-level financial-network lecture and Streamlit app.

## Files

```text
app.py                                      # Streamlit v9 app
helper.py                                   # reusable Ricci/flow/surgery/HMM functions
Perelman_Ricci_Finance_Lecture_v9.ipynb     # graduate lecture notebook
README-9.md                                 # this file
requirements.txt                            # Python dependencies
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional but recommended for true Ollivier-Ricci curvature:

```bash
pip install GraphRicciCurvature pot networkit
```

If `GraphRicciCurvature` is not available, `helper.py` uses a didactic fallback curvature so the lecture still runs.

## Run the Streamlit app

```bash
streamlit run app.py
```

## GitHub / Streamlit Cloud layout

Put all files in the same GitHub repository folder:

```text
RicciFinanceV9/
├── app.py
├── helper.py
├── README-9.md
├── requirements.txt
└── Perelman_Ricci_Finance_Lecture_v9.ipynb
```

In `app.py`, the helper module is loaded normally:

```python
from helper import build_rolling_frames, run_ricci_flow, perform_financial_surgery
```

Do **not** use `from .helper import ...` unless you convert the project into a package.

## What v9 adds

v7 used rolling Ricci financial networks, Plotly animation, IPO-aware nodes, and HMM regimes.
v8 added Ricci flow.
v9 adds Perelman-inspired financial surgery.

Pipeline:

```text
prices
  -> log returns
  -> rolling correlation matrix
  -> financial distance d_ij = sqrt(2(1-rho_ij))
  -> graph construction
  -> Ollivier-Ricci curvature
  -> Ricci flow
  -> singular-edge detection
  -> surgery / cutting
  -> topology and regime diagnostics
```

## Why Ricci flow matters in finance

Curvature is a static measurement. Ricci flow makes it dynamic.

Positive-curvature edges contract. In finance this often means coherent sector or factor clustering.
Negative-curvature edges stretch. In finance this often means fragile bridges, crowded trades, cross-sector contagion, or liquidity stress.

The flow asks:

> If the market geometry is allowed to evolve according to its own curvature, which links stabilize and which links become singular?

## Why surgery matters

In geometric Ricci flow, singularities can form. Perelman/Hamilton surgery cuts problematic neck regions so the remaining geometry can continue evolving.

In finance, the analogy is:

| Geometry | Finance analogue |
|---|---|
| negative-curvature neck | fragile market bridge |
| singularity | liquidity/correlation blow-up |
| cut/surgery | remove unstable contagion channel |
| post-surgery components | true market sectors/regimes |
| component entropy | fragmentation disorder |

v9 detects singular edges using curvature, long-distance ranking, and optional graph-bridge tests, then removes those edges and recomputes topology.

## Streamlit controls

### Edge mode

- `threshold`: only strong relation edges pass correlation/distance filters.
- `knn`: each ticker keeps nearest neighbors, reducing isolated nodes.
- `hybrid`: threshold graph plus kNN anti-isolation edges. This is the recommended lecture setting.

Recommended first settings:

```text
Edge mode: hybrid
kNN neighbors: 3
Minimum correlation threshold: 0.20
Max financial distance: 1.15
Flow iterations: 10
Flow step size: 0.25
Singular curvature threshold: -0.35
Long-edge quantile: 0.80
```

## Important interpretation warning

This app is an exploratory geometry tool, not a trading signal by itself. Curvature, flow, HMM, and surgery should be compared with volume, macro events, earnings, options flow, liquidity, and out-of-sample validation.

## Deploy to Streamlit Community Cloud

1. Push `app.py`, `helper.py`, `README-9.md`, `requirements.txt`, and the notebook to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app.
4. Select the repository.
5. Set main file path to:

```text
app.py
```

6. Deploy.

If deployment fails on `GraphRicciCurvature` or `networkit`, remove those two optional packages from `requirements.txt`; the fallback curvature will still let the app run.

## v9.1 IPO-aware node appearance fix

The synthetic lecture data now matches the v8 IPO behavior: late-start tickers such as `QNT`, `BNT`, `SPCX`, `CBRS`, and `DXYZ` have `NaN` prices before their synthetic IPO/start date. Therefore a ticker is drawn only after it has at least `Min observations to show node` valid return observations inside the rolling window. The stable layout may reserve a coordinate for the ticker, but the node itself will not be visible before it is active.

For live yfinance data, the same rule is applied naturally from missing pre-IPO history.

## v9.1 patch: edge labels + capital-flow proxy

This patch restores the v8-style visible edge label option in the Plotly animation:

- `Show edge weight labels`
- `Number of visible edge labels`

The label `w=...` is the financial distance weight, usually

```text
w = sqrt(2 * (1 - correlation))
```

Small weight means two tickers move more similarly. Large weight means the relation is weaker or more distant.

The patch also adds a **Capital-flow proxy by theme** panel. Ricci geometry does not directly measure capital flows. It measures correlation geometry. Therefore AI or Quantum can receive capital and still fail to appear as one Ricci cluster if the tickers do not move together strongly enough inside the selected rolling window. The sector-flow proxy uses recent cumulative return and breadth by theme to separate momentum/flow interpretation from network-topology interpretation.
