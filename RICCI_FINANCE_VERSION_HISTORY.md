# Ricci-Finance Version History

## Overview

Ricci-Finance evolved from a financial-network visualization prototype into a research platform combining:

- Financial correlation / distance networks
- Ollivier–Ricci curvature and Ricci flow
- Rolling dynamic market graphs
- HMM / Viterbi market-regime detection
- Sector momentum and capital-flow analysis
- Dynamic communities
- GNN / GAT representations
- UMAP / t-SNE latent-space visualization
- Temporal network animation
- 3-D Galaxy visualization
- Jupyter Notebook and Streamlit interfaces
- Docker-based reproducible environments

---

## Version History

| Version | Major Features / Changes |
|---|---|
| **V1** | Initial financial network; stocks as nodes and relationships as edges; basic network visualization. |
| **V2** | Financial distance \(d_{ij}=\sqrt{2(1-\rho_{ij})}\); shifted from correlation visualization toward geometric modeling. |
| **V3** | Ollivier–Ricci curvature; curvature-based edge/network analysis; positive/negative curvature interpretation. |
| **V4** | Ricci-flow-inspired network evolution; repeated curvature/flow calculations; market treated as an evolving geometric object. |
| **V5** | Rolling financial networks; sequence \(G_{t_1},G_{t_2},\ldots,G_{t_T}\); temporal network statistics. |
| **V6** | Graph-derived time series; HMM and Viterbi; market-regime identification. |
| **V7** | Streamlit interactive platform; ticker selection; interactive graph visualization; modular application structure. |
| **V8** | Enhanced network visualization; improved layouts, node/edge information, curvature display, filtering, and interaction. |
| **V9** | Ricci Surgery; identification and restructuring/removal of unstable or highly negatively curved regions. |
| **V10** | Integrated dynamic market structure; rolling analysis, curvature, communities, and network statistics. |
| **V11** | Temporal visualization; frame-based network evolution; `make_movie.py`; MoviePy movie generation. |
| **V12** | Mathematical and educational version; documentation of financial distance, Ricci curvature/flow, HMM, Viterbi, and regimes; bilingual lecture materials. |
| **V13** | Streamlit → Jupyter conversion; notebook-compatible analysis and visualization; research/teaching workflow. |
| **V14** | Notebook integration and visualization refinement; sector momentum, sector flow, capital-flow analysis, ECharts/Plotly improvements. |
| **V15** | Integrated Ricci-Finance pipeline: rolling Ricci networks, graph features, HMM/Viterbi, sector momentum, capital flow, Ricci Surgery, and 3-D Galaxy. |
| **V16** | Multi-sector membership, dynamic sector evolution, automatic themes, dynamic communities, GNN/GCN/GAT, embeddings, attention visualization, UMAP/t-SNE, temporal animation, and Galaxy evolution. |
| **V16.x** | Notebook/Docker architecture; separate Streamlit and Jupyter environments; shared `ricci_finance/`, `pyproject.toml`, and `uv.lock`; reproducible research setup. |

---

## Detailed Milestones

### V1–V4 — Financial Geometry

```text
Stocks
  ↓
Financial Network
  ↓
Financial Distance
  ↓
Ollivier–Ricci Curvature
  ↓
Ricci Flow
```

**V1:** Basic financial graph.

**V2:** Correlation transformed into financial distance:

\[
d_{ij}=\sqrt{2(1-\rho_{ij})}
\]

**V3:** Added Ollivier–Ricci curvature.

**V4:** Introduced Ricci-flow-inspired network evolution.

---

### V5–V6 — Dynamic Market Geometry

```text
Rolling Network
      ↓
Graph Time Series
      ↓
Ricci Features
      ↓
HMM / Viterbi
      ↓
Market Regimes
```

V5 introduced rolling windows and dynamic graphs.

V6 connected graph-derived features with HMM/Viterbi regime detection.

---

### V7–V10 — Interactive Ricci-Finance

```text
Research Algorithms
      ↓
Streamlit
      ↓
Interactive Network
      ↓
Dynamic Analysis
      ↓
Ricci Surgery
```

The project became an interactive financial-analysis application while retaining the Ricci-geometry research core.

---

### V11–V14 — Temporal and Educational Platform

```text
Temporal Network
      ↓
Movie / Animation
      ↓
Mathematical Documentation
      ↓
Jupyter Notebook
      ↓
Teaching / Research
```

Major additions:

- Temporal network animation
- MoviePy-based movie generation
- Mathematical explanations
- Bilingual lecture materials
- Streamlit-to-Jupyter conversion
- Sector momentum
- Sector-flow and capital-flow visualization

---

### V15 — Integrated Ricci-Finance Research Pipeline

```text
Market Data
    ↓
Rolling Financial Graph
    ↓
Ollivier–Ricci Curvature
    ↓
Ricci Flow
    ↓
Graph Features
    ↓
HMM / Viterbi
    ↓
Sector Momentum
    ↓
Capital Flow
    ↓
Ricci Surgery
    ↓
3-D Galaxy
```

Key components:

1. **Rolling Financial Graph**
2. **Financial distance**
3. **Ollivier–Ricci curvature**
4. **Graph features**
5. **HMM / Viterbi**
6. **Sector momentum**
7. **Sector capital-flow matrix**
8. **Ricci Surgery**
9. **3-D Galaxy visualization**

V15 intentionally kept the core focused on Ricci geometry, dynamic financial networks, HMM, sector/capital-flow analysis, surgery, and Galaxy visualization.

---

### V16 — Multi-Modal Financial Network Intelligence

#### 1. Multi-Sector Membership

Stocks can belong to multiple sectors:

```python
SECTOR_MAP = {
    "NVDA": ["Semiconductor", "AI"],
    "QNT": ["QuantumComputing"],
    "IONQ": ["QuantumComputing"],
    "QBTS": ["QuantumComputing"],
}
```

#### 2. Dynamic Sector Evolution

Sector characteristics can evolve through time and can be compared with detected themes and communities.

#### 3. Automatic Theme Detection

Market themes can be inferred from:

- Correlation structure
- Ricci curvature
- Communities
- Momentum
- Capital flow
- GNN representations

#### 4. Dynamic Communities

Compare:

- Conventional sectors
- User-defined sectors
- Detected themes
- Ricci communities
- GNN clusters

#### 5. GNN / GAT

Added graph neural-network analysis:

- GCN
- GAT
- Node embeddings
- Latent representations
- Attention weights

#### 6. UMAP / t-SNE

Project learned representations into low-dimensional space to compare:

```text
Traditional Sector
        vs
Ricci Community
        vs
GNN Cluster
```

#### 7. Attention Visualization

GAT attention weights provide a way to investigate important learned financial relationships.

#### 8. Temporal Network Animation

Animate changes in:

- Network topology
- Communities
- Sector structure
- Momentum
- Capital flow
- Ricci curvature

#### 9. 3-D Galaxy Evolution

The Galaxy representation can evolve through time:

```text
Galaxy(t1)
   ↓
Galaxy(t2)
   ↓
Galaxy(t3)
   ↓
...
```

---

## Architecture Evolution

### Early Architecture

```text
Market Data
    ↓
Graph
    ↓
Ricci Curvature
    ↓
Visualization
```

### V15 Architecture

```text
Data
 ↓
Rolling Ricci Network
 ↓
Graph Features
 ↓
HMM / Viterbi
 ↓
Sector / Capital Flow
 ↓
Ricci Surgery
 ↓
Galaxy
```

### V16 Architecture

```text
                         ┌─ Ricci Geometry
                         ├─ Communities
                         ├─ Capital Flow
Financial Data → Graph ──┼─ HMM Regimes
                         ├─ GNN / GAT
                         ├─ Sector / Theme
                         └─ Temporal Galaxy
```

---

## Overall Evolution

The project can be divided into six stages:

| Stage | Versions | Focus |
|---|---|---|
| **1. Financial Geometry** | V1–V4 | Network → distance → Ricci curvature → flow |
| **2. Dynamic Market Geometry** | V5–V6 | Rolling graphs → graph time series → HMM regimes |
| **3. Interactive Platform** | V7–V10 | Streamlit → visualization → Ricci Surgery |
| **4. Temporal/Educational Platform** | V11–V14 | Animation → mathematics → Jupyter → teaching |
| **5. Integrated Ricci-Finance** | V15 | Ricci + HMM + sectors + capital flow + surgery + Galaxy |
| **6. Multi-Modal Network Intelligence** | V16 | Multi-sector + communities + GNN/GAT + embeddings + temporal Galaxy |

---

## Current Conceptual Position

Ricci-Finance has evolved from a **financial graph visualization system** into a research framework for studying market structure through multiple complementary representations:

$$
\begin{align}{
  \text{Financial Data} 
\rightarrow
\text{Dynamic Graph}
\rightarrow
\begin{cases}
\text{Ricci Geometry}\\
\text{Communities}\\
\text{Capital Flow}\\
\text{HMM Regimes}\\
\text{GNN Embeddings}
\end{cases}
\rightarrow
\text{Market Structure}
}
\end{align}
$$

The central research question has therefore evolved from:

> **How are stocks related?**

to:

> **How does the geometry, topology, community structure, information flow, and learned representation of the financial market evolve through time?**

---

## Versioning Note

The early **V1–V6** history is a reconstruction of the project's conceptual development, while **V7–V16** reflects increasingly detailed implementation records.

For formal publication, distinguish:

- **Development milestones** — conceptual evolution
- **Release versions** — actual software releases/tags
- **V15/V16 subversions** — implementation refinements

This avoids implying that every conceptual milestone was necessarily a formally tagged Git release.
