# Dynamic Multi-Sector Ricci Flow + Graph Neural Networks


Financial markets are dynamic networks whose topology changes continuously through capital flows.
Ricci Finance V16 models the market as a weighted graph and combines:

金融市場可視為一個隨時間演化的複雜網路。V16 將市場建構為加權圖(Graph)，結合:

- Dynamic sector evolution 動態產業權重
- Ricci curvature, Ricci 曲率
- Ricci Flow
- Graph Neural Networks, 圖神經網路(GNN)
- Dynamic community detection, 動態社群分析
- Capital-flow analysis, 資金流分析

to investigate how money moves between industries over time. Unlike previous versions, V16 no longer assumes that every company belongs to a single industry.
Instead, every company owns a normalized <b>multi-sector</b> profile which evolves through time.

研究資金如何在不同產業間流動。V16 最大特色為：<b>每家公司不再只有一個 Sector，而是具有隨時間演化的多產業權重。</b>

---

# 13. References / 參考文獻

Ricci Finance V16 is built upon numerous outstanding open-source software packages and decades of research in differential geometry, graph theory, machine learning, and quantitative finance.

Ricci Finance V16 建立於許多優秀的開源軟體，以及微分幾何、圖論、機器學習與量化金融等領域的重要研究成果。

---

# 13.1 Software Packages / 使用套件

| Package | Purpose in V16 | 用途 |
|----------|----------------|------|
| GraphRicciCurvature | Ollivier Ricci Curvature, Forman Ricci Curvature, Ricci Flow | Ricci 曲率與 Ricci Flow 計算 |
| NetworkX | Graph construction and graph algorithms | 建立金融網路 |
| NetworKit | High-performance graph computation | 加速 Ricci 計算 |
| POT (Python Optimal Transport) | Wasserstein distance for Ollivier Ricci | Optimal Transport |
| NumPy | Numerical computation | 數值計算 |
| SciPy | Statistics and Linear Algebra | 統計與線性代數 |
| Pandas | Financial time-series processing | 金融資料處理 |
| yfinance | Yahoo Finance market data | 股票資料下載 |
| PyTorch | Graph Neural Networks (GCN/GAT) | GNN 深度學習 |
| scikit-learn | PCA, t-SNE, clustering, evaluation | 降維與分析 |
| UMAP-learn | Nonlinear embedding visualization | 高維資料降維 |
| Plotly | Interactive 3D visualization | 3D 互動圖形 |
| Apache ECharts | Heatmaps, animations, dashboards | Dashboard 視覺化 |
| streamlit-echarts | ECharts wrapper | Streamlit ECharts |
| Streamlit | Interactive web application | Web GUI |

---

# 13.2 Mathematical Foundations / 數學基礎

## Ricci Curvature

### Ollivier, Y. (2009)

**Ricci Curvature of Markov Chains on Metric Spaces**

Journal of Functional Analysis, 256(3), 810–864.

Introduced Ollivier Ricci Curvature for metric measure spaces.

本研究提出 Ollivier Ricci 曲率，是 Graph Ricci Curvature 的理論基礎。

---

### Ni, C.-C., Lin, Y.-Y., Gao, J., Gu, X., Saucan, E.

**Ricci Curvature of the Internet Topology**

IEEE INFOCOM, 2015.

Used in V16 for

- Ollivier Ricci Curvature
- Edge robustness analysis
- Financial network geometry

V16 用於

- Ollivier Ricci 曲率
- 網路穩定性分析
- 金融市場拓樸分析

---

### Ni, C.-C., Lin, Y.-Y., Gao, J., Gu, X.

**Network Alignment by Discrete Ollivier-Ricci Flow**

Graph Drawing, 2018.

Introduced discrete Ricci Flow on graphs.

V16 用於 Ricci Flow。

---

### Ni, C.-C., Lin, Y.-Y., Luo, F., Gao, J.

**Community Detection on Networks with Ricci Flow**

Scientific Reports, 2019.

Used in V16 for

- Ricci community detection
- Dynamic market clustering

V16 使用於

- 社群偵測
- 市場群聚分析

---

### Forman, R. (2003)

**Bochner's Method for Cell Complexes and Combinatorial Ricci Curvature**

Discrete & Computational Geometry.

Original Forman Ricci Curvature.

---

### Sreejith, R. P., et al. (2016)

**Forman Curvature for Complex Networks**

Journal of Statistical Mechanics.

Used for

- Fast curvature approximation
- Large-scale financial networks

---

### Samal, A., et al. (2018)

**Comparative Analysis of Two Discretizations of Ricci Curvature for Complex Networks**

Scientific Reports.

Provides comparison between

- Ollivier Ricci
- Forman Ricci

This paper motivated the comparison module implemented in V16.

---

# 13.3 Graph Neural Networks / 圖神經網路

### Kipf, T. N., & Welling, M. (2017)

**Semi-Supervised Classification with Graph Convolutional Networks**

ICLR.

Used in

- GCN implementation
- Node embedding

---

### Veličković, P., et al. (2018)

**Graph Attention Networks**

ICLR.

Used in

- GAT
- Attention visualization
- Attention heatmaps

---

### Hamilton, W., Ying, Z., & Leskovec, J. (2017)

**Inductive Representation Learning on Large Graphs (GraphSAGE)**

NeurIPS.

Provides scalable graph embedding concepts.

---

# 13.4 Dynamic Graph Learning / 動態圖神經網路

Dynamic Graph Neural Networks have inspired the temporal extensions implemented in V16.

V16 introduces

- Dynamic sector evolution
- Temporal graph embeddings
- Rolling-window GNN learning

V16 參考動態圖神經網路概念，加入

- 動態產業演化
- 時間序列嵌入
- Rolling Window GNN

---

# 13.5 Machine Learning / 機器學習

### Rabiner, L. R. (1989)

**A Tutorial on Hidden Markov Models**

Proceedings of the IEEE.

Used for

- Market regime detection
- Hidden state estimation

---

### Pearson, K. (1901)

Principal Component Analysis (PCA)

Used for

- Embedding visualization

---

### van der Maaten, L., & Hinton, G. (2008)

**Visualizing Data using t-SNE**

Journal of Machine Learning Research.

Used for

- Nonlinear embedding visualization

---

### McInnes, L., Healy, J., & Melville, J. (2018)

**UMAP: Uniform Manifold Approximation and Projection**

Used for

- High-dimensional visualization
- GNN latent embedding

---

# 13.6 Quantitative Finance / 量化金融

### Markowitz, H. (1952)

**Portfolio Selection**

Journal of Finance.

Introduced Modern Portfolio Theory.

---

### Mantegna, R. N. (1999)

**Hierarchical Structure in Financial Markets**

European Physical Journal B.

Introduced the correlation distance

\[
d=\sqrt{2(1-\rho)}
\]

which is adopted in Ricci Finance to construct the financial network.

V16 使用此距離建立金融圖。

---

# 13.7 Visualization Libraries / 視覺化工具

### Plotly

Used for

- 3D Galaxy visualization
- Interactive network visualization
- Animated scatter plots

---

### Apache ECharts

Used for

- Sector Momentum
- Capital Flow Matrix
- Heatmaps
- Timeline animations

---

### Streamlit

Used for

- Interactive dashboard
- Research demonstration
- Parameter exploration

---

# 13.8 Acknowledgements / 致謝

This project would not have been possible without the contributions of the open-source community.

Special thanks to the developers and researchers of

- GraphRicciCurvature
- PyTorch
- NetworkX
- Plotly
- Apache ECharts
- Streamlit
- Yahoo Finance
- scikit-learn
- UMAP
- POT

Ricci Finance V16 extends these excellent tools into a unified framework for financial network analysis using Ricci geometry and Graph Neural Networks.

本專案建立於眾多開源社群的努力之上，並將 Ricci Geometry、Graph Neural Networks 以及金融網路分析整合成一套完整研究平台，在此向所有研究者與開源開發者致上最高敬意。