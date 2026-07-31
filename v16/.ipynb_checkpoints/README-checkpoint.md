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