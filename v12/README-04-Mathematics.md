# Mathematics

This chapter introduces the mathematical principles behind the v12 Ricci-Finance framework. Rather than using individual stock prices alone, v12 models the financial market as a dynamic graph whose geometry evolves over time. Geometry, probability, and machine learning are combined to reveal hidden market structures, capital flow, and market regimes.

---

# 數學原理

本章介紹 v12 Ricci-Finance 所使用的核心數學理論。v12 並非只分析單一股票價格，而是將整個金融市場視為一個隨時間演化的圖形（Graph）或流形（Manifold），並結合幾何學、機率模型與機器學習，分析市場結構、資金流向以及隱藏市場狀態。

---

# 1. Mathematical Philosophy

## English

Traditional quantitative finance mainly studies individual time series such as stock prices, returns, volatility, or technical indicators.

However, financial markets are **networks**, not isolated assets.

Stocks influence each other through

- industrial supply chains,
- institutional capital,
- ETF rebalancing,
- macroeconomic events,
- investor sentiment.

Therefore, instead of asking

> "How will one stock move?"

v12 asks

> "How does the entire market geometry evolve?"

The objective is to detect structural changes before they become obvious in prices.

---

## 中文

傳統量化金融主要研究單一股票的時間序列，例如價格、報酬率、波動率或技術指標。

然而，金融市場本質上是一個**網路系統**，而非彼此獨立的股票集合。

股票之間透過

- 產業供應鏈
- 法人資金
- ETF 調整
- 總體經濟
- 市場情緒

互相影響。

因此，本專案不只是回答

> 「某一檔股票會怎麼走？」

而是回答

> 「整個市場幾何如何演化？」

目的是在價格尚未完全反映之前，就觀察市場結構是否開始改變。

---

# 2. Rolling Window

## Why?

Financial markets are highly non-stationary.

Relationships between stocks today are different from those several months ago.

Therefore, v12 analyzes the market using rolling windows instead of one static dataset.

For every window,

```
Window 1
↓

Window 2
↓

Window 3
↓

...

Window N
```

one market graph is constructed.

These graphs become the snapshots of the evolving market manifold.

---

## 中文

金融市場具有高度非平穩性（Non-stationary）。

股票之間的關係會隨時間持續改變，因此不能使用整段歷史資料建立一張固定的網路。

v12 採用 Rolling Window。

每一個視窗都建立一張市場圖，

形成

```
Window1 → Window2 → Window3 → ...
```

也就是市場流形隨時間演化的過程。

---

# 3. Correlation Distance

The Pearson correlation coefficient

\[
\rho_{ij}
\]

is converted into a geometric distance

\[
d_{ij}
=
\sqrt{2(1-\rho_{ij})}.
\]

Properties:

- correlation = 1

distance = 0

- correlation = 0

distance = √2

- correlation = -1

largest distance

This transforms statistical similarity into graph geometry.

---

## 中文

利用 Pearson 相關係數

\[
\rho_{ij}
\]

建立金融距離

\[
d_{ij}
=
\sqrt{2(1-\rho_{ij})}.
\]

性質如下：

- 完全相關 → 距離為 0
- 無相關 → 距離約 √2
- 完全負相關 → 最大距離

因此可以將統計相關性轉換成圖形上的幾何距離。

---

# 4. Capital-aware Distance

Correlation alone ignores market activity.

Two stocks with the same correlation may have very different trading volumes.

v12 therefore optionally incorporates

- trading volume,
- dollar volume,
- market capitalization,
- estimated capital flow,

to modify graph distances.

The resulting graph better reflects actual market interactions.

---

## 中文

只有相關係數並不足以描述市場。

例如兩檔股票可能具有相同相關係數，

但成交量完全不同。

因此 v12 可以加入

- 成交量
- 成交金額
- 市值
- 資金流

修正圖形距離，使市場幾何更接近真實資金流動。

---

# 5. Ollivier–Ricci Curvature

## Why Ricci Curvature?

Distance only measures how far two stocks are.

Ricci curvature measures

**how similar their neighborhoods are.**

Positive curvature indicates

- coherent sectors,
- synchronized movement,
- stable capital concentration.

Negative curvature indicates

- bridge stocks,
- sector rotation,
- structural stress,
- possible market transitions.

Instead of studying individual edges, Ricci curvature evaluates the local geometry surrounding every edge.

---

## 中文

距離只能表示股票彼此相隔多遠。

Ricci 曲率則描述

**兩個鄰域是否具有相似結構。**

正曲率通常代表

- 穩定產業群
- 同步漲跌
- 資金集中

負曲率通常代表

- 橋接股票
- 類股輪動
- 市場壓力
- 結構轉換

Ricci 曲率分析的是整個局部幾何，而不是單一邊。

---

# 6. Ricci Flow

Hamilton introduced Ricci Flow as

\[
\frac{\partial g}{\partial t}
=
-2Ric.
\]

On graphs,

Ricci Flow repeatedly updates edge weights according to curvature.

Its purposes are

- reducing noise,
- sharpening communities,
- emphasizing important structures,
- stabilizing graph geometry.

---

## 中文

Hamilton 提出的 Ricci Flow

\[
\frac{\partial g}{\partial t}
=
-2Ric
\]

在圖形中可視為反覆利用曲率更新邊權重。

主要用途包括

- 去除雜訊
- 強化群聚
- 突顯重要結構
- 穩定市場網路

---

# 7. Graph Features

For every rolling window, v12 computes a feature vector.

Typical features include

- Mean Ricci curvature
- Curvature variance
- Negative-edge ratio
- Graph density
- Connected components
- Largest component size
- Edge stability
- Capital concentration
- Total edge capital flow

These features summarize one market graph numerically.

---

## 中文

每一個 Rolling Window 都會萃取一組圖形特徵。

包括

- 平均 Ricci 曲率
- 曲率變異
- 負曲率比例
- 網路密度
- 連通元件
- 最大群聚
- 邊穩定度
- 資金集中度
- 總資金流

因此，一張市場圖可以表示成一個特徵向量。

---

# 8. Hidden Markov Model (HMM)

## Why HMM?

The market regime cannot be observed directly.

We only observe graph features extracted from each rolling window.

Examples of hidden regimes include

- Bull Market
- Consolidation
- Sector Rotation
- Correction
- Panic
- Recovery

HMM estimates these hidden states from observed graph features.

---

## 中文

市場真正所處的狀態無法直接觀察。

例如

- 多頭
- 整理
- 類股輪動
- 修正
- 恐慌
- 復甦

都屬於隱藏狀態。

我們真正能觀察的是

每個 Rolling Window 所萃取出的圖形特徵。

因此 HMM 利用觀測資料推估隱藏市場狀態。

---

# 9. Why use HMM on Rolling Windows?

Each rolling window produces one observation vector

\[
x_t
=
[
\mu_{Ricci},
\sigma_{Ricci},
Density,
Components,
Flow,
...
].
\]

Therefore,

```
Window1
↓

Feature Vector1

Window2
↓

Feature Vector2

...

WindowN
↓

Feature VectorN
```

These vectors form the observation sequence

\[
x_1,x_2,\cdots,x_T
\]

used by the Hidden Markov Model.

HMM is **not trained on raw prices**.

Instead, it is trained on **market geometry** extracted from every rolling window.

This allows the model to identify hidden market regimes rather than simply fitting price movements.

---

## 中文

每一個 Rolling Window 都會產生一組圖形特徵

\[
x_t
=
[
平均曲率,
曲率變異,
網路密度,
資金流,
...
]
\]

形成

```
Window1
↓

Feature1

Window2
↓

Feature2

...

WindowN
↓

FeatureN
```

所有 Feature Vector

形成

\[
x_1,x_2,\cdots,x_T
\]

作為 HMM 的輸入。

因此，

HMM **不是分析價格**，

而是分析

**市場幾何（Market Geometry）**

以及市場結構如何隨時間演化。

---

# 10. Information Produced by HMM

For every rolling window, HMM estimates

- Most probable hidden state
- Posterior probability of every state
- State transition probability
- State persistence
- Expected next regime

These outputs are attached to every animation frame, allowing users to observe both the network evolution and the inferred market regime simultaneously.

---

## 中文

對於每一個 Rolling Window，

HMM 可輸出

- 最可能市場狀態
- 各狀態機率
- 狀態轉移機率
- 狀態持續時間
- 下一個可能狀態

因此，每一個動畫畫格(Frame)除了市場網路之外，也包含對應的市場 Regime。

---

# 11. Complete Mathematical Pipeline

```
Price Data
      │
      ▼
Rolling Window
      │
      ▼
Correlation Matrix
      │
      ▼
Financial Distance
      │
      ▼
Capital-aware Distance
      │
      ▼
Weighted Graph
      │
      ▼
Ricci Curvature
      │
      ▼
Ricci Flow
      │
      ▼
Graph Feature Extraction
      │
      ▼
Feature Vector
      │
      ▼
Hidden Markov Model
      │
      ▼
Hidden Market Regime
      │
      ▼
Visualization and Decision Support
```

---

## 中文

```
市場價格
      │
      ▼
Rolling Window
      │
      ▼
相關矩陣
      │
      ▼
金融距離
      │
      ▼
資金修正距離
      │
      ▼
市場圖形
      │
      ▼
Ricci 曲率
      │
      ▼
Ricci Flow
      │
      ▼
圖形特徵
      │
      ▼
Feature Vector
      │
      ▼
Hidden Markov Model
      │
      ▼
市場 Regime
      │
      ▼
視覺化與決策支援
```

This pipeline summarizes the complete mathematical framework used in the v12 Ricci-Finance project, integrating geometry, network science, probability, and machine learning to characterize evolving market structures.
