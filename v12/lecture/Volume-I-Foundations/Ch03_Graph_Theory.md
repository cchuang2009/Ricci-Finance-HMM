# Chapter 3 — Graph Theory and Network Science

# 第三章：圖論與網路科學

---

## Learning Objectives / 學習目標

### English

- Understand nodes, edges, paths, components, density, and communities.
- Interpret graph quantities financially.

### 繁體中文

- 理解節點、邊、路徑、連通元件、密度與群聚。
- 以金融角度解讀圖形指標。

## Nodes and Edges / 節點與邊

### English

A node represents an asset. An edge exists when a similarity rule is met.
         Edge attributes may store correlation, distance, capital flow, and curvature.

### 繁體中文

節點代表資產；當兩資產符合相似性條件時建立邊。邊可儲存相關係數、
         距離、資金流與曲率等屬性。

## Connectivity / 連通性

### English

Connected components reveal whether the market behaves as one integrated
         system or splits into isolated groups. Rapid fragmentation may indicate regime
         change or liquidity stress.

### 繁體中文

連通元件可顯示市場是高度整合，還是分裂成數個孤立群組。
         快速碎裂可能代表市場狀態改變或流動性壓力。

## Functions Used / 使用函式

### `build_graph()`

**Purpose:** Construct a weighted market graph.

**用途：** 建立加權市場圖。

**Inputs / 輸入：** correlation matrix, threshold

**Outputs / 輸出：** NetworkX graph

### `compute_components()`

**Purpose:** Find connected components and their sizes.

**用途：** 找出連通元件與其大小。

**Inputs / 輸入：** graph

**Outputs / 輸出：** component statistics

### `compute_density()`

**Purpose:** Measure realized edges relative to all possible edges.

**用途：** 衡量實際邊數占所有可能邊數的比例。

**Inputs / 輸入：** graph

**Outputs / 輸出：** density scalar

## Summary / 摘要

This chapter connects the mathematical idea, the financial interpretation, and the software implementation used in Ricci-Finance v12.

本章將數學概念、金融意義與 Ricci-Finance v12 的程式實作連結在一起。
