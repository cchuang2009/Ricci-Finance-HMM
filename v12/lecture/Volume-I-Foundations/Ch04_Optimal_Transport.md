# Chapter 4 — Optimal Transport

# 第四章：最佳傳輸

---

## Learning Objectives / 學習目標

### English

- Understand neighborhood probability measures and transport cost.
- See how Wasserstein distance enters Ollivier–Ricci curvature.

### 繁體中文

- 理解鄰域機率分布與傳輸成本。
- 理解 Wasserstein 距離如何進入 Ollivier–Ricci 曲率。

## Neighborhood Measures / 鄰域分布

### English

Each node distributes probability mass to itself and its neighbors.
         This converts local graph structure into a probability distribution.

### 繁體中文

每個節點將機率質量分配給自身與鄰居，藉此將局部圖形結構轉換為機率分布。

## Wasserstein Distance / Wasserstein 距離

### English

The first Wasserstein distance \(W_1(m_i,m_j)\) is the minimum cost required
         to transport neighborhood mass from node \(i\) to node \(j\). It is the graph
         analogue of Earth Mover's Distance.

### 繁體中文

第一階 Wasserstein 距離 \(W_1(m_i,m_j)\) 表示將節點 \(i\) 的鄰域質量
         搬運成節點 \(j\) 鄰域質量所需的最小成本，也就是 Earth Mover's Distance 的圖形版本。

## Functions Used / 使用函式

### `build_node_distribution()`

**Purpose:** Create a probability measure around a node.

**用途：** 建立節點周圍的機率分布。

**Inputs / 輸入：** graph, node, alpha

**Outputs / 輸出：** neighbor probability vector

### `wasserstein_distance()`

**Purpose:** Compute minimum transport cost.

**用途：** 計算最小傳輸成本。

**Inputs / 輸入：** two node distributions, ground distance

**Outputs / 輸出：** transport distance

## Summary / 摘要

This chapter connects the mathematical idea, the financial interpretation, and the software implementation used in Ricci-Finance v12.

本章將數學概念、金融意義與 Ricci-Finance v12 的程式實作連結在一起。
