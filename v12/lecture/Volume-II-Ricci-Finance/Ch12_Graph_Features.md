# Chapter 12 — Graph Feature Engineering

# 第十二章：圖形特徵工程

---

## Learning Objectives / 學習目標

### English

- Understand every graph-level feature and its financial meaning.

### 繁體中文

- 理解每個圖形特徵及其金融意義。

## Geometry Features / 幾何特徵

### English

Mean curvature, dispersion, and negative-edge ratio summarize local geometry.

### 繁體中文

平均曲率、離散程度與負曲率比例概括局部幾何。

## Topology and Capital / 拓撲與資金

### English

Density, components, concentration, entropy, and total edge capital describe structure and allocation.

### 繁體中文

密度、連通元件、集中度、熵與總邊資金描述結構與配置。

## Functions Used / 使用函式

### `extract_graph_features()`

**Purpose:** Create one feature record per frame.

**用途：** 每個畫格建立一筆特徵。

**Inputs / 輸入：** graph and metadata

**Outputs / 輸出：** feature dictionary

### `compute_edge_stability()`

**Purpose:** Measure persistence of edges across frames.

**用途：** 衡量邊在不同畫格中的持續性。

**Inputs / 輸入：** current and previous edge sets

**Outputs / 輸出：** stability score

### `compute_capital_statistics()`

**Purpose:** Summarize edge capital allocation.

**用途：** 彙總邊上的資金配置。

**Inputs / 輸入：** capital-aware graph

**Outputs / 輸出：** capital metrics

## Summary / 摘要

This chapter connects the mathematical idea, the financial interpretation, and the software implementation used in Ricci-Finance v12.

本章將數學概念、金融意義與 Ricci-Finance v12 的程式實作連結在一起。
