# Developer Guide — Graph Module

# 開發者指南：Graph 模組

---

## Learning Objectives / 學習目標

### English

- Understand graph construction and feature extraction APIs.

### 繁體中文

- 理解圖形建構與特徵萃取 API。

## Responsibilities / 責任

### English

The graph module converts statistics into NetworkX graphs and computes topology features.

### 繁體中文

Graph 模組將統計資料轉成 NetworkX 圖，並計算拓撲特徵。

## Functions Used / 使用函式

### `build_weighted_graph()`

**Purpose:** Build graph.

**用途：** 建立圖。

**Inputs / 輸入：** matrices and metadata

**Outputs / 輸出：** graph

### `compute_components()`

**Purpose:** Connected-component statistics.

**用途：** 連通元件統計。

**Inputs / 輸入：** graph

**Outputs / 輸出：** component metrics

### `extract_graph_features()`

**Purpose:** Frame-level features.

**用途：** 畫格層級特徵。

**Inputs / 輸入：** graph

**Outputs / 輸出：** feature dictionary

## Summary / 摘要

This chapter connects the mathematical idea, the financial interpretation, and the software implementation used in Ricci-Finance v12.

本章將數學概念、金融意義與 Ricci-Finance v12 的程式實作連結在一起。
