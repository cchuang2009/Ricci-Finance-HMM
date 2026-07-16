# Chapter 9 — Network Construction

# 第九章：網路建構

---

## Learning Objectives / 學習目標

### English

- Build a financially meaningful weighted graph.

### 繁體中文

- 建立具有金融意義的加權圖。

## Thresholding / 門檻化

### English

Weak correlations may be removed to reduce noise and improve interpretability.

### 繁體中文

可移除弱相關以降低雜訊並提高可解釋性。

## Capital-aware Weights / 資金感知權重

### English

Volume or dollar-volume can modify edge strength and distance.

### 繁體中文

成交量或成交金額可修正邊強度與距離。

## Functions Used / 使用函式

### `correlation_to_distance()`

**Purpose:** Convert correlation to metric distance.

**用途：** 將相關係數轉成度量距離。

**Inputs / 輸入：** correlation matrix

**Outputs / 輸出：** distance matrix

### `apply_capital_weight()`

**Purpose:** Adjust edges using capital information.

**用途：** 以資金資訊修正邊。

**Inputs / 輸入：** graph, volume/capital matrix

**Outputs / 輸出：** capital-aware graph

### `build_weighted_graph()`

**Purpose:** Create graph with all edge attributes.

**用途：** 建立含完整邊屬性的圖。

**Inputs / 輸入：** distance, threshold, metadata

**Outputs / 輸出：** weighted graph

## Summary / 摘要

This chapter connects the mathematical idea, the financial interpretation, and the software implementation used in Ricci-Finance v12.

本章將數學概念、金融意義與 Ricci-Finance v12 的程式實作連結在一起。
