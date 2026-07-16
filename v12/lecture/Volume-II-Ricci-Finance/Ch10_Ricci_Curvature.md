# Chapter 10 — Ricci Curvature Computation

# 第十章：Ricci 曲率計算

---

## Learning Objectives / 學習目標

### English

- Compute and validate curvature frame by frame.

### 繁體中文

- 逐畫格計算並驗證曲率。

## Numerical Considerations / 數值考量

### English

Disconnected graphs, tiny edge weights, and degenerate neighborhoods require safeguards.

### 繁體中文

非連通圖、極小邊權與退化鄰域都需要保護機制。

## Interpretation / 解讀

### English

Curvature must be read with density, components, and capital flow, not in isolation.

### 繁體中文

曲率必須與密度、連通元件與資金流一起解讀，不能孤立判斷。

## Functions Used / 使用函式

### `compute_ricci_frame()`

**Purpose:** Compute curvature for one frame.

**用途：** 計算單一畫格曲率。

**Inputs / 輸入：** graph, curvature config

**Outputs / 輸出：** curved graph

### `summarize_curvature()`

**Purpose:** Aggregate edge curvature statistics.

**用途：** 彙總邊曲率統計。

**Inputs / 輸入：** curved graph

**Outputs / 輸出：** mean, std, negative ratio

## Summary / 摘要

This chapter connects the mathematical idea, the financial interpretation, and the software implementation used in Ricci-Finance v12.

本章將數學概念、金融意義與 Ricci-Finance v12 的程式實作連結在一起。
