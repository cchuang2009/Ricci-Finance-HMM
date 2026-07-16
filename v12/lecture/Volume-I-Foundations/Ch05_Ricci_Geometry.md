# Chapter 5 — Ricci Geometry

# 第五章：Ricci 幾何

---

## Learning Objectives / 學習目標

### English

- Understand curvature as neighborhood contraction or expansion.
- Interpret positive and negative curvature in financial networks.
- Understand the role of Ricci Flow.

### 繁體中文

- 理解曲率如何描述鄰域收縮或擴張。
- 解讀金融網路中的正曲率與負曲率。
- 理解 Ricci Flow 的作用。

## Ollivier–Ricci Curvature / Ollivier–Ricci 曲率

### English

For an edge \((i,j)\),
         \[
         \kappa(i,j)=1-\frac{W_1(m_i,m_j)}{d(i,j)}.
         \]
         Positive curvature means nearby neighborhoods are similar and contract;
         negative curvature means they diverge and may form a bridge.

### 繁體中文

對一條邊 \((i,j)\)，
         \[
         \kappa(i,j)=1-\frac{W_1(m_i,m_j)}{d(i,j)}.
         \]
         正曲率表示鄰域相似且具有收縮性；負曲率則表示鄰域分離，可能形成橋接結構。

## Financial Interpretation / 金融解讀

### English

Positive curvature often appears inside stable sectors. Negative curvature
         often appears between sectors, during capital rotation, or when the network is
         under structural stress.

### 繁體中文

正曲率常出現在穩定產業群內；負曲率常出現在產業群之間、資金輪動期間，
         或市場網路承受結構壓力時。

## Ricci Flow / Ricci Flow

### English

Ricci Flow updates edge weights using curvature. In finance it is used as a
         geometric smoothing and projection tool, not as a guaranteed forecast.

### 繁體中文

Ricci Flow 依曲率更新邊權重。在金融應用中，它主要用於幾何平滑與投影，
         並不代表保證性的價格預測。

## Functions Used / 使用函式

### `compute_ricci()`

**Purpose:** Compute edge and node curvature.

**用途：** 計算邊與節點曲率。

**Inputs / 輸入：** weighted graph, alpha, method

**Outputs / 輸出：** graph with curvature attributes

### `compute_ricci_flow()`

**Purpose:** Iteratively evolve graph weights.

**用途：** 反覆演化圖形權重。

**Inputs / 輸入：** curved graph, iterations, step

**Outputs / 輸出：** flowed graph

### `compare_before_after_flow()`

**Purpose:** Compare original and flowed geometry.

**用途：** 比較原始與 Flow 後幾何。

**Inputs / 輸入：** two graphs

**Outputs / 輸出：** comparison statistics and plots

## Summary / 摘要

This chapter connects the mathematical idea, the financial interpretation, and the software implementation used in Ricci-Finance v12.

本章將數學概念、金融意義與 Ricci-Finance v12 的程式實作連結在一起。
