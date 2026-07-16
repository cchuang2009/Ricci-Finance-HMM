# Chapter 2 — Financial Mathematics

# 第二章：金融數學

---

## Learning Objectives / 學習目標

### English

- Understand returns, covariance, correlation, volatility, and normalization.
- Connect financial statistics to graph construction.

### 繁體中文

- 理解報酬率、共變異數、相關係數、波動率與標準化。
- 將金融統計與圖形建構連結。

## Returns / 報酬率

### English

Simple return is \(r_t=P_t/P_{t-1}-1\). Log return is
         \(\log P_t-\log P_{t-1}\). Log returns are often preferred because they are
         additive over time and numerically stable for modeling.

### 繁體中文

簡單報酬率為 \(r_t=P_t/P_{t-1}-1\)，對數報酬率為
         \(\log P_t-\log P_{t-1}\)。對數報酬率可跨期相加，且在數值模型中通常更穩定。

## Covariance and Correlation / 共變異數與相關係數

### English

Covariance measures co-movement in original scale, while correlation
         normalizes by volatility:
         \[
         \rho_{ij}=\frac{\operatorname{Cov}(r_i,r_j)}{\sigma_i\sigma_j}.
         \]
         Correlation becomes the basis of the market network.

### 繁體中文

共變異數描述原始尺度下的共同變動，相關係數則以波動率標準化：
         \[
         \rho_{ij}=\frac{\operatorname{Cov}(r_i,r_j)}{\sigma_i\sigma_j}.
         \]
         相關矩陣是市場網路的基礎。

## Functions Used / 使用函式

### `compute_returns()`

**Purpose:** Convert prices to simple or log returns.

**用途：** 將價格轉成簡單或對數報酬率。

**Inputs / 輸入：** price DataFrame

**Outputs / 輸出：** return DataFrame

### `compute_correlation()`

**Purpose:** Estimate the correlation matrix in one window.

**用途：** 估計單一視窗內的相關矩陣。

**Inputs / 輸入：** return window

**Outputs / 輸出：** correlation matrix

### `standardize_features()`

**Purpose:** Scale features before probabilistic modeling.

**用途：** 在機率模型前標準化特徵。

**Inputs / 輸入：** feature matrix

**Outputs / 輸出：** standardized feature matrix

## Summary / 摘要

This chapter connects the mathematical idea, the financial interpretation, and the software implementation used in Ricci-Finance v12.

本章將數學概念、金融意義與 Ricci-Finance v12 的程式實作連結在一起。
