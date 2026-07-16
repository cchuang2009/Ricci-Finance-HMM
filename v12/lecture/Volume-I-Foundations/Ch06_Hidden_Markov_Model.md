# Chapter 6 — Hidden Markov Models

# 第六章：隱藏馬可夫模型

---

## Learning Objectives / 學習目標

### English

- Understand hidden states, observations, transitions, and posterior probabilities.
- Understand why v12 applies HMM to rolling-window graph features.

### 繁體中文

- 理解隱藏狀態、觀測值、轉移機率與後驗機率。
- 理解 v12 為何將 HMM 套用於 Rolling Window 圖形特徵。

## Hidden States and Observations / 隱藏狀態與觀測值

### English

The hidden state may represent accumulation, expansion, rotation, stress,
         or recovery. The observations are graph-level features computed for each window.

### 繁體中文

隱藏狀態可代表吸籌、擴張、輪動、壓力或復甦；觀測值則是每個視窗所計算的圖形特徵。

## Why One Observation per Window? / 為何每個視窗是一筆觀測？

### English

Each rolling window summarizes one market geometry. Therefore the sequence of
         windows becomes a multivariate observation sequence for the HMM.

### 繁體中文

每個 Rolling Window 都概括一個市場幾何，因此視窗序列自然形成 HMM 的多變量觀測序列。

## Posterior Probabilities / 後驗機率

### English

The model returns not only one selected state, but also the probability of
         every state for every frame. This is important because market regimes are uncertain.

### 繁體中文

模型不只輸出單一狀態，也輸出每個畫格對所有狀態的機率。
         因為市場狀態具有不確定性，因此後驗機率比單一標籤更重要。

## Functions Used / 使用函式

### `build_feature_matrix()`

**Purpose:** Stack frame-level features into a time-ordered matrix.

**用途：** 將各畫格特徵依時間堆疊成矩陣。

**Inputs / 輸入：** frame statistics

**Outputs / 輸出：** T x F feature matrix

### `fit_hmm()`

**Purpose:** Estimate HMM parameters from graph features.

**用途：** 由圖形特徵估計 HMM 參數。

**Inputs / 輸入：** standardized features, state count

**Outputs / 輸出：** trained HMM

### `predict_states()`

**Purpose:** Assign the most likely state to each frame.

**用途：** 為每個畫格指定最可能狀態。

**Inputs / 輸入：** trained HMM, features

**Outputs / 輸出：** state sequence

### `posterior_probabilities()`

**Purpose:** Return state probabilities for each frame.

**用途：** 回傳每個畫格的狀態機率。

**Inputs / 輸入：** trained HMM, features

**Outputs / 輸出：** posterior matrix

## Summary / 摘要

This chapter connects the mathematical idea, the financial interpretation, and the software implementation used in Ricci-Finance v12.

本章將數學概念、金融意義與 Ricci-Finance v12 的程式實作連結在一起。
