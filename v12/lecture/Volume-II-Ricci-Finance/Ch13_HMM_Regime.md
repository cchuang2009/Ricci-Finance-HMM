# Chapter 13 — Hidden Market Regimes

# 第十三章：隱藏市場狀態

---

## Learning Objectives / 學習目標

### English

- Fit, label, and interpret HMM regimes.

### 繁體中文

- 訓練、命名並解讀 HMM 狀態。

## State Estimation / 狀態估計

### English

The HMM learns emission distributions and a transition matrix from the feature sequence.

### 繁體中文

HMM 從特徵序列學習發射分布與轉移矩陣。

## State Naming / 狀態命名

### English

Numeric states should be labeled after training using return, curvature, stress, and flow statistics.

### 繁體中文

數字狀態應在訓練後，依報酬、曲率、壓力與資金流統計命名。

## Functions Used / 使用函式

### `select_hmm_features()`

**Purpose:** Choose robust model inputs.

**用途：** 選擇穩健的模型輸入。

**Inputs / 輸入：** feature table

**Outputs / 輸出：** selected matrix

### `fit_gaussian_hmm()`

**Purpose:** Train a Gaussian HMM.

**用途：** 訓練 Gaussian HMM。

**Inputs / 輸入：** features, n_states

**Outputs / 輸出：** model

### `label_regimes()`

**Purpose:** Map numeric states to readable names.

**用途：** 將數字狀態映射為可讀名稱。

**Inputs / 輸入：** state summaries

**Outputs / 輸出：** label mapping

## Summary / 摘要

This chapter connects the mathematical idea, the financial interpretation, and the software implementation used in Ricci-Finance v12.

本章將數學概念、金融意義與 Ricci-Finance v12 的程式實作連結在一起。
