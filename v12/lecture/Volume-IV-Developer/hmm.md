# Developer Guide — HMM Module

# 開發者指南：HMM 模組

---

## Learning Objectives / 學習目標

### English

- Understand HMM preparation, fitting, and interpretation.

### 繁體中文

- 理解 HMM 資料準備、訓練與解讀。

## Responsibilities / 責任

### English

The HMM module standardizes selected features, fits the model, and returns states and probabilities.

### 繁體中文

HMM 模組標準化所選特徵、訓練模型，並回傳狀態與機率。

## Functions Used / 使用函式

### `prepare_hmm_features()`

**Purpose:** Clean and scale model inputs.

**用途：** 清理並縮放模型輸入。

**Inputs / 輸入：** feature table

**Outputs / 輸出：** matrix and scaler

### `fit_gaussian_hmm()`

**Purpose:** Fit model with robust covariance handling.

**用途：** 以穩健共變異數處理訓練模型。

**Inputs / 輸入：** matrix, config

**Outputs / 輸出：** trained model

### `predict_hmm_outputs()`

**Purpose:** Return states and posterior probabilities.

**用途：** 回傳狀態與後驗機率。

**Inputs / 輸入：** model, matrix

**Outputs / 輸出：** HMM result

## Summary / 摘要

This chapter connects the mathematical idea, the financial interpretation, and the software implementation used in Ricci-Finance v12.

本章將數學概念、金融意義與 Ricci-Finance v12 的程式實作連結在一起。
