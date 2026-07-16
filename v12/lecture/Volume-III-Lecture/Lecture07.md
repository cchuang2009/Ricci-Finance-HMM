# Lecture 7 — Hidden Markov Models

# 第七講：隱藏馬可夫模型

---

## Learning Objectives / 學習目標

### English

- Use the sequence of graph features to estimate hidden market regimes.

### 繁體中文

- 利用圖形特徵序列推估隱藏市場狀態。

## Lecture Goal / 課程目標

### English

This lecture focuses on the following goal: Use the sequence of graph features to estimate hidden market regimes.

### 繁體中文

本講重點為：利用圖形特徵序列推估隱藏市場狀態。

## Teaching Workflow / 教學流程

### English

Concept introduction → function walkthrough → live demonstration → result interpretation → exercise.

### 繁體中文

概念導入 → 函式說明 → 即時示範 → 結果解讀 → 練習。

## Expected Outcome / 預期成果

### English

Students should be able to reproduce the analysis and explain what the output means.

### 繁體中文

學生應能重現分析，並說明輸出結果的意義。

## Functions Used / 使用函式

### `build_feature_matrix()`

**Purpose:** Builds HMM observations.

**用途：** 建立 HMM 觀測矩陣。

**Inputs / 輸入：** lecture data and configuration

**Outputs / 輸出：** result used in the lecture

### `fit_gaussian_hmm()`

**Purpose:** Trains the HMM.

**用途：** 訓練 HMM。

**Inputs / 輸入：** lecture data and configuration

**Outputs / 輸出：** result used in the lecture

### `posterior_probabilities()`

**Purpose:** Computes state probabilities.

**用途：** 計算狀態機率。

**Inputs / 輸入：** lecture data and configuration

**Outputs / 輸出：** result used in the lecture

## Exercises / 練習

1. Change one major parameter and explain the effect.
   - 修改一個主要參數並解釋影響。
2. Identify one limitation of the method.
   - 指出此方法的一項限制。

## Summary / 摘要

This chapter connects the mathematical idea, the financial interpretation, and the software implementation used in Ricci-Finance v12.

本章將數學概念、金融意義與 Ricci-Finance v12 的程式實作連結在一起。
