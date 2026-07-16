# Lecture 2 — Correlation Networks

# 第二講：相關性網路

---

## Learning Objectives / 學習目標

### English

- Convert returns into a weighted financial network.

### 繁體中文

- 將報酬率轉換成加權金融網路。

## Lecture Goal / 課程目標

### English

This lecture focuses on the following goal: Convert returns into a weighted financial network.

### 繁體中文

本講重點為：將報酬率轉換成加權金融網路。

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

### `compute_correlation()`

**Purpose:** Estimates pairwise correlation.

**用途：** 估計兩兩相關係數。

**Inputs / 輸入：** lecture data and configuration

**Outputs / 輸出：** result used in the lecture

### `correlation_to_distance()`

**Purpose:** Creates graph distance.

**用途：** 建立圖距離。

**Inputs / 輸入：** lecture data and configuration

**Outputs / 輸出：** result used in the lecture

### `build_weighted_graph()`

**Purpose:** Builds the market graph.

**用途：** 建立市場圖。

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
