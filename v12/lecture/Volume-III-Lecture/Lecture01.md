# Lecture 1 — Market Data and Rolling Windows

# 第一講：市場資料與滾動視窗

---

## Learning Objectives / 學習目標

### English

- Build reliable market data and understand how one window becomes one frame.

### 繁體中文

- 建立可靠市場資料，並理解一個視窗如何形成一個畫格。

## Lecture Goal / 課程目標

### English

This lecture focuses on the following goal: Build reliable market data and understand how one window becomes one frame.

### 繁體中文

本講重點為：建立可靠市場資料，並理解一個視窗如何形成一個畫格。

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

### `download_market_data()`

**Purpose:** Downloads OHLCV data.

**用途：** 下載 OHLCV 資料。

**Inputs / 輸入：** lecture data and configuration

**Outputs / 輸出：** result used in the lecture

### `compute_returns()`

**Purpose:** Creates return series.

**用途：** 建立報酬率序列。

**Inputs / 輸入：** lecture data and configuration

**Outputs / 輸出：** result used in the lecture

### `iter_rolling_windows()`

**Purpose:** Generates overlapping windows.

**用途：** 產生重疊視窗。

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
