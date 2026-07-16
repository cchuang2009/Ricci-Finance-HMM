# Case Study — COVID-19 Market Shock

# 案例研究：COVID-19 市場衝擊

---

## Learning Objectives / 學習目標

### English

- Apply the complete pipeline to a defined market episode.
- Separate observed evidence from interpretation.

### 繁體中文

- 將完整流程套用於特定市場事件。
- 區分觀測證據與主觀解讀。

## Research Question / 研究問題

### English

How did network geometry, capital structure, and HMM regime behavior change during COVID-19 Market Shock?

### 繁體中文

在COVID-19 市場衝擊期間，網路幾何、資金結構與 HMM 狀態如何改變？

## Required Analysis / 必要分析

### English

Compare pre-event, event, and post-event windows; inspect curvature, density, components, capital concentration, and posterior probabilities.

### 繁體中文

比較事件前、事件中與事件後視窗，檢查曲率、密度、連通元件、資金集中度與後驗機率。

## Interpretation Rules / 解讀規則

### English

Do not claim causality from curvature alone. Use prices, volume, event dates, and model diagnostics as supporting evidence.

### 繁體中文

不可僅憑曲率宣稱因果關係；必須搭配價格、成交量、事件日期與模型診斷。

## Functions Used / 使用函式

### `run_case_study()`

**Purpose:** Execute a reproducible event study.

**用途：** 執行可重現的事件研究。

**Inputs / 輸入：** tickers, dates, config

**Outputs / 輸出：** case-study result

### `compare_periods()`

**Purpose:** Compare before, during, and after periods.

**用途：** 比較事件前、中、後期間。

**Inputs / 輸入：** frame table, period labels

**Outputs / 輸出：** comparison table

### `export_case_report()`

**Purpose:** Export figures and tables.

**用途：** 匯出圖表與表格。

**Inputs / 輸入：** case-study result

**Outputs / 輸出：** report assets

## Exercises / 練習

1. Change the event window and test robustness.
   - 改變事件視窗並測試穩健性。
2. Identify one alternative explanation.
   - 提出一項替代解釋。

## Summary / 摘要

This chapter connects the mathematical idea, the financial interpretation, and the software implementation used in Ricci-Finance v12.

本章將數學概念、金融意義與 Ricci-Finance v12 的程式實作連結在一起。
