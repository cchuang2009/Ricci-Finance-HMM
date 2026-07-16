# Chapter 7 — Data Pipeline

# 第七章：資料流程

---

## Learning Objectives / 學習目標

### English

- Understand ingestion, cleaning, alignment, and caching.

### 繁體中文

- 理解資料擷取、清理、對齊與快取。

## Data Ingestion / 資料擷取

### English

v12 downloads OHLCV data and validates ticker availability, dates, and adjusted prices.

### 繁體中文

v12 下載 OHLCV 資料，並驗證股票代號、日期與還原價格。

## Alignment and Missing Data / 資料對齊與缺值

### English

Assets with different IPO dates require column-wise alignment and minimum-observation rules.

### 繁體中文

不同上市日期的資產需逐欄對齊，並設定最少觀測值規則。

## Functions Used / 使用函式

### `download_market_data()`

**Purpose:** Download and validate market data.

**用途：** 下載並驗證市場資料。

**Inputs / 輸入：** tickers, period, interval

**Outputs / 輸出：** aligned price/volume data

### `clean_market_data()`

**Purpose:** Remove unusable columns and handle missing values.

**用途：** 移除不可用欄位並處理缺值。

**Inputs / 輸入：** raw data

**Outputs / 輸出：** clean data

### `cache_market_data()`

**Purpose:** Reuse downloaded data across app reruns.

**用途：** 在 App 重跑時重用資料。

**Inputs / 輸入：** query parameters

**Outputs / 輸出：** cached dataset

## Summary / 摘要

This chapter connects the mathematical idea, the financial interpretation, and the software implementation used in Ricci-Finance v12.

本章將數學概念、金融意義與 Ricci-Finance v12 的程式實作連結在一起。
