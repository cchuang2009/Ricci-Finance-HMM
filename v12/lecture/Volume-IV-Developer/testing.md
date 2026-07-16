# Developer Guide — Testing

# 開發者指南：測試

---

## Learning Objectives / 學習目標

### English

- Design unit, integration, regression, and numerical tests.

### 繁體中文

- 設計單元、整合、回歸與數值測試。

## Test Strategy / 測試策略

### English

Test formulas on small graphs, validate frame alignment, and protect against HMM numerical failures.

### 繁體中文

以小圖測試公式、驗證畫格對齊，並防止 HMM 數值失敗。

## Functions Used / 使用函式

### `test_correlation_distance()`

**Purpose:** Verify metric conversion.

**用途：** 驗證度量轉換。

**Inputs / 輸入：** known correlation values

**Outputs / 輸出：** assertions

### `test_frame_alignment()`

**Purpose:** Verify dates and windows.

**用途：** 驗證日期與視窗。

**Inputs / 輸入：** synthetic time index

**Outputs / 輸出：** assertions

### `test_hmm_covariance()`

**Purpose:** Verify positive-definite covariance safeguards.

**用途：** 驗證正定共變異數保護。

**Inputs / 輸入：** degenerate features

**Outputs / 輸出：** stable model fit

## Summary / 摘要

This chapter connects the mathematical idea, the financial interpretation, and the software implementation used in Ricci-Finance v12.

本章將數學概念、金融意義與 Ricci-Finance v12 的程式實作連結在一起。
