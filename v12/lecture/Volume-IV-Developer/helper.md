# Developer Guide — Helper Utilities

# 開發者指南：輔助工具

---

## Learning Objectives / 學習目標

### English

- Understand shared utility responsibilities.

### 繁體中文

- 理解共用工具的責任。

## Scope / 範圍

### English

Helpers handle validation, formatting, dates, logging, and safe numerical operations.

### 繁體中文

Helper 負責驗證、格式化、日期、紀錄與安全數值運算。

## Functions Used / 使用函式

### `safe_divide()`

**Purpose:** Avoid division-by-zero failures.

**用途：** 避免除以零。

**Inputs / 輸入：** numerator, denominator

**Outputs / 輸出：** safe result

### `format_metric()`

**Purpose:** Format values for UI.

**用途：** 格式化 UI 數值。

**Inputs / 輸入：** value, precision

**Outputs / 輸出：** string

### `validate_dataframe()`

**Purpose:** Check required columns and minimum rows.

**用途：** 檢查必要欄位與最少列數。

**Inputs / 輸入：** DataFrame, schema

**Outputs / 輸出：** validated DataFrame

## Summary / 摘要

This chapter connects the mathematical idea, the financial interpretation, and the software implementation used in Ricci-Finance v12.

本章將數學概念、金融意義與 Ricci-Finance v12 的程式實作連結在一起。
