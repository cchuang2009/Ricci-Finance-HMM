# Developer API Reference

# 開發者 API 參考

---

## Learning Objectives / 學習目標

### English

- Understand the public API and data contracts.

### 繁體中文

- 理解公開 API 與資料契約。

## API Principles / API 原則

### English

Functions should be small, typed, deterministic where possible, and explicit about errors.

### 繁體中文

函式應保持小型、具型別、盡量可重現，並明確處理錯誤。

## Data Contracts / 資料契約

### English

Core objects include price tables, graphs, frame records, feature tables, and HMM outputs.

### 繁體中文

核心物件包含價格表、圖形、畫格紀錄、特徵表與 HMM 輸出。

## Functions Used / 使用函式

### `run_pipeline()`

**Purpose:** Top-level orchestration.

**用途：** 最高層流程協調。

**Inputs / 輸入：** PipelineConfig

**Outputs / 輸出：** PipelineResult

### `validate_config()`

**Purpose:** Validate all user settings.

**用途：** 驗證所有使用者設定。

**Inputs / 輸入：** PipelineConfig

**Outputs / 輸出：** validated configuration or error

## Summary / 摘要

This chapter connects the mathematical idea, the financial interpretation, and the software implementation used in Ricci-Finance v12.

本章將數學概念、金融意義與 Ricci-Finance v12 的程式實作連結在一起。
