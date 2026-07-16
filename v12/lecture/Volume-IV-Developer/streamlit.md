# Developer Guide — Streamlit UI

# 開發者指南：Streamlit UI

---

## Learning Objectives / 學習目標

### English

- Understand separation between computation and presentation.

### 繁體中文

- 理解計算與展示的分離。

## Design / 設計

### English

UI functions should receive prepared data and avoid recomputing expensive analysis.

### 繁體中文

UI 函式應接收已準備資料，避免重複進行昂貴運算。

## Functions Used / 使用函式

### `render_app()`

**Purpose:** Top-level UI entry.

**用途：** 最高層 UI 入口。

**Inputs / 輸入：** pipeline services

**Outputs / 輸出：** rendered app

### `render_diagnostics()`

**Purpose:** Display data and model warnings.

**用途：** 顯示資料與模型警告。

**Inputs / 輸入：** diagnostic records

**Outputs / 輸出：** UI messages

## Summary / 摘要

This chapter connects the mathematical idea, the financial interpretation, and the software implementation used in Ricci-Finance v12.

本章將數學概念、金融意義與 Ricci-Finance v12 的程式實作連結在一起。
