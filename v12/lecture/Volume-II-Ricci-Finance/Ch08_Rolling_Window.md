# Chapter 8 — Rolling Windows

# 第八章：滾動視窗

---

## Learning Objectives / 學習目標

### English

- Understand window length, stride, overlap, and frame dates.

### 繁體中文

- 理解視窗長度、步長、重疊與畫格日期。

## Window Design / 視窗設計

### English

Window length controls statistical stability; stride controls animation resolution.

### 繁體中文

視窗長度控制統計穩定性；步長控制動畫解析度。

## Frame Semantics / 畫格意義

### English

A frame represents a historical interval, not a single trading day.

### 繁體中文

一個畫格代表一段歷史區間，而不是單一交易日。

## Functions Used / 使用函式

### `iter_rolling_windows()`

**Purpose:** Yield time-ordered windows.

**用途：** 依時間輸出滾動視窗。

**Inputs / 輸入：** data, length, stride

**Outputs / 輸出：** window iterator

### `select_frame_dates()`

**Purpose:** Assign representative dates to frames.

**用途：** 為畫格指定代表日期。

**Inputs / 輸入：** window index

**Outputs / 輸出：** frame date

## Summary / 摘要

This chapter connects the mathematical idea, the financial interpretation, and the software implementation used in Ricci-Finance v12.

本章將數學概念、金融意義與 Ricci-Finance v12 的程式實作連結在一起。
