# Chapter 14 — Streamlit Application

# 第十四章：Streamlit 應用程式

---

## Learning Objectives / 學習目標

### English

- Understand application tabs, controls, caching, and visualization.

### 繁體中文

- 理解應用分頁、控制項、快取與視覺化。

## Application Design / 應用設計

### English

The app separates data, geometry, HMM, animation, and diagnostics into focused views.

### 繁體中文

App 將資料、幾何、HMM、動畫與診斷分成不同視圖。

## Interactive Analysis / 互動分析

### English

Users can change tickers, windows, thresholds, HMM states, and animation frames.

### 繁體中文

使用者可調整股票、視窗、門檻、HMM 狀態數與動畫畫格。

## Functions Used / 使用函式

### `render_sidebar()`

**Purpose:** Collect user configuration.

**用途：** 收集使用者設定。

**Inputs / 輸入：** Streamlit widgets

**Outputs / 輸出：** config

### `render_network_tab()`

**Purpose:** Display network and curvature results.

**用途：** 顯示網路與曲率結果。

**Inputs / 輸入：** analysis output

**Outputs / 輸出：** interactive plots

### `render_hmm_tab()`

**Purpose:** Display states and posterior probabilities.

**用途：** 顯示狀態與後驗機率。

**Inputs / 輸入：** HMM output

**Outputs / 輸出：** charts and tables

### `render_animation_tab()`

**Purpose:** Play frame evolution.

**用途：** 播放畫格演化。

**Inputs / 輸入：** frames

**Outputs / 輸出：** animation

## Summary / 摘要

This chapter connects the mathematical idea, the financial interpretation, and the software implementation used in Ricci-Finance v12.

本章將數學概念、金融意義與 Ricci-Finance v12 的程式實作連結在一起。
