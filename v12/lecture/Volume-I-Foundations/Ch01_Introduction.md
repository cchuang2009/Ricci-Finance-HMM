# Chapter 1 — Introduction to Ricci-Finance

# 第一章：Ricci-Finance 導論

---

## Learning Objectives / 學習目標

### English

- Understand why financial markets are modeled as evolving networks.
- Distinguish price prediction from structural market analysis.
- Understand the role of geometry, probability, and visualization.

### 繁體中文

- 理解為何金融市場適合建模為動態網路。
- 區分價格預測與市場結構分析。
- 理解幾何、機率與視覺化在本專案中的角色。

## Why Geometry? / 為何使用幾何？

### English

A market contains many interacting assets. Correlation alone gives pairwise
         similarity, but geometry reveals neighborhoods, bridges, clusters, and structural
         stress. Ricci-Finance treats the market as an evolving graph rather than a table
         of unrelated prices.

### 繁體中文

金融市場由許多彼此互動的資產構成。相關係數只能描述兩兩關係，
         幾何方法則能揭示鄰域、橋接節點、群聚以及結構壓力。Ricci-Finance
         將市場視為一個持續演化的圖，而不是互不相關的價格表。

## Project Pipeline / 專案流程

### English

The project transforms prices into returns, returns into correlation,
         correlation into graph distance, and graph distance into curvature. Rolling-window
         features are then modeled by an HMM to estimate latent market regimes.

### 繁體中文

專案先將價格轉為報酬率，再將報酬率轉為相關矩陣，接著建立圖距離與曲率。
         之後，以 Rolling Window 產生的圖形特徵作為 HMM 觀測值，推估隱藏市場狀態。

## Functions Used / 使用函式

### `run_pipeline()`

**Purpose:** Coordinate the complete analysis workflow.

**用途：** 協調完整分析流程。

**Inputs / 輸入：** configuration, tickers, dates

**Outputs / 輸出：** graphs, features, regimes, figures

### `build_frame()`

**Purpose:** Create one market graph for one rolling window.

**用途：** 為單一 Rolling Window 建立市場圖。

**Inputs / 輸入：** price window, parameters

**Outputs / 輸出：** graph frame

## Summary / 摘要

This chapter connects the mathematical idea, the financial interpretation, and the software implementation used in Ricci-Finance v12.

本章將數學概念、金融意義與 Ricci-Finance v12 的程式實作連結在一起。
