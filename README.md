<div align="center">

# ⚡ Black-Box Quant AI Engine

**An institutional-grade, single-consensus Machine Learning trading dashboard built entirely in Streamlit.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Ensemble-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-purple.svg)](https://xgboost.ai)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Zero-config setup • 7-Model Stacked Ensemble • Kelly Criterion Position Sizing • Sub-millisecond Execution Tracking*

[🚀 Get Started](#-quick-start--zero-config) • [🧠 Architecture](#-black-box-architecture) • [🧮 Features](#-core-features) • [⚠️ Disclaimer](#-disclaimer)

</div>

---

## 🌟 Overview

The Black-Box Quant AI Engine moves beyond simple technical analysis. It ingests raw market data, engineers high-precision financial features (including Garman-Klass volatility and Z-score normalized indicators), and passes them through a stacked ensemble of **7 distinct machine learning algorithms**. 

The models vote, a meta-learner calculates the final probability, and the system outputs a single, definitive **BUY**, **SELL**, or **HOLD** signal—complete with Kelly Criterion-based position sizing and an out-of-sample equity curve.

## 🚀 Quick Start (Zero-Config)

You **only** need Streamlit installed. The application automatically detects, installs, and imports all other complex dependencies (XGBoost, LightGBM, CatBoost, etc.) at runtime.

```bash
# 1. Install Streamlit (the only manual requirement)
pip install streamlit

# 2. Run the application
streamlit run app.py
```

*(Note: First run may take 10-20 seconds as it auto-installs the ML libraries via pip.)*

---

## 🧠 Black-Box Architecture

The engine avoids relying on a single model, which is prone to overfitting in financial markets. Instead, it uses a **Two-Layer Stacking Ensemble**.

### Layer 1: Base Learners (7 Models)
All base models are heavily regularized (`max_depth=2`, `learning_rate=0.01`) to prevent curve-fitting:
1. **XGBoost** (Gradient Boosting)
2. **LightGBM** (Leaf-wise Gradient Boosting)
3. **CatBoost** (Ordered Boosting)
4. **Random Forest** (Bagging)
5. **Extra Trees** (Extremely Randomized Trees)
6. **Scikit-Learn Gradient Boosting**
7. **Logistic Regression** (Linear baseline)

### Layer 2: Meta-Learner
A regularized **Logistic Regression** model takes the prediction probabilities from all 7 base models as inputs and learns how to best weigh their opinions to generate the final Master Probability.

### Signal Logic
* `Master Probability ≥ 0.55` → **BUY** 🚀
* `Master Probability ≤ 0.45` → **SELL** 🔻
* Between 0.45 and 0.55 → **HOLD** ⚖️

---

## 🧮 Core Features

### 1. High-Precision Feature Pipeline
The app doesn't just feed raw prices to ML models. It calculates advanced mathematical features:
* **Garman-Klass Volatility**: More precise than standard historical volatility by incorporating intra-day high/low ranges.
* **Z-Score Normalization**: RSI, MACD, and Volume are normalized against their 50-day rolling mean/std to provide mean-reversion context rather than raw oscillating values.
* **ATR-Based Asymmetric Targets**: The ML target variable isn't just "up or down." It defines a bullish signal only if the asset moves up by `1.2x ATR` *without* violating a downside `1.2x ATR` stop.

### 2. Kelly Criterion Position Sizer
Replaces arbitrary share sizing with quantitative risk management:
* Calculates Win Probability based on Out-of-Sample (OOS) accuracy.
* Applies the **Half-Kelly Fraction** to prevent ruin.
* Cross-references with user-defined max risk % and ATR-based stop distance to output exact share allocation, target price, and stop loss.

### 3. Out-of-Sample (OOS) Backtesting
* The data is strictly split 80/20 (Chronological, no look-ahead bias).
* Models are trained on the past 80% and evaluated on the most recent 20%.
* Plots the Strategy Equity Curve vs. a Buy & Hold benchmark.

### 4. Precision Charting & Financials
* Interactive Plotly candlestick charts with 20/50 SMA overlays and colored volume bars.
* Direct API pulls for Income Statements, Balance Sheets, and Cash Flow statements.

---

## 📖 Usage Guide

1. **Select an Asset**: Choose from the preset list (NVDA, AAPL, TSLA, etc.) or type any valid Yahoo Finance ticker (e.g., `BTC-USD`, `RELIANCE.NS`).
2. **Configure Risk**: Set your total Portfolio Capital and Maximum Risk per Trade (%) in the sidebar.
3. **Read the Signal**: The giant dashboard banner is your single consensus AI decision.
4. **Size the Trade**: Go to the 🧮 *Position Sizer* tab. Enter your desired Reward/Risk ratio to see exactly how many shares to buy and where to set your stop/target.
5. **Validate**: Check the 📈 *Backtest Performance* tab to ensure the model is currently performing well out-of-sample on this specific asset.

---

## 🎨 UI Theme
The application features a custom "Balanced Monochromatic" dark theme engineered for high-visibility in low-light trading environments. Text is forced to pure crisp white (`#FFFFFF`) against medium slate grey (`#1E293B`), with vivid emerald/rose accents for instant signal recognition.

---

## ⚠️ Disclaimer

**This software is for educational and quantitative research purposes only.** 
It does not constitute financial advice, solicitation, or recommendation to buy or sell any securities. Machine learning models in finance are prone to overfitting and regime changes. Always paper-trade before risking real capital. Past model accuracy (OOS) does not guarantee future performance.

---
<div align="center">
Made with ⚡ by a Quant, for Quants.
</div>
```
