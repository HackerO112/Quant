# ⚡ Black-Box Quant AI Engine

An institutional-grade, real-time quantitative trading workstation powered by Streamlit, Plotly, and an ensemble of **7 Machine Learning algorithms** (XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees, Gradient Boosting, and Logistic Regression).

The application processes live Yahoo Finance data, constructs volatility and technical indicators, executes meta-learning stacked consensus inference, and computes position sizing using fractional Kelly Criterion and ATR-based risk parameters.

---

## 🔑 Key Features

* **Auto-Dependency Management**: Self-checks and installs required libraries (`yfinance`, `xgboost`, `catboost`, etc.) at runtime automatically.
* **Stacked ML Ensemble (7 Models)**: Trains base classifiers and uses stacked meta-logistic regression to output a single unified `BUY`, `SELL`, or `HOLD` signal with confidence metrics.
* **High-Precision Feature Engineering**: Computes Garman-Klass volatility, ATR %, Normalized RSI-Z, MACD-Z, Volume Z-Scores, and Historical Volatility.
* **Kelly & ATR Risk Sizer**: Calculates recommended share quantities, dollar risk limits, target prices, and stop losses dynamically.
* **Interactive Financial Workstation**:
* **Out-of-Sample Backtesting**: Equity curve comparison against Buy & Hold strategy.
* **Precision Candlestick Charts**: Dual-panel Plotly layout featuring moving averages and volume action.
* **Financial Statements**: Dynamic inspection of Income Statements, Balance Sheets, and Cash Flow metrics.


* **High-Contrast Monochromatic UI**: Optimized custom CSS interface designed for rapid visual scanning.

---

## 🛠️ Tech Stack & Dependencies

The application relies on the following core Python libraries:

| Category | Libraries |
| --- | --- |
| **Frontend / Web UI** | `streamlit` |
| **Data Fetching & Processing** | `yfinance`, `pandas`, `numpy`, `scipy`, `statsmodels` |
| **Machine Learning & Stats** | `scikit-learn`, `xgboost`, `lightgbm`, `catboost` |
| **Data Visualization** | `plotly` |

---

## 🚀 Quick Start Guide

### Prerequisites

* Python 3.9, 3.10, or 3.11 installed.
* `pip` package installer updated.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/blackbox-quant-ai.git
cd blackbox-quant-ai

```

### 2. Set Up a Virtual Environment (Recommended)

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate

```

### 3. Install Core Dependencies

*(Note: The script includes auto-install routines for core libraries, but pre-installing dependencies is recommended).*

```bash
pip install streamlit yfinance plotly xgboost lightgbm catboost scikit-learn scipy statsmodels

```

### 4. Run the Streamlit Application

```bash
streamlit run app.py

```

---

## ⚙️ How It Works

### Data & Feature Pipeline

1. **Price Retrieval**: Pulls 3 years of OHLCV history via `yfinance` with execution latency tracking.
2. **Feature Generation**:
* **Volatility**: Garman-Klass estimator & 20-day annualized historical volatility.
* **Momentum & Trend**: ATR %, RSI Z-Score, MACD Z-Score, and Volume Ratio Z-Score.


3. **Target Engineering**: Multi-day horizon targets based on volatility-adjusted ATR thresholds.

### Black-Box Ensemble Mechanics

```
 Raw Market Data
        │
        ▼
 Technical & Volatility Feature Extraction
        │
        ▼
 SelectKBest (Mutual Information) + StandardScaler
        │
 ┌──────┴────────────────────────────────────────┐
 │ Base Classifiers (XGB, LGBM, Cat, RF, ET, GB, LR) │
 └──────┬────────────────────────────────────────┘
        │
        ▼
 Meta Logistic Regression Stack
        │
        ▼
 Unified Signal (BUY / SELL / HOLD) + Confidence %

```

---

## 📁 Project Structure

```text
├── app.py              # Main application script containing UI, data pipelines, & ML models
├── README.md           # Documentation
└── requirements.txt    # (Optional) Dependency list for deterministic builds

```

---

## ⚠️ Disclaimer

This software is for **educational and quantitative research purposes only**. It does not constitute financial advice, investment strategy recommendations, or trade execution guarantees. Quantitative trading involves significant financial risk.
