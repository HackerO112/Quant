# Quant — Technical Analysis & Research Dashboard

A Streamlit dashboard for exploring market data, technical indicators, heuristic market-condition scores, position-sizing scenarios, and historical simulations.

> **Research and education only.** The current signal engine is a rule-based scoring system, not the seven-model machine-learning ensemble described in earlier documentation. It does not execute trades and must not be treated as a buy/sell instruction.

## Current implementation

The signal score starts at 50 and adjusts according to available indicator conditions:

- Price relative to SMA20, SMA20/SMA50, and SMA50/SMA200
- RSI and MACD relative to its signal line
- ADX trend strength
- Relative volume and 20-bar rate of change
- Position within the Bollinger range

The score is clipped to 0–100 and grouped into:
- **65–100:** bullish / BUY BIAS
- **0–35:** bearish / SELL / AVOID BIAS
- **36–64:** sideways / mixed / WAIT / NEUTRAL

These are heuristic labels, not calibrated probabilities. They are not evidence of future returns.

## Quick start

Use Python 3.11 or a compatible version supported by the dependencies.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Data and limitations

- Market data is retrieved through Yahoo Finance via `yfinance`; availability, coverage, latency, and rate limits are controlled by that provider.
- Always check the instrument, exchange suffix, currency, timezone, adjustment settings, and timestamp before interpreting a chart.
- Missing or stale data can invalidate indicator values. Verify the underlying data independently.
- Historical simulations are not live execution. Review the code's assumptions before comparing results; transaction costs, taxes, slippage, liquidity, spreads, and market impact may not be fully represented.
- Position sizing depends on user-supplied capital, entry/stop/target inputs, and an estimated win probability. It is a mathematical illustration, not a recommendation.
- The application does not connect to a broker or place orders.

## Risk controls

The current sizing helper calculates a quarter-Kelly fraction and caps the risk budget at 2% of capital, then limits share count to available cash. This does not eliminate gap risk, execution risk, model error, or loss of capital. Do not use it as a substitute for an independently tested risk policy.

## Verification

The repository's CI currently installs the declared Python dependencies and checks Python syntax. It does **not** yet contain a quantitative test suite or independently validated backtest. See [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md) for the remaining release gates.

## Disclaimer

Nothing in this repository is financial, investment, tax, or legal advice. Market data and calculations may be incomplete, delayed, or wrong. You are responsible for independent verification and all investment decisions. Never risk money you cannot afford to lose.
