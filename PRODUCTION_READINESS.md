# Production readiness assessment

Status: **research/analytics prototype; not approved for live trading**.

## Verified from the current default branch
- Streamlit application uses Yahoo Finance data and technical/quantitative analysis.
- README describes a stacked ML ensemble and out-of-sample backtesting, while the current code includes a separate rule-based signal and backtest path.
- No test suite, CI workflow, dependency lockfile, or security policy was found.

## Release blockers
1. Reconcile README claims with the actual executable model path; publish a reproducible model/data lineage.
2. Add time-series walk-forward validation, leakage tests, survivorship-bias checks, realistic fees, slippage, spreads, corporate actions, and market-session handling.
3. Validate stale/missing/outlier data and provider failures; never silently turn missing values into tradeable signals.
4. Replace blanket warning suppression with targeted handling and structured logs.
5. Add deterministic tests for indicators, signal boundaries, position sizing, drawdown, and backtest accounting.
6. Clearly label outputs as research-only, show uncertainty and sample sizes, and prohibit automatic order execution in this project unless separately specified and secured.

## Release gate
No live-trading claims until independent quantitative validation, reproducible data, risk review, and test coverage exist. Past performance and model probabilities are not guarantees.
