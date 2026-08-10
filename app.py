# app.py
# Quant AI — Streamlit + yfinance
#
# Install:
#   pip install streamlit yfinance pandas numpy plotly scipy
#
# Run:
#   streamlit run app.py

import math
import textwrap
import warnings
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import os
import json
from pathlib import Path

warnings.filterwarnings("ignore")


def render_html(html: str) -> None:
    """Render an HTML block, stripping the indentation and blank lines that
    would otherwise make Streamlit's markdown parser treat it as a code block."""
    cleaned = "\n".join(
        line for line in textwrap.dedent(html).splitlines() if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


# ============================================================
# PAGE CONFIGURATION & CUSTOM HIGH-FIDELITY CSS UI
# ============================================================

st.set_page_config(
    page_title="Quant AI",
    page_icon="⚡",
    layout="wide",
    # "auto" keeps the sidebar open on desktop but collapsed on phones, where it
    # would otherwise cover the whole dashboard.
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0d1312 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #e2e8f0;
    }

    /* Hide default header element overhead */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #090e0d !important;
        border-right: 1px solid #1a2624 !important;
    }

    /* Glass Cards */
    .lb-card {
        background: #121a18;
        border: 1px solid #1c2b27;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }

    /* Native bordered containers styled like glass cards */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]) {
        background: #121a18;
        border: 1px solid #1c2b27 !important;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }

    .lb-card-sm {
        background: #121a18;
        border: 1px solid #1c2b27;
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }

    .lb-accent-btn {
        background-color: #b0ff22 !important;
        color: #090e0d !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 10px 20px !important;
        transition: all 0.2s ease-in-out;
    }

    .lb-accent-btn:hover {
        background-color: #9ee61a !important;
        box-shadow: 0 0 15px rgba(176, 255, 34, 0.4);
    }

    /* Typography helpers */
    .lb-title {
        color: #ffffff;
        font-weight: 700;
        font-size: 24px;
        margin-bottom: 4px;
    }

    .lb-subtitle {
        color: #6b7c77;
        font-size: 13px;
        font-weight: 500;
    }

    .lb-green-text {
        color: #b0ff22 !important;
        font-weight: 600;
    }

    .lb-red-text {
        color: #ff5252 !important;
        font-weight: 600;
    }

    .lb-val {
        font-size: 26px;
        font-weight: 700;
        color: #ffffff;
    }

    .lb-val-sm {
        font-size: 18px;
        font-weight: 600;
        color: #ffffff;
    }

    /* Input Fields Override */
    input, textarea, div[data-baseweb="select"] > div {
        background-color: #16221f !important;
        border: 1px solid #233631 !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }

    label {
        color: #8fa39d !important;
        font-size: 13px !important;
    }

    /* Dataframe Styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid #1c2b27;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Expander Override */
    .streamlit-expanderHeader,
    div[data-testid="stExpander"] details {
        background-color: #121a18 !important;
        border-radius: 12px !important;
        border: 1px solid #1c2b27 !important;
        color: #e2e8f0 !important;
    }

    /* Sidebar section labels */
    .lb-nav-label {
        color: #6b7c77;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.4px;
        margin: 18px 0 2px;
        padding-bottom: 6px;
        border-bottom: 1px solid #1a2624;
    }

    /* Top bar */
    .lb-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 14px;
    }
    .lb-topbar-left, .lb-topbar-right {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .lb-status-chip {
        display: flex;
        align-items: center;
        gap: 8px;
        background: #121a18;
        border: 1px solid #1c2b27;
        padding: 7px 14px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
        color: #e2e8f0;
    }
    .lb-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        box-shadow: 0 0 8px currentColor;
    }
    .lb-ts { font-size: 12px; color: #6b7c77; }
    .lb-chip {
        display: flex;
        flex-direction: column;
        background: #121a18;
        border: 1px solid #1c2b27;
        border-radius: 12px;
        padding: 8px 14px;
        line-height: 1.35;
    }
    .lb-chip-val { font-size: 14px; font-weight: 600; color: #fff; }
    .lb-pill {
        padding: 10px 18px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.4px;
    }
    .lb-pill-green { background: rgba(176,255,34,0.12); color: #b0ff22; border: 1px solid rgba(176,255,34,0.35); }
    .lb-pill-red { background: rgba(255,82,82,0.12); color: #ff5252; border: 1px solid rgba(255,82,82,0.35); }
    .lb-pill-neutral { background: rgba(245,158,11,0.12); color: #f59e0b; border: 1px solid rgba(245,158,11,0.35); }

    /* KPI strip */
    .lb-kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 12px;
        margin-bottom: 18px;
    }
    .lb-kpi {
        background: #121a18;
        border: 1px solid #1c2b27;
        border-radius: 14px;
        padding: 14px 16px;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .lb-kpi:hover { border-color: #2c403b; transform: translateY(-2px); }
    .lb-kpi-val { font-size: 20px; font-weight: 700; color: #fff; margin-top: 4px; }
    .lb-kpi-delta { font-size: 12px; margin-top: 2px; }

    /* Chart legend */
    .lb-legend { display: flex; gap: 14px; font-size: 12px; color: #8fa39d; }
    .lb-legend i {
        display: inline-block;
        width: 10px; height: 3px;
        border-radius: 2px;
        margin-right: 6px;
        vertical-align: middle;
    }

    /* Stat lists */
    .lb-stat-list { display: flex; flex-direction: column; }
    .lb-stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 9px 0;
        border-bottom: 1px dashed #1c2b27;
        font-size: 14px;
        font-weight: 600;
    }
    .lb-stat-row:last-child { border-bottom: none; }

    /* Model factor rows */
    .lb-factor-list { display: flex; flex-direction: column; }
    .lb-factor {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 9px 0;
        border-bottom: 1px dashed #1c2b27;
    }
    .lb-factor:last-child { border-bottom: none; }
    .lb-factor-badge {
        flex: 0 0 auto;
        min-width: 42px;
        text-align: center;
        padding: 4px 8px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
    }
    .lb-factor-up { background: rgba(176,255,34,0.12); color: #b0ff22; border: 1px solid rgba(176,255,34,0.3); }
    .lb-factor-down { background: rgba(255,82,82,0.12); color: #ff5252; border: 1px solid rgba(255,82,82,0.3); }
    .lb-factor-flat { background: rgba(107,124,119,0.12); color: #8fa39d; border: 1px solid #233631; }
    .lb-factor-text {
        display: flex;
        flex-direction: column;
        font-size: 13px;
        color: #e2e8f0;
        line-height: 1.35;
    }
    .lb-factor-text .lb-subtitle { font-size: 11px; letter-spacing: 0.4px; text-transform: uppercase; }
    .lb-factor-total {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid #1c2b27;
    }

    /* Buttons & misc */
    button[kind="primary"] {
        background-color: #b0ff22 !important;
        color: #090e0d !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
    }
    button[kind="primary"]:hover { box-shadow: 0 0 18px rgba(176,255,34,0.35); }
    div[data-testid="stProgress"] > div > div > div > div { background-color: #b0ff22 !important; }
    /* ---------------- Responsive / mobile ---------------- */
    @media (max-width: 900px) {
        .main .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
        /* Streamlit keeps columns side by side in wide layout; stack them. */
        div[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 100% !important;
        }
        .lb-kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
        .lb-kpi { padding: 12px; }
        .lb-kpi-val { font-size: 18px; }
        .lb-topbar { flex-direction: column; align-items: stretch; }
        .lb-topbar-left, .lb-topbar-right { flex-wrap: wrap; }
        .lb-topbar-right { justify-content: space-between; }
        .lb-val { font-size: 22px; }
        .lb-title { font-size: 19px !important; }
        .lb-card { padding: 14px; border-radius: 12px; }
        .lb-legend { flex-wrap: wrap; }
    }

    @media (max-width: 520px) {
        .lb-kpi-grid { grid-template-columns: 1fr 1fr; }
        .lb-ts { display: none; }
        .lb-factor { align-items: flex-start; }
        .lb-factor-text { font-size: 12px; }
        .lb-stat-row { font-size: 13px; }
        /* Keep the sidebar from swallowing the viewport on phones. */
        section[data-testid="stSidebar"] { min-width: 78vw !important; width: 78vw !important; }
    }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-thumb { background: #233631; border-radius: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GLOBAL MARKET MAP
# ============================================================

MARKET_MAP = {
    ".NS": ("India", "INR", "Asia/Kolkata", "NSE"),
    ".BO": ("India", "INR", "Asia/Kolkata", "BSE"),
    ".L": ("United Kingdom", "GBP", "Europe/London", "LSE"),
    ".DE": ("Germany", "EUR", "Europe/Berlin", "XETRA"),
    ".PA": ("France", "EUR", "Europe/Paris", "Euronext Paris"),
    ".AS": ("Netherlands", "EUR", "Europe/Amsterdam", "Euronext Amsterdam"),
    ".MI": ("Italy", "EUR", "Europe/Rome", "Borsa Italiana"),
    ".SW": ("Switzerland", "CHF", "Europe/Zurich", "SIX"),
    ".TO": ("Canada", "CAD", "America/Toronto", "TSX"),
    ".V": ("Canada", "CAD", "America/Vancouver", "TSXV"),
    ".HK": ("Hong Kong", "HKD", "Asia/Hong_Kong", "HKEX"),
    ".T": ("Japan", "JPY", "Asia/Tokyo", "Tokyo Stock Exchange"),
    ".KS": ("South Korea", "KRW", "Asia/Seoul", "Korea Exchange"),
    ".KQ": ("South Korea", "KRW", "Asia/Seoul", "KOSDAQ"),
    ".SS": ("China", "CNY", "Asia/Shanghai", "Shanghai"),
    ".SZ": ("China", "CNY", "Asia/Shanghai", "Shenzhen"),
    ".TW": ("Taiwan", "TWD", "Asia/Taipei", "TWSE"),
    ".AX": ("Australia", "AUD", "Australia/Sydney", "ASX"),
    ".NZ": ("New Zealand", "NZD", "Pacific/Auckland", "NZX"),
    ".SG": ("Singapore", "SGD", "Asia/Singapore", "SGX"),
    ".SA": ("Brazil", "BRL", "America/Sao_Paulo", "B3"),
    ".MX": ("Mexico", "MXN", "America/Mexico_City", "BMV"),
    ".JO": ("South Africa", "ZAR", "Africa/Johannesburg", "JSE"),
    ".IL": ("Israel", "ILS", "Asia/Jerusalem", "TASE"),
}

DEFAULT_MARKET = ("United States", "USD", "America/New_York", "NYSE/NASDAQ")


# ============================================================
# MARKET DETECTION
# ============================================================

def detect_market(ticker: str):
    ticker = ticker.upper().strip()

    for suffix, info in MARKET_MAP.items():
        if ticker.endswith(suffix):
            country, currency, tz, exchange = info
            return {
                "country": country,
                "currency": currency,
                "timezone": tz,
                "exchange": exchange,
            }

    return {
        "country": DEFAULT_MARKET[0],
        "currency": DEFAULT_MARKET[1],
        "timezone": DEFAULT_MARKET[2],
        "exchange": DEFAULT_MARKET[3],
    }


def get_currency_symbol(currency):
    symbols = {
        "USD": "$",
        "INR": "₹",
        "GBP": "£",
        "EUR": "€",
        "JPY": "¥",
        "CNY": "¥",
        "HKD": "HK$",
        "CAD": "C$",
        "AUD": "A$",
        "SGD": "S$",
        "CHF": "CHF ",
        "KRW": "₩",
        "TWD": "NT$",
        "BRL": "R$",
        "MXN": "MX$",
        "ZAR": "R",
        "ILS": "₪",
        "NZD": "NZ$",
    }
    return symbols.get(currency, currency + " ")


# ============================================================
# MARKET SESSION
# ============================================================

def session_status(market):
    now = datetime.now(ZoneInfo(market["timezone"]))

    if now.weekday() >= 5:
        return "CLOSED", now

    sessions = {
        "NYSE/NASDAQ": (9, 30, 16, 0),
        "NSE": (9, 15, 15, 30),
        "BSE": (9, 15, 15, 30),
        "LSE": (8, 0, 16, 30),
        "XETRA": (9, 0, 17, 30),
        "Tokyo Stock Exchange": (9, 0, 15, 0),
        "HKEX": (9, 30, 16, 0),
        "ASX": (10, 0, 16, 0),
        "TSX": (9, 30, 16, 0),
        "SIX": (9, 0, 17, 30),
        "SGX": (9, 0, 17, 0),
    }

    start_h, start_m, end_h, end_m = sessions.get(
        market["exchange"],
        (9, 30, 16, 0),
    )

    start = now.replace(
        hour=start_h,
        minute=start_m,
        second=0,
        microsecond=0,
    )

    end = now.replace(
        hour=end_h,
        minute=end_m,
        second=0,
        microsecond=0,
    )

    if start <= now <= end:
        return "OPEN", now

    return "CLOSED", now


# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=60)
def download_data(ticker, period="2y", interval="1d"):
    intervals_order = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d"]

    try:
        attempts = [interval] + [i for i in intervals_order if i != interval]

        for intv in attempts:
            data = yf.download(
                ticker,
                period=period,
                interval=intv,
                auto_adjust=True,
                progress=False,
                threads=True,
            )

            if data is None or data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = data.rename(columns=str.title)

            required = ["Open", "High", "Low", "Close", "Volume"]

            for col in required:
                if col not in data.columns:
                    data[col] = np.nan

            data = data[required].dropna(subset=["Close"])

            if not data.empty:
                data.attrs["used_interval"] = intv
                return data

        return pd.DataFrame()

    except Exception:
        return pd.DataFrame()


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

TRADING_DAYS = 252


def periods_per_year(index):
    """Annualisation factor inferred from the bar spacing of the series.

    Daily or coarser bars annualise with 252 trading days; intraday bars scale
    that by the number of bars per session so Sharpe, volatility and CAGR stay
    comparable across intervals.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return float(TRADING_DAYS)

    step = pd.Series(index).diff().dt.total_seconds().dropna()
    step = step[step > 0]
    if step.empty:
        return float(TRADING_DAYS)

    median_step = float(step.median())
    if median_step >= 86_400:
        return float(TRADING_DAYS)

    bars_per_session = max(6.5 * 3600 / median_step, 1.0)
    return float(TRADING_DAYS * bars_per_session)


def add_indicators(df):
    d = df.copy()
    ppy = periods_per_year(d.index)

    close = d["Close"]
    high = d["High"]
    low = d["Low"]
    volume = d["Volume"]

    d["SMA20"] = close.rolling(20).mean()
    d["SMA50"] = close.rolling(50).mean()
    d["SMA100"] = close.rolling(100).mean()
    d["SMA200"] = close.rolling(200).mean()

    d["EMA12"] = close.ewm(span=12, adjust=False).mean()
    d["EMA26"] = close.ewm(span=26, adjust=False).mean()

    d["MACD"] = d["EMA12"] - d["EMA26"]
    d["MACDSignal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACDHist"] = d["MACD"] - d["MACDSignal"]

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # An empty average loss means an uninterrupted advance, i.e. RSI 100.
    d["RSI"] = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    d["TR"] = tr
    d["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["ATR_pct"] = d["ATR"] / close * 100

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()

    d["BBMid"] = bb_mid
    d["BBUpper"] = bb_mid + 2 * bb_std
    d["BBLower"] = bb_mid - 2 * bb_std

    band_width = d["BBUpper"] - d["BBLower"]

    d["BBPosition"] = (
        (close - d["BBLower"]) / band_width.replace(0, np.nan) * 100
    )

    d["ROC20"] = close.pct_change(20, fill_method=None) * 100

    lowest = low.rolling(14).min()
    highest = high.rolling(14).max()

    d["StochK"] = (
        100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    )

    d["StochD"] = d["StochK"].rolling(3).mean()

    typical = (high + low + close) / 3
    tp_sma = typical.rolling(20).mean()
    mad = typical.rolling(20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )

    d["CCI"] = (typical - tp_sma) / (0.015 * mad.replace(0, np.nan))

    direction = np.sign(close.diff()).fillna(0)
    d["OBV"] = (direction * volume).cumsum()

    d["AvgVolume20"] = volume.rolling(20).mean()
    d["RelativeVolume"] = volume / d["AvgVolume20"].replace(0, np.nan)

    returns = close.pct_change(fill_method=None)
    d["DailyReturn"] = returns
    d["Volatility20"] = returns.rolling(20).std() * np.sqrt(ppy) * 100
    d["Volatility60"] = returns.rolling(60).std() * np.sqrt(ppy) * 100

    rolling_mean = close.rolling(20).mean()
    rolling_std = close.rolling(20).std()
    d["ZScore20"] = (close - rolling_mean) / rolling_std.replace(0, np.nan)

    d["PricePercentile"] = close.rolling(252).rank(pct=True) * 100

    d["SMA20Slope"] = d["SMA20"].pct_change(10, fill_method=None) * 100
    d["SMA50Slope"] = d["SMA50"].pct_change(20, fill_method=None) * 100

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    smoothed_plus = plus_dm.ewm(alpha=1 / 14, adjust=False).mean()
    smoothed_minus = minus_dm.ewm(alpha=1 / 14, adjust=False).mean()

    atr14 = d["ATR"].replace(0, np.nan)

    plus_di = 100 * smoothed_plus / atr14
    minus_di = 100 * smoothed_minus / atr14

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    d["ADX"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    return d


# ============================================================
# MATHEMATICAL METRICS
# ============================================================

def max_drawdown(equity):
    # The starting value of 1.0 is the first peak, otherwise an initial loss is
    # measured against an already-depressed high-water mark.
    equity = pd.concat([pd.Series([1.0]), pd.Series(equity.values)], ignore_index=True)
    dd = equity / equity.cummax() - 1
    return dd.min() * 100


def sharpe_ratio(returns, rf=0.0, ppy=TRADING_DAYS):
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan

    excess = returns - rf / ppy
    std = excess.std(ddof=1)

    if std == 0 or np.isnan(std):
        return np.nan

    return excess.mean() / std * np.sqrt(ppy)


def sortino_ratio(returns, rf=0.0, ppy=TRADING_DAYS):
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan

    excess = returns - rf / ppy

    # Target semideviation: shortfalls squared and averaged over every period,
    # not only the losing ones.
    downside_std = np.sqrt((excess.clip(upper=0) ** 2).mean())
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan

    return excess.mean() / downside_std * np.sqrt(ppy)


def historical_stats(df, rf=0.0):
    returns = df["DailyReturn"].dropna()
    ppy = periods_per_year(df.index)

    if len(returns) == 0:
        return {
            "Total Return": 0.0,
            "CAGR": 0.0,
            "Win Rate": 0.0,
            "Avg Win": 0.0,
            "Avg Loss": 0.0,
            "Payoff": np.nan,
            "Expectancy": 0.0,
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "Max Drawdown": 0.0,
        }

    equity = (1 + returns).cumprod()
    final_equity = float(equity.iloc[-1])

    total_return = (final_equity - 1) * 100.0

    years = len(returns) / ppy

    # Annualising a sub-year sample explodes the growth rate, so only compound
    # once there is at least a full year of history.
    if final_equity <= 0 or years <= 0:
        cagr = np.nan
    elif years < 1:
        cagr = total_return
    else:
        cagr = (final_equity ** (1.0 / years) - 1.0) * 100.0

    win_rate = (returns > 0).mean() * 100.0

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = abs(losses.mean()) if len(losses) else 0.0

    payoff = avg_win / avg_loss if avg_loss > 0 else np.nan

    expectancy = ((win_rate / 100.0) * avg_win - (1 - win_rate / 100.0) * avg_loss) * 100.0

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Win Rate": win_rate,
        "Avg Win": avg_win * 100.0,
        "Avg Loss": avg_loss * 100.0,
        "Payoff": payoff,
        "Expectancy": expectancy,
        "Sharpe": sharpe_ratio(returns, rf, ppy),
        "Sortino": sortino_ratio(returns, rf, ppy),
        "Max Drawdown": max_drawdown(equity),
    }


# ============================================================
# SIGNAL ENGINE
# ============================================================

def explain(value, name):
    if name == "RSI":
        if value >= 70:
            return "Overbought momentum; upside may be stretched."
        if value <= 30:
            return "Oversold momentum; rebound risk is elevated."
        if value >= 55:
            return "Momentum favors buyers."
        if value <= 45:
            return "Momentum favors sellers."
        return "Momentum is neutral."

    if name == "MACD":
        return (
            "Bullish momentum because MACD is above its signal."
            if value > 0
            else
            "Bearish momentum because MACD is below its signal."
        )

    if name == "ADX":
        if value >= 25:
            return "Trend strength is meaningful."
        return "Trend strength is weak; sideways behavior is more likely."

    if name == "RelativeVolume":
        if value >= 1.5:
            return "Trading volume is unusually strong."
        if value >= 1:
            return "Volume confirms normal participation."
        return "Volume is below its recent average."

    if name == "ZScore":
        if value >= 2:
            return "Price is unusually high versus its recent mean."
        if value <= -2:
            return "Price is unusually low versus its recent mean."
        return "Price is within a normal recent statistical range."

    return ""


def build_signal(d):
    x = d.iloc[-1]

    score = 50
    reasons = []

    # A moving average that has not warmed up yet must not be scored, otherwise
    # a short history is penalised as if the trend were bearish.
    for fast, slow, weight, label in [
        ("Close", "SMA20", 5, "Price"),
        ("SMA20", "SMA50", 6, "SMA20"),
        ("SMA50", "SMA200", 7, "SMA50"),
    ]:
        if pd.isna(x[fast]) or pd.isna(x[slow]):
            reasons.append(("Trend", "0", f"{slow} needs more history to be scored."))
            continue

        above = x[fast] > x[slow]
        score += weight if above else -weight
        reasons.append(
            (
                "Trend",
                f"{'+' if above else '-'}{weight}",
                f"{label} is {'above' if above else 'below'} {slow}.",
            )
        )

    rsi = x["RSI"]

    if pd.notna(rsi):
        if 50 <= rsi < 70:
            score += 8
            reasons.append(("Momentum", "+8", explain(rsi, "RSI")))
        elif rsi > 70:
            score -= 2
            reasons.append(("Momentum", "-2", explain(rsi, "RSI")))
        elif rsi < 30:
            score += 2
            reasons.append(("Momentum", "+2", explain(rsi, "RSI")))
        else:
            score -= 3
            reasons.append(("Momentum", "-3", explain(rsi, "RSI")))

    if pd.notna(x["MACD"]) and pd.notna(x["MACDSignal"]):
        if x["MACD"] > x["MACDSignal"]:
            score += 8
            reasons.append(("Momentum", "+8", "MACD is above its signal line."))
        else:
            score -= 8
            reasons.append(("Momentum", "-8", "MACD is below its signal line."))

    if pd.notna(x["ADX"]):
        if x["ADX"] >= 25:
            score += 5 if x["SMA20"] > x["SMA50"] else -5
            reasons.append(
                (
                    "Trend Strength",
                    "+5" if x["SMA20"] > x["SMA50"] else "-5",
                    explain(x["ADX"], "ADX"),
                )
            )

    if pd.notna(x["RelativeVolume"]):
        if x["RelativeVolume"] >= 1.2:
            score += 5
            reasons.append(
                ("Volume", "+5", explain(x["RelativeVolume"], "RelativeVolume"))
            )
        else:
            score -= 2
            reasons.append(
                ("Volume", "-2", explain(x["RelativeVolume"], "RelativeVolume"))
            )

    if pd.notna(x["ROC20"]):
        if x["ROC20"] > 0:
            score += 5
            reasons.append(("ROC", "+5", "20-bar rate of change is positive."))
        else:
            score -= 5
            reasons.append(("ROC", "-5", "20-bar rate of change is negative."))

    bb = x["BBPosition"]

    if pd.notna(bb):
        if 50 <= bb <= 90:
            score += 3
            reasons.append(
                ("Volatility", "+3", "Price is in the upper half of its Bollinger range.")
            )
        elif bb < 20:
            score += 1
            reasons.append(
                ("Volatility", "+1", "Price is near the lower Bollinger region.")
            )

    score = float(np.clip(score, 0, 100))

    if score >= 65:
        regime = "BULLISH"
        action = "BUY BIAS"
    elif score <= 35:
        regime = "BEARISH"
        action = "SELL / AVOID BIAS"
    else:
        regime = "SIDEWAYS / MIXED"
        action = "WAIT / NEUTRAL"

    return score, regime, action, reasons


# ============================================================
# POSITION SIZING
# ============================================================

MAX_RISK_PER_TRADE = 0.02


def position_math(capital, price, stop, target, win_probability, payoff_ratio=None):
    """Fractional-Kelly sizing for the planned trade.

    The Kelly payoff is the reward/risk ratio of *this* trade (target and stop
    distance), not a historical payoff, so the fraction and the trade geometry
    describe the same bet.
    """
    risk_per_share = abs(price - stop)
    reward_per_share = max(target - price, 0.0)

    empty = {
        "shares": 0,
        "risk": 0.0,
        "profit": 0.0,
        "kelly": 0.0,
        "quarter_kelly": 0.0,
        "rr": np.nan,
        "expectancy": 0.0,
        "capital_used": 0.0,
    }

    if risk_per_share <= 0 or price <= 0 or capital <= 0:
        return empty

    rr = reward_per_share / risk_per_share

    p = float(np.clip(win_probability, 0.01, 0.99))
    q = 1 - p
    b = max(rr if payoff_ratio is None else payoff_ratio, 0.01)

    kelly = (b * p - q) / b
    quarter_kelly = max(0.0, kelly / 4)

    risk_fraction = min(quarter_kelly, MAX_RISK_PER_TRADE)
    shares = math.floor(capital * risk_fraction / risk_per_share)

    # Never size beyond the cash that is actually available.
    shares = min(shares, math.floor(capital / price))

    return {
        "shares": shares,
        "risk": shares * risk_per_share,
        "profit": shares * reward_per_share,
        "kelly": kelly,
        "quarter_kelly": quarter_kelly,
        "rr": rr,
        "expectancy": shares * (p * reward_per_share - q * risk_per_share),
        "capital_used": shares * price,
    }


# ============================================================
# MONTE CARLO & BACKTEST
# ============================================================

def detect_patterns(df):
    return {"patterns_found": []}


def backtest_strategy(df, entry_signal, exit_signal=None, capital=10000.0):
    returns = df["DailyReturn"].fillna(0)
    position = 0
    cash = capital
    shares = 0
    equity = []
    trades = []

    for ts, r in returns.items():
        price = df.loc[ts, "Close"]
        if entry_signal is not None and entry_signal.get(ts, False) and position == 0:
            shares = math.floor(cash / price)
            cost = shares * price
            cash -= cost
            position = 1
            trades.append({"time": ts, "type": "enter", "price": price, "shares": shares})
        if exit_signal is not None and exit_signal.get(ts, False) and position == 1:
            proceeds = shares * price
            cash += proceeds
            trades.append({"time": ts, "type": "exit", "price": price, "shares": shares})
            shares = 0
            position = 0
        mm = cash + shares * price
        equity.append((ts, mm))

    eq_index = [t for t, v in equity]
    eq_values = [v for t, v in equity]
    equity_series = pd.Series(eq_values, index=eq_index)
    return equity_series, trades


def monte_carlo(returns, simulations=3000, days=252, mode="bootstrap"):
    returns = pd.Series(returns).dropna().values

    if len(returns) < 30:
        return None

    rng = np.random.default_rng(42)

    days = max(int(days), 1)

    if mode == "bootstrap":
        sampled = rng.choice(returns, size=(simulations, days), replace=True)
        paths = np.cumprod(1 + sampled, axis=1)
    elif mode == "gbm":
        # Drift and volatility are estimated in log space on the same bar size
        # as the input returns, so one simulated step is one bar.
        log_returns = np.log1p(returns[returns > -1])
        mu = float(np.mean(log_returns))
        sigma = float(np.std(log_returns, ddof=1))
        z = rng.normal(size=(simulations, days))
        paths = np.exp(np.cumsum(mu + sigma * z, axis=1))
    else:
        raise ValueError("Unsupported monte_carlo mode")

    # Every path starts at 1.0, which is also the first running peak.
    paths = np.hstack([np.ones((simulations, 1)), paths])

    final_values = paths[:, -1]
    drawdowns = (paths / np.maximum.accumulate(paths, axis=1) - 1).min(axis=1)

    return {
        "median_final": np.median(final_values),
        "p05_final": np.percentile(final_values, 5),
        "p95_final": np.percentile(final_values, 95),
        "median_drawdown": np.median(drawdowns),
        "p05_drawdown": np.percentile(drawdowns, 5),
        "p95_drawdown": np.percentile(drawdowns, 95),
    }


# ============================================================
# NEWS
# ============================================================

def get_news(ticker):
    try:
        obj = yf.Ticker(ticker)
        news = getattr(obj, "news", []) or []

        rows = []
        for item in news[:15]:
            title = item.get("title") or item.get("headline") or item.get("summary") or ""

            publisher = ""
            if "publisher" in item:
                publisher = item.get("publisher") or ""
            else:
                provider = item.get("provider") or item.get("source") or {}
                if isinstance(provider, dict):
                    publisher = provider.get("name") or provider.get("displayName") or ""

            link = item.get("link") or item.get("url") or item.get("canonicalUrl") or ""
            if isinstance(link, dict):
                link = link.get("url") or link.get("href") or ""

            if title:
                rows.append({"Title": title, "Publisher": publisher, "Link": link})

        return rows
    except Exception:
        return []


# ============================================================
# CHART (STYLING MATCHING GREEN NEON THEME)
# ============================================================

ACCENT = "#b0ff22"
DANGER = "#ff5252"
PANEL_BG = "#121a18"
GRID = "#1a2724"
MUTED = "#6b7c77"


def price_chart(df, ticker, stop=None, target=None, chart_style="Candles"):
    """Price panel with Bollinger envelope, moving averages and a volume subplot."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.76, 0.24],
        vertical_spacing=0.04,
    )

    if {"BBUpper", "BBLower"} <= set(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["BBUpper"], name="BB Upper", mode="lines",
                line=dict(color="rgba(176,255,34,0.25)", width=1), hoverinfo="skip",
                showlegend=False,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["BBLower"], name="Bollinger", mode="lines",
                line=dict(color="rgba(176,255,34,0.25)", width=1),
                fill="tonexty", fillcolor="rgba(176,255,34,0.05)", hoverinfo="skip",
            ),
            row=1, col=1,
        )

    if chart_style == "Area":
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["Close"], name="Price", mode="lines",
                line=dict(color=ACCENT, width=2),
                fill="tozeroy", fillcolor="rgba(176,255,34,0.08)",
            ),
            row=1, col=1,
        )
    else:
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                name="Price",
                increasing_line_color=ACCENT, increasing_fillcolor=ACCENT,
                decreasing_line_color=DANGER, decreasing_fillcolor=DANGER,
                line=dict(width=1),
            ),
            row=1, col=1,
        )

    for col, color in [("SMA20", "#38bdf8"), ("SMA50", "#f59e0b"), ("SMA200", "#a855f7")]:
        if col in df:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df[col], name=col,
                    line=dict(color=color, width=1.4), opacity=0.9,
                ),
                row=1, col=1,
            )

    for level, color, label in [(stop, DANGER, "Stop"), (target, ACCENT, "Target")]:
        if level is not None and np.isfinite(level):
            fig.add_hline(
                y=level, line=dict(color=color, width=1, dash="dot"),
                annotation_text=f"{label} {level:,.2f}",
                annotation_position="right",
                annotation_font=dict(color=color, size=11),
                row=1, col=1,
            )

    if "Volume" in df:
        vol_colors = np.where(df["Close"] >= df["Open"], "rgba(176,255,34,0.45)", "rgba(255,82,82,0.45)")
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=vol_colors, showlegend=False),
            row=2, col=1,
        )

    lo = float(np.nanmin(df["Low"] if chart_style != "Area" else df["Close"]))
    hi = float(np.nanmax(df["High"] if chart_style != "Area" else df["Close"]))
    for level in (stop, target):
        if level is not None and np.isfinite(level):
            lo, hi = min(lo, level), max(hi, level)
    pad = max((hi - lo) * 0.08, hi * 0.005)

    fig.update_layout(
        height=520,
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=PANEL_BG,
        bargap=0.1,
        margin=dict(l=10, r=10, t=46, b=10),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#16221f", bordercolor="#233631", font=dict(family="Inter", color="#e2e8f0")),
        font=dict(family="Inter", color="#8fa39d"),
        dragmode="pan",
        showlegend=False,
    )

    fig.update_xaxes(
        showgrid=True, gridcolor=GRID, showline=False, tickfont=dict(color=MUTED),
        showspikes=True, spikecolor="#2c403b", spikethickness=1, spikedash="dot", spikemode="across",
        rangeslider_visible=False,
    )
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(step="all", label="ALL"),
            ],
            bgcolor="#16221f",
            activecolor="#233631",
            bordercolor="#233631",
            borderwidth=1,
            font=dict(color="#8fa39d", size=11),
            x=0, y=1.16, yanchor="bottom",
        ),
        row=1, col=1,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=GRID, showline=False, tickfont=dict(color=MUTED),
        zeroline=False, side="right", range=[lo - pad, hi + pad], row=1, col=1,
    )
    fig.update_yaxes(
        showgrid=False, tickfont=dict(color=MUTED, size=10), zeroline=False,
        side="right", title_text="Vol", title_font=dict(size=10, color=MUTED), row=2, col=1,
    )

    return fig


def score_gauge(score):
    """Compact conviction gauge for the model score."""
    color = ACCENT if score >= 65 else (DANGER if score <= 35 else "#f59e0b")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(suffix="<span style='font-size:14px;color:#6b7c77'> /100</span>",
                        font=dict(size=34, color="#ffffff", family="Inter")),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=MUTED, tickfont=dict(size=10, color=MUTED), tickwidth=1),
                bar=dict(color=color, thickness=0.28),
                bgcolor="#16221f",
                borderwidth=0,
                steps=[
                    dict(range=[0, 35], color="rgba(255,82,82,0.12)"),
                    dict(range=[35, 65], color="rgba(245,158,11,0.12)"),
                    dict(range=[65, 100], color="rgba(176,255,34,0.12)"),
                ],
                threshold=dict(line=dict(color="#ffffff", width=2), thickness=0.75, value=score),
            ),
        )
    )
    fig.update_layout(
        height=190,
        margin=dict(l=10, r=10, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=MUTED),
    )
    return fig


# ============================================================
# SIDEBAR NAVIGATION & PARAMETERS
# ============================================================

def get_query_params_safe():
    try:
        return {k: v if isinstance(v, list) else [v] for k, v in st.query_params.to_dict().items()}
    except Exception:
        try:
            return st.experimental_get_query_params()
        except Exception:
            return {}


def safe_query_index(values, default_index, query_key=None):
    if query_key is None:
        return default_index

    current = _q.get(query_key, [])
    if not current:
        return default_index

    try:
        return values.index(current[0])
    except (ValueError, TypeError, IndexError):
        return default_index


_q = get_query_params_safe()

# Sidebar Brand Header
st.sidebar.markdown(
    """
    <div style='display:flex; align-items:center; gap:10px; margin-bottom: 20px;'>
    <div style='background:#b0ff22; color:#090e0d; font-size:18px; font-weight:800; padding:6px 12px; border-radius:10px;'>QA</div>
    <div style='font-size:20px; font-weight:700; color:#ffffff; letter-spacing:0.5px;'>QUANT AI</div>
    </div>
    """,
    unsafe_allow_html=True,
)

period_options = ["6mo", "1y", "2y", "5y", "10y"]
interval_options = ["1m", "2m", "5m", "15m", "30m", "60m", "1d"]
refresh_options = ["5s", "15s", "30s", "1m", "5m", "1h"]

st.sidebar.markdown("<p class='lb-nav-label'>MARKET DATA</p>", unsafe_allow_html=True)

ticker = st.sidebar.text_input(
    "Asset Ticker Symbol",
    value=_q.get("ticker", ["AAPL"])[0],
    help="Examples: AAPL, MSFT, RELIANCE.NS, 7203.T",
).upper().strip()

_range_col, _interval_col = st.sidebar.columns(2)
period = _range_col.selectbox(
    "Range",
    period_options,
    index=safe_query_index(period_options, 2, "period"),
)
interval = _interval_col.selectbox(
    "Interval",
    interval_options,
    index=safe_query_index(interval_options, 6, "interval"),
)

st.sidebar.markdown("<p class='lb-nav-label'>CHART</p>", unsafe_allow_html=True)
chart_style = st.sidebar.radio(
    "Chart Style",
    ["Candles", "Area"],
    horizontal=True,
    label_visibility="collapsed",
)
lookback = st.sidebar.slider("Bars Shown", 50, 500, 200, 25, help="Number of most recent candles plotted.")
show_levels = st.sidebar.toggle("Overlay stop / target levels", value=True)

st.sidebar.markdown("<p class='lb-nav-label'>RISK & SIZING</p>", unsafe_allow_html=True)

capital = st.sidebar.number_input(
    "Portfolio Capital ($)",
    min_value=0.0,
    value=float(_q.get("capital", [10000.0])[0]),
    step=100.0,
)

risk_stop_atr = st.sidebar.slider(
    "Stop Distance (ATR Multiplier)",
    0.5,
    5.0,
    float(_q.get("risk_stop_atr", [2.0])[0]),
    0.1,
)

target_atr = st.sidebar.slider(
    "Target Distance (ATR Multiplier)",
    0.5,
    10.0,
    float(_q.get("target_atr", [4.0])[0]),
    0.1,
)

with st.sidebar.expander("⏱️ Live Refresh", expanded=False):
    autorefresh = st.checkbox("Auto-refresh stream", value=("autorefresh" in _q))
    refresh_mode = st.selectbox(
        "Refresh Mode",
        ["soft", "hard"],
        index=safe_query_index(["soft", "hard"], 0, "refresh_mode"),
    )
    refresh_interval = st.selectbox(
        "Poll Interval",
        refresh_options,
        index=safe_query_index(refresh_options, 3, "refresh_interval"),
    )

st.sidebar.markdown("---")
run = st.sidebar.button("⚡ Run Analysis", type="primary", use_container_width=True)
st.sidebar.caption("Educational analytics only — not investment advice.")


# ============================================================
# MAIN APPLICATION WORKSPACE
# ============================================================

if not ticker:
    st.info("Enter a ticker in the sidebar to begin.")
    st.stop()

market = detect_market(ticker)
symbol = get_currency_symbol(market["currency"])
status, local_time = session_status(market)

with st.spinner(f"Loading {ticker} market data…"):
    data = download_data(ticker, period=period, interval=interval)

fetch_ts = datetime.now(timezone.utc)
fetch_ts_ms = fetch_ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + ' UTC'

if data.empty:
    st.error("No market data returned. Verify the ticker symbol or try adjusting the interval.")
    st.stop()

if autorefresh:
    interval_map = {"5s": 5_000, "15s": 15_000, "30s": 30_000, "1m": 60_000, "5m": 5 * 60_000, "1h": 60 * 60_000}
    ms = interval_map.get(refresh_interval, 30_000)
    if refresh_mode == "hard":
        st.markdown(f"<script>setTimeout(()=>window.location.reload(), {ms});</script>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"""
            <script>
            setTimeout(()=>{{
                const qp = new URLSearchParams(window.location.search);
                qp.set('_poll', Date.now());
                window.history.replaceState(null, '', window.location.pathname + '?' + qp.toString());
            }}, {ms});
            </script>
            """,
            unsafe_allow_html=True,
        )

data = add_indicators(data)
last = data.iloc[-1]
price = float(last["Close"])
score, regime, action, reasons = build_signal(data)

atr = float(last["ATR"]) if pd.notna(last["ATR"]) else price * 0.02
stop = max(price - risk_stop_atr * atr, 0.0)
target = price + target_atr * atr
stats = historical_stats(data)
win_probability = float(np.clip(stats["Win Rate"] / 100, 0.05, 0.95))
pm = position_math(capital, price, stop, target, win_probability)


# ------------------------------------------------------------
# TOP HEADER BAR UI (MATCHING REFERENCE DESIGN)
# ------------------------------------------------------------

regime_class = "lb-green-text" if regime == "BULLISH" else ("lb-red-text" if regime == "BEARISH" else "lb-subtitle")
action_class = "lb-pill-green" if score >= 65 else ("lb-pill-red" if score <= 35 else "lb-pill-neutral")

render_html(
    f"""
    <div class='lb-topbar'>
        <div class='lb-topbar-left'>
            <div class='lb-status-chip'>
                <span class='lb-dot' style='background:{ACCENT if status == "OPEN" else DANGER};'></span>
                <span>{market["exchange"]} • {status}</span>
            </div>
            <span class='lb-ts'>{fetch_ts_ms}</span>
        </div>
        <div class='lb-topbar-right'>
            <div class='lb-chip'>
                <span class='lb-subtitle'>MARKET</span>
                <span class='lb-chip-val'>{market["country"]} ({market["currency"]})</span>
            </div>
            <div class='lb-chip'>
                <span class='lb-subtitle'>REGIME</span>
                <span class='lb-chip-val {regime_class}'>{regime}</span>
            </div>
            <div class='lb-pill {action_class}'>{action}</div>
        </div>
    </div>
    """
)

# KPI strip
prev_close = float(data["Close"].iloc[-2]) if len(data) > 1 else price
day_change = (price / prev_close - 1) * 100 if prev_close else 0.0
rsi_val = float(last["RSI"]) if pd.notna(last.get("RSI")) else float("nan")
rel_vol = float(last["RelativeVolume"]) if pd.notna(last.get("RelativeVolume")) else float("nan")

kpis = [
    ("LAST PRICE", f"{symbol}{price:,.2f}", f"{day_change:+.2f}% today", day_change >= 0),
    ("MODEL SCORE", f"{score:.0f}/100", regime.title(), score >= 50),
    ("ATR (VOLATILITY)", f"{symbol}{atr:,.2f}", f"{atr / price * 100:.2f}% of price", None),
    ("RSI (14)", "n/a" if np.isnan(rsi_val) else f"{rsi_val:.1f}",
     "n/a" if np.isnan(rsi_val) else ("Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"),
     None if np.isnan(rsi_val) else 30 <= rsi_val <= 70),
    ("REL. VOLUME", "n/a" if np.isnan(rel_vol) else f"{rel_vol:.2f}x",
     "vs 20-bar average", None if np.isnan(rel_vol) else rel_vol >= 1),
    ("20-BAR RETURN", f"{last.get('ROC20', 0.0):+.2f}%", "momentum", last.get("ROC20", 0.0) >= 0),
]


def _delta_class(good):
    if good is None:
        return "lb-subtitle"
    return "lb-green-text" if good else "lb-red-text"


render_html(
    "<div class='lb-kpi-grid'>"
    + "".join(
        f"<div class='lb-kpi'><div class='lb-subtitle'>{label}</div>"
        f"<div class='lb-kpi-val'>{value}</div>"
        f"<div class='lb-kpi-delta {_delta_class(good)}'>{delta}</div></div>"
        for label, value, delta, good in kpis
    )
    + "</div>"
)


# ------------------------------------------------------------
# MAIN GRID: BALANCE & CHART (LEFT) vs POSITION CALCULATOR (RIGHT)
# ------------------------------------------------------------

left_col, right_col = st.columns([2.2, 1])

with left_col:
    ret_color = "lb-green-text" if day_change >= 0 else "lb-red-text"
    arrow = "↑" if day_change >= 0 else "↓"

    with st.container(border=True):
        render_html(
            f"""
            <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                <div>
                    <div class='lb-subtitle'>{ticker} · {market["exchange"]} · {interval} candles</div>
                    <div class='lb-val'>{symbol}{price:,.2f} <span class='{ret_color}' style='font-size:16px;'>{arrow} {abs(day_change):.2f}%</span></div>
                </div>
                <div class='lb-legend'>
                    <span><i style='background:#38bdf8'></i>SMA20</span>
                    <span><i style='background:#f59e0b'></i>SMA50</span>
                    <span><i style='background:#a855f7'></i>SMA200</span>
                </div>
            </div>
            """
        )
        st.plotly_chart(
            price_chart(
                data.tail(lookback),
                ticker,
                stop=stop if show_levels else None,
                target=target if show_levels else None,
                chart_style=chart_style,
            ),
            use_container_width=True,
            config={"displayModeBar": False, "scrollZoom": False},
        )

with right_col:
    with st.container(border=True):
        render_html(
            "<div class='lb-title' style='font-size:18px;'>Conviction score</div>"
            "<div class='lb-subtitle'>Composite of trend, momentum & volume factors</div>"
        )
        st.plotly_chart(score_gauge(score), use_container_width=True, config={"displayModeBar": False})

    # Position Size Calculator Glass Card (Reference Matched)
    render_html(
        f"""
        <div class='lb-card'>
            <div class='lb-title' style='font-size:18px;'>Position size calculator</div>
            <div class='lb-subtitle' style='margin-bottom:15px;'>Risk Math & Fractional Kelly Model</div>

            <div style='display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-bottom:12px;'>
                <div>
                    <div class='lb-subtitle'>Capital</div>
                    <div class='lb-val-sm'>{symbol}{capital:,.0f}</div>
                </div>
                <div>
                    <div class='lb-subtitle'>Risk/Reward</div>
                    <div class='lb-val-sm'>{f"{pm['rr']:.2f}x" if np.isfinite(pm['rr']) else "n/a"}</div>
                </div>
            </div>

            <div style='display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-bottom:12px;'>
                <div>
                    <div class='lb-subtitle'>Calculated Stop</div>
                    <div class='lb-val-sm' style='color:#ff5252;'>{symbol}{stop:,.2f}</div>
                </div>
                <div>
                    <div class='lb-subtitle'>Calculated Target</div>
                    <div class='lb-val-sm' style='color:#b0ff22;'>{symbol}{target:,.2f}</div>
                </div>
            </div>

            <div style='background:#16221f; border-radius:10px; padding:12px; margin-top:15px; border:1px solid #233631;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                    <span class='lb-subtitle'>Quarter-Kelly budget ({win_probability*100:.0f}% win prob.)</span>
                    <span style='font-weight:600; color:#ffffff;'>{pm['quarter_kelly']*100:.2f}%</span>
                </div>
                <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                    <span class='lb-subtitle'>Units (deal size)</span>
                    <span style='font-weight:700; color:#b0ff22;'>{pm['shares']:,} shares</span>
                </div>
                <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                    <span class='lb-subtitle'>Capital deployed</span>
                    <span style='font-weight:600; color:#ffffff;'>{symbol}{pm['capital_used']:,.2f}</span>
                </div>
                <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                    <span class='lb-subtitle'>Amount at risk</span>
                    <span style='font-weight:600; color:#ff5252;'>{symbol}{pm['risk']:,.2f}</span>
                </div>
                <div style='display:flex; justify-content:space-between;'>
                    <span class='lb-subtitle'>Profit at target</span>
                    <span style='font-weight:600; color:#b0ff22;'>{symbol}{pm['profit']:,.2f}</span>
                </div>
            </div>
        </div>
        """
    )


# ------------------------------------------------------------
# BOTTOM ROW: REASONS, INDICATORS & RISK SIMULATION (GRID)
# ------------------------------------------------------------

b_col1, b_col2, b_col3 = st.columns([1.6, 1, 1])

with b_col1:
    with st.container(border=True):
        render_html(
            "<div class='lb-title' style='font-size:16px;'>🔍 Model Factors</div>"
            "<div class='lb-subtitle' style='margin-bottom:12px;'>Score factor breakdown</div>"
        )

        factor_rows = []
        for factor, impact, interpretation in reasons:
            points = int(pd.to_numeric(impact, errors="coerce") or 0)
            tone = "lb-factor-up" if points > 0 else ("lb-factor-down" if points < 0 else "lb-factor-flat")
            factor_rows.append(
                f"<div class='lb-factor'>"
                f"<span class='lb-factor-badge {tone}'>{points:+d}</span>"
                f"<span class='lb-factor-text'>{interpretation}"
                f"<span class='lb-subtitle'>{factor}</span></span></div>"
            )

        render_html(
            f"<div class='lb-factor-list'>{''.join(factor_rows)}</div>"
            f"<div class='lb-factor-total'><span class='lb-subtitle'>Baseline 50 ± factors</span>"
            f"<span class='lb-chip-val'>{score:.0f}/100</span></div>"
        )

with b_col2:
    with st.container(border=True):
        render_html(
            "<div class='lb-title' style='font-size:16px;'>📊 Quant Statistics</div>"
            "<div class='lb-subtitle' style='margin-bottom:12px;'>Historical performance metrics</div>"
        )

        def stat_cell(value, fmt, good=None):
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                return "<span class='lb-subtitle'>n/a</span>"
            tone = "lb-subtitle" if good is None else ("lb-green-text" if good else "lb-red-text")
            return f"<span class='{tone}'>{fmt.format(value)}</span>"

        stat_rows = [
            ("Bar win rate", stat_cell(stats["Win Rate"], "{:.1f}%", stats["Win Rate"] >= 50)),
            ("Payoff ratio", stat_cell(stats["Payoff"], "{:.2f}", stats["Payoff"] >= 1)),
            ("Sharpe ratio", stat_cell(stats["Sharpe"], "{:.2f}", stats["Sharpe"] >= 1)),
            ("Sortino ratio", stat_cell(stats["Sortino"], "{:.2f}", stats["Sortino"] >= 1)),
            ("Max drawdown", stat_cell(stats["Max Drawdown"], "{:.2f}%", False)),
            ("CAGR", stat_cell(stats["CAGR"], "{:.2f}%", stats["CAGR"] >= 0)),
        ]
        render_html(
            "<div class='lb-stat-list'>"
            + "".join(
                f"<div class='lb-stat-row'><span class='lb-subtitle'>{name}</span>{cell}</div>"
                for name, cell in stat_rows
            )
            + "</div>"
        )

with b_col3:
    with st.container(border=True):
        # One simulated year expressed in bars of the selected interval.
        horizon = int(round(periods_per_year(data.index)))
        render_html(
            "<div class='lb-title' style='font-size:16px;'>🎲 Monte Carlo Risk</div>"
            f"<div class='lb-subtitle' style='margin-bottom:12px;'>3,000 resampled paths · {horizon:,} bars ≈ 1 year</div>"
        )

        mc = monte_carlo(data["DailyReturn"].dropna(), simulations=3000, days=horizon)
        if mc:
            growth_tone = "lb-green-text" if mc["median_final"] >= 1 else "lb-red-text"
            render_html(
                f"""
                <div class='lb-stat-list'>
                    <div class='lb-stat-row'><span class='lb-subtitle'>Median growth</span><span class='{growth_tone}'>{mc['median_final']:.2f}x</span></div>
                    <div class='lb-stat-row'><span class='lb-subtitle'>5th percentile</span><span style='color:#e2e8f0; font-weight:600;'>{mc['p05_final']:.2f}x</span></div>
                    <div class='lb-stat-row'><span class='lb-subtitle'>95th percentile</span><span style='color:#e2e8f0; font-weight:600;'>{mc['p95_final']:.2f}x</span></div>
                    <div class='lb-stat-row'><span class='lb-subtitle'>Median max drawdown</span><span class='lb-red-text'>{mc['median_drawdown']*100:.2f}%</span></div>
                </div>
                """
            )
            band_position = float(
                np.clip(
                    (mc["median_final"] - mc["p05_final"])
                    / max(mc["p95_final"] - mc["p05_final"], 1e-9),
                    0,
                    1,
                )
            )
            render_html(
                "<div class='lb-factor-total' style='margin-top:8px; padding-top:10px;'>"
                "<span class='lb-subtitle'>Median within 5th–95th band</span>"
                f"<span class='lb-chip-val'>{band_position*100:.0f}%</span></div>"
            )
            st.progress(band_position)
        else:
            st.info("Not enough history for a reliable simulation.")


# ------------------------------------------------------------
# EXPANDABLE DETAILED DATA & RAW TABLE
# ------------------------------------------------------------

with st.expander("📰 Recent Market News & Headlines"):
    news = get_news(ticker)
    if news:
        for item in news[:5]:
            link = item["Link"]
            render_html(
                f"<div class='lb-stat-row' style='flex-direction:column; align-items:flex-start; gap:4px;'>"
                f"<span style='color:#e2e8f0;'>{item['Title']}</span>"
                f"<span class='lb-subtitle'>{item['Publisher']}</span></div>"
            )
            if link:
                st.markdown(f"[Read full article]({link})")
    else:
        st.info("No recent headlines retrieved.")

with st.expander("🔬 Raw Indicator Data & Export Options"):
    display_cols = [
        "Open", "High", "Low", "Close", "Volume", "SMA20", "SMA50",
        "RSI", "MACD", "ATR", "BBPosition", "ADX", "ZScore20"
    ]
    available = [c for c in display_cols if c in data.columns]
    df_display = data[available].tail(100).round(4)

    st.dataframe(df_display, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            label="Download Tail 100 CSV",
            data=df_display.to_csv(index=True).encode("utf-8"),
            file_name=f"{ticker}_indicators_tail100.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            label="Download Full Dataset CSV",
            data=data.to_csv(index=True).encode("utf-8"),
            file_name=f"{ticker}_full_data.csv",
            mime="text/csv",
        )


# ------------------------------------------------------------
# FINAL CONCLUSION BANNER
# ------------------------------------------------------------

render_html(
    f"""
    <div class='lb-card' style='border-color:#233631; background:#0e1513; margin-top:10px;'>
        <div style='display:flex; align-items:center; gap:12px;'>
            <div style='font-size:20px;'>🧠</div>
            <div>
                <div style='font-weight:700; color:#ffffff;'>Model Decision Summary</div>
                <div style='color:#8fa39d; font-size:13px;'>{ticker}: <span class='{"lb-green-text" if regime=="BULLISH" else ("lb-red-text" if regime=="BEARISH" else "lb-subtitle")}'>{regime}</span> bias with score {score:.1f}/100 based on calculated indicator confluence.</div>
            </div>
        </div>
    </div>
    """
)
