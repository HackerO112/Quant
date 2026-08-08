# app.py
# Quant Research AI — Streamlit + yfinance
#
# Install:
#   pip install streamlit yfinance pandas numpy plotly scipy
#
# Run:
#   streamlit run app.py
#
# IMPORTANT:
# This application calculates indicators and statistical estimates.
# It does NOT guarantee future prices, profits, or 100% accuracy.
# for google colab must %%writefile app.py
import math
import warnings
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Quant Research AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {background-color:#0b1220;}
    .metric-card {
        padding:16px;
        border-radius:12px;
        background:#111827;
        border:1px solid #243044;
        margin-bottom:10px;
    }
    .bull {color:#22c55e;font-weight:700;}
    .bear {color:#ef4444;font-weight:700;}
    .neutral {color:#f59e0b;font-weight:700;}
    .small {color:#9ca3af;font-size:0.85rem;}
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
    """
    Approximate regular session status.
    Exact exchange holidays are not inferred here.
    """

    now = datetime.now(ZoneInfo(market["timezone"]))

    if now.weekday() >= 5:
        return "CLOSED", now

    # Common approximate regular sessions.
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
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.rename(columns=str.title)

    required = ["Open", "High", "Low", "Close", "Volume"]

    for col in required:
        if col not in data.columns:
            data[col] = np.nan

    data = data[required].dropna(subset=["Close"])

    return data


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def add_indicators(df):

    d = df.copy()

    close = d["Close"]
    high = d["High"]
    low = d["Low"]
    volume = d["Volume"]

    # Moving averages
    d["SMA20"] = close.rolling(20).mean()
    d["SMA50"] = close.rolling(50).mean()
    d["SMA100"] = close.rolling(100).mean()
    d["SMA200"] = close.rolling(200).mean()

    d["EMA12"] = close.ewm(span=12, adjust=False).mean()
    d["EMA26"] = close.ewm(span=26, adjust=False).mean()

    # MACD
    d["MACD"] = d["EMA12"] - d["EMA26"]
    d["MACDSignal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACDHist"] = d["MACD"] - d["MACDSignal"]

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    d["RSI"] = 100 - (100 / (1 + rs))

    # ATR
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
    d["ATR"] = tr.rolling(14).mean()
    d["ATR_pct"] = d["ATR"] / close * 100

    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()

    d["BBMid"] = bb_mid
    d["BBUpper"] = bb_mid + 2 * bb_std
    d["BBLower"] = bb_mid - 2 * bb_std

    band_width = d["BBUpper"] - d["BBLower"]

    d["BBPosition"] = (
        (close - d["BBLower"])
        / band_width.replace(0, np.nan)
        * 100
    )

    # ROC
    d["ROC20"] = close.pct_change(20) * 100

    # Stochastic
    lowest = low.rolling(14).min()
    highest = high.rolling(14).max()

    d["StochK"] = (
        100
        * (close - lowest)
        / (highest - lowest).replace(0, np.nan)
    )

    d["StochD"] = d["StochK"].rolling(3).mean()

    # CCI
    typical = (high + low + close) / 3
    tp_sma = typical.rolling(20).mean()
    mad = typical.rolling(20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True,
    )

    d["CCI"] = (
        (typical - tp_sma)
        / (0.015 * mad.replace(0, np.nan))
    )

    # OBV
    direction = np.sign(close.diff()).fillna(0)
    d["OBV"] = (direction * volume).cumsum()

    # Relative volume
    d["AvgVolume20"] = volume.rolling(20).mean()
    d["RelativeVolume"] = (
        volume / d["AvgVolume20"].replace(0, np.nan)
    )

    # Historical volatility
    returns = close.pct_change()

    d["DailyReturn"] = returns
    d["Volatility20"] = returns.rolling(20).std() * np.sqrt(252) * 100
    d["Volatility60"] = returns.rolling(60).std() * np.sqrt(252) * 100

    # Z-score
    rolling_mean = close.rolling(20).mean()
    rolling_std = close.rolling(20).std()

    d["ZScore20"] = (
        (close - rolling_mean)
        / rolling_std.replace(0, np.nan)
    )

    # Percentile rank
    d["PricePercentile"] = close.rolling(252).rank(pct=True) * 100

    # Trend slopes
    d["SMA20Slope"] = d["SMA20"].pct_change(10) * 100
    d["SMA50Slope"] = d["SMA50"].pct_change(20) * 100

    # ADX
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where(
        (plus_dm > minus_dm) & (plus_dm > 0),
        0,
    )

    minus_dm = minus_dm.where(
        (minus_dm > plus_dm) & (minus_dm > 0),
        0,
    )

    atr14 = d["ATR"].replace(0, np.nan)

    plus_di = 100 * plus_dm.rolling(14).sum() / atr14
    minus_di = 100 * minus_dm.rolling(14).sum() / atr14

    dx = (
        (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
        * 100
    )

    d["ADX"] = dx.rolling(14).mean()

    return d


# ============================================================
# MATHEMATICAL METRICS
# ============================================================

def max_drawdown(equity):
    running_max = equity.cummax()
    dd = equity / running_max - 1
    return dd.min() * 100


def sharpe_ratio(returns):
    if returns.std() == 0 or len(returns) < 2:
        return np.nan
    return returns.mean() / returns.std() * np.sqrt(252)


def sortino_ratio(returns):
    downside = returns[returns < 0]

    if downside.std() == 0 or len(downside) < 2:
        return np.nan

    return returns.mean() / downside.std() * np.sqrt(252)


def historical_stats(df):

    returns = df["DailyReturn"].dropna()

    equity = (1 + returns).cumprod()

    total_return = (equity.iloc[-1] - 1) * 100

    years = max(len(returns) / 252, 0.1)

    cagr = (
        (equity.iloc[-1] ** (1 / years)) - 1
    ) * 100

    win_rate = (returns > 0).mean() * 100

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    avg_win = wins.mean() if len(wins) else 0
    avg_loss = abs(losses.mean()) if len(losses) else 0

    payoff = (
        avg_win / avg_loss
        if avg_loss > 0
        else np.nan
    )

    expectancy = (
        (win_rate / 100) * avg_win
        - (1 - win_rate / 100) * avg_loss
    )

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Win Rate": win_rate,
        "Avg Win": avg_win * 100,
        "Avg Loss": avg_loss * 100,
        "Payoff": payoff,
        "Expectancy": expectancy * 100,
        "Sharpe": sharpe_ratio(returns),
        "Sortino": sortino_ratio(returns),
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

    # Trend
    if x["Close"] > x["SMA20"]:
        score += 5
        reasons.append(("Trend", "+5", "Price is above SMA20."))
    else:
        score -= 5
        reasons.append(("Trend", "-5", "Price is below SMA20."))

    if x["SMA20"] > x["SMA50"]:
        score += 6
        reasons.append(("Trend", "+6", "SMA20 is above SMA50."))
    else:
        score -= 6
        reasons.append(("Trend", "-6", "SMA20 is below SMA50."))

    if x["SMA50"] > x["SMA200"]:
        score += 7
        reasons.append(("Trend", "+7", "SMA50 is above SMA200."))
    else:
        score -= 7
        reasons.append(("Trend", "-7", "SMA50 is below SMA200."))

    # RSI
    rsi = x["RSI"]

    if pd.notna(rsi):
        if 50 <= rsi < 70:
            score += 8
            reasons.append(
                ("Momentum", "+8", explain(rsi, "RSI"))
            )
        elif rsi > 70:
            score -= 2
            reasons.append(
                ("Momentum", "-2", explain(rsi, "RSI"))
            )
        elif rsi < 30:
            score += 2
            reasons.append(
                ("Momentum", "+2", explain(rsi, "RSI"))
            )
        else:
            score -= 3
            reasons.append(
                ("Momentum", "-3", explain(rsi, "RSI"))
            )

    # MACD
    if pd.notna(x["MACD"]) and pd.notna(x["MACDSignal"]):

        if x["MACD"] > x["MACDSignal"]:
            score += 8
            reasons.append(
                ("Momentum", "+8", "MACD is above its signal line.")
            )
        else:
            score -= 8
            reasons.append(
                ("Momentum", "-8", "MACD is below its signal line.")
            )

    # ADX
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

    # Volume
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

    # ROC
    if pd.notna(x["ROC20"]):

        if x["ROC20"] > 0:
            score += 5
            reasons.append(
                ("ROC", "+5", "20-day return is positive.")
            )
        else:
            score -= 5
            reasons.append(
                ("ROC", "-5", "20-day return is negative.")
            )

    # Bollinger
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

def position_math(
    capital,
    price,
    stop,
    target,
    win_probability,
    payoff_ratio,
):

    risk_per_share = abs(price - stop)

    if risk_per_share <= 0:
        return {
            "shares": 0,
            "risk": 0,
            "profit": 0,
            "kelly": 0,
            "quarter_kelly": 0,
            "rr": np.nan,
        }

    p = np.clip(win_probability, 0.01, 0.99)
    q = 1 - p
    b = max(payoff_ratio, 0.01)

    # Kelly: f = (bp - q) / b
    kelly = (b * p - q) / b
    quarter_kelly = max(0, kelly / 4)

    # Never allow more than 2% capital risk in this UI calculation.
    risk_budget = capital * min(quarter_kelly, 0.02)

    shares = math.floor(risk_budget / risk_per_share)

    potential_loss = shares * risk_per_share
    potential_profit = shares * max(target - price, 0)

    rr = (
        max(target - price, 0) / risk_per_share
        if risk_per_share > 0
        else np.nan
    )

    return {
        "shares": shares,
        "risk": potential_loss,
        "profit": potential_profit,
        "kelly": kelly,
        "quarter_kelly": quarter_kelly,
        "rr": rr,
    }


# ============================================================
# MONTE CARLO
# ============================================================

def monte_carlo(returns, simulations=3000, days=252):

    returns = returns.dropna().values

    if len(returns) < 30:
        return None

    rng = np.random.default_rng(42)

    sampled = rng.choice(
        returns,
        size=(simulations, days),
        replace=True,
    )

    paths = np.cumprod(1 + sampled, axis=1)

    final_values = paths[:, -1]

    drawdowns = []

    for path in paths:
        peak = np.maximum.accumulate(path)
        dd = path / peak - 1
        drawdowns.append(dd.min())

    return {
        "median_final": np.median(final_values),
        "p05_final": np.percentile(final_values, 5),
        "p95_final": np.percentile(final_values, 95),
        "median_drawdown": np.median(drawdowns),
        "p95_drawdown": np.percentile(drawdowns, 5),
    }


# ============================================================
# NEWS
# ============================================================

def get_news(ticker):

    try:
        obj = yf.Ticker(ticker)
        news = obj.news

        rows = []

        for item in news[:15]:

            content = item.get("content", item)

            title = content.get("title", "")
            publisher = content.get("provider", {}).get(
                "displayName",
                "",
            )

            link = (
                content.get("canonicalUrl", {})
                .get("url", "")
            )

            if title:
                rows.append(
                    {
                        "Title": title,
                        "Publisher": publisher,
                        "Link": link,
                    }
                )

        return rows

    except Exception:
        return []


# ============================================================
# CHART
# ============================================================

def price_chart(df, ticker):

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        )
    )

    for col, color in [
        ("SMA20", "#22c55e"),
        ("SMA50", "#f59e0b"),
        ("SMA200", "#ef4444"),
    ]:

        if col in df:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    name=col,
                    line=dict(color=color, width=1.5),
                )
            )

    fig.update_layout(
        title=f"{ticker} — Price & Trend",
        height=600,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
    )

    return fig


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Quant Research AI")

ticker = st.sidebar.text_input(
    "Ticker",
    value="AAPL",
    help="Examples: AAPL, MSFT, RELIANCE.NS, 7203.T, 0700.HK",
).upper().strip()

period = st.sidebar.selectbox(
    "Historical Data",
    [
        "6mo",
        "1y",
        "2y",
        "5y",
        "10y",
    ],
    index=2,
)

capital = st.sidebar.number_input(
    "Trading Capital",
    min_value=0.0,
    value=10000.0,
    step=100.0,
)

risk_stop_atr = st.sidebar.slider(
    "Stop Distance — ATR ×",
    0.5,
    5.0,
    2.0,
    0.1,
)

target_atr = st.sidebar.slider(
    "Target Distance — ATR ×",
    0.5,
    10.0,
    4.0,
    0.1,
)

run = st.sidebar.button(
    "🔄 Analyze Ticker",
    type="primary",
    use_container_width=True,
)


# ============================================================
# MAIN
# ============================================================

st.title("🧠 Quant Research AI")

st.caption(
    "Mathematical analysis, technical signals, statistical risk and explainable scoring — "
    "not a guarantee of future returns."
)

if not ticker:
    st.info("Enter a ticker to begin.")
    st.stop()

market = detect_market(ticker)
symbol = get_currency_symbol(market["currency"])

status, local_time = session_status(market)

data = download_data(
    ticker,
    period=period,
    interval="1d",
)

if data.empty:
    st.error(
        "No market data was returned. Check the ticker symbol or exchange suffix."
    )
    st.stop()

data = add_indicators(data)

last = data.iloc[-1]

price = float(last["Close"])

# ============================================================
# HEADER
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Price",
    f"{symbol}{price:,.2f}",
)

c2.metric(
    "Market",
    market["country"],
)

c3.metric(
    "Currency",
    market["currency"],
)

c4.metric(
    "Exchange",
    market["exchange"],
)

c5.metric(
    "Session",
    status,
)

st.caption(
    f"Local exchange time: {local_time.strftime('%Y-%m-%d %H:%M:%S %Z')} "
    f"• Session status is approximate and does not independently verify exchange holidays."
)


# ============================================================
# SIGNAL
# ============================================================

score, regime, action, reasons = build_signal(data)

st.subheader("🎯 Explainable Signal")

if regime == "BULLISH":
    color = "green"
elif regime == "BEARISH":
    color = "red"
else:
    color = "orange"

st.markdown(
    f"### :{color}[{regime}]"
)

a, b, c = st.columns(3)

a.metric(
    "Quant Score",
    f"{score:.1f}/100",
)

b.metric(
    "Signal",
    action,
)

c.metric(
    "Current Price",
    f"{symbol}{price:,.2f}",
)

st.info(
    "One-line interpretation: "
    f"the model is **{regime.lower()}** because the combined trend, momentum, "
    "volume and volatility evidence produces the displayed score."
)


# ============================================================
# REASON BREAKDOWN
# ============================================================

st.subheader("🔍 Why did the model give this score?")

reason_df = pd.DataFrame(
    reasons,
    columns=["Factor", "Score Impact", "One-Line Mathematical Interpretation"],
)

st.dataframe(
    reason_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# CHART
# ============================================================

st.plotly_chart(
    price_chart(data.tail(300), ticker),
    use_container_width=True,
)


# ============================================================
# TECHNICAL + MATHEMATICAL INDICATORS
# ============================================================

st.subheader("📐 Technical & Mathematical Engine")

indicator_rows = [
    (
        "RSI",
        last["RSI"],
        "RSI = 100 − 100/(1 + RS)",
        explain(last["RSI"], "RSI")
        if pd.notna(last["RSI"])
        else "Insufficient data.",
    ),
    (
        "MACD",
        last["MACD"],
        "MACD = EMA12 − EMA26",
        explain(last["MACD"], "MACD")
        if pd.notna(last["MACD"])
        else "Insufficient data.",
    ),
    (
        "ADX",
        last["ADX"],
        "ADX measures trend strength from directional movement.",
        explain(last["ADX"], "ADX")
        if pd.notna(last["ADX"])
        else "Insufficient data.",
    ),
    (
        "ATR %",
        last["ATR_pct"],
        "ATR% = ATR / Price × 100",
        "Higher ATR% means larger expected daily price movement.",
    ),
    (
        "Bollinger Position",
        last["BBPosition"],
        "Position = (Price − Lower Band)/(Upper − Lower) × 100",
        "Shows where price sits inside its recent volatility envelope.",
    ),
    (
        "20D ROC %",
        last["ROC20"],
        "ROC = (Price_t / Price_t−20 − 1) × 100",
        "Positive ROC means price gained over the last 20 sessions.",
    ),
    (
        "Stochastic %K",
        last["StochK"],
        "%K = 100 × (Close − Lowest)/(Highest − Lowest)",
        "Measures the closing price's position inside its recent range.",
    ),
    (
        "CCI",
        last["CCI"],
        "CCI = (Typical Price − SMA) / (0.015 × Mean Deviation)",
        "Measures deviation from the recent average price.",
    ),
    (
        "Relative Volume",
        last["RelativeVolume"],
        "RVOL = Current Volume / 20D Average Volume",
        explain(last["RelativeVolume"], "RelativeVolume")
        if pd.notna(last["RelativeVolume"])
        else "Insufficient data.",
    ),
    (
        "Z-Score",
        last["ZScore20"],
        "Z = (Price − Rolling Mean) / Rolling Standard Deviation",
        explain(last["ZScore20"], "ZScore")
        if pd.notna(last["ZScore20"])
        else "Insufficient data.",
    ),
    (
        "Historical Volatility",
        last["Volatility20"],
        "σannual = Std(Daily Returns) × √252",
        "Higher volatility means a wider distribution of possible outcomes.",
    ),
]

indicator_df = pd.DataFrame(
    indicator_rows,
    columns=[
        "Indicator",
        "Current Value",
        "Mathematical Definition",
        "One-Line Meaning",
    ],
)

st.dataframe(
    indicator_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# TREND REGIME
# ============================================================

st.subheader("📊 Trend Regime")

trend_cols = st.columns(4)

trend_values = [
    ("SMA20", last["SMA20"]),
    ("SMA50", last["SMA50"]),
    ("SMA200", last["SMA200"]),
    ("ADX", last["ADX"]),
]

for col, (name, value) in zip(trend_cols, trend_values):

    if pd.notna(value):
        col.metric(
            name,
            f"{value:,.2f}",
        )

if pd.notna(last["ADX"]):

    if last["ADX"] >= 25:
        st.success(
            f"Trend strength: STRONGER TREND — ADX = {last['ADX']:.2f}. "
            "Direction still comes from the price/DI relationship."
        )
    else:
        st.warning(
            f"Trend strength: WEAK — ADX = {last['ADX']:.2f}. "
            "Sideways/choppy conditions are more plausible."
        )


# ============================================================
# RISK / TRADE MATH
# ============================================================

st.subheader("💰 Risk & Position Mathematics")

atr = float(last["ATR"]) if pd.notna(last["ATR"]) else price * 0.02

stop = price - risk_stop_atr * atr
target = price + target_atr * atr

stats = historical_stats(data)

# Conservative estimate from historical statistics.
win_probability = np.clip(
    stats["Win Rate"] / 100,
    0.05,
    0.95,
)

payoff = (
    stats["Payoff"]
    if pd.notna(stats["Payoff"])
    else 1.5
)

pm = position_math(
    capital,
    price,
    stop,
    target,
    win_probability,
    payoff,
)

r1, r2, r3, r4 = st.columns(4)

r1.metric(
    "Calculated Stop",
    f"{symbol}{stop:,.2f}",
)

r2.metric(
    "Calculated Target",
    f"{symbol}{target:,.2f}",
)

r3.metric(
    "Risk / Reward",
    f"{pm['rr']:.2f}x"
    if pd.notna(pm["rr"])
    else "N/A",
)

r4.metric(
    "Quarter-Kelly",
    f"{pm['quarter_kelly'] * 100:.2f}%",
)

r5, r6, r7 = st.columns(3)

r5.metric(
    "Suggested Shares",
    f"{pm['shares']:,}",
)

r6.metric(
    "Potential Loss",
    f"{symbol}{pm['risk']:,.2f}",
)

r7.metric(
    "Potential Profit",
    f"{symbol}{pm['profit']:,.2f}",
)

st.caption(
    "One-line explanation: position size is limited by stop distance and "
    "fractional Kelly; this is a risk model, not a prediction that the target will be reached."
)


# ============================================================
# HISTORICAL STATISTICS
# ============================================================

st.subheader("📈 Historical Quant Statistics")

hs = historical_stats(data)

stats_df = pd.DataFrame(
    [
        ["Total Return", f"{hs['Total Return']:.2f}%", "Historical price return over the selected sample."],
        ["CAGR", f"{hs['CAGR']:.2f}%", "Annualized compounded historical growth rate."],
        ["Win Rate", f"{hs['Win Rate']:.2f}%", "Percentage of positive daily returns."],
        ["Average Win", f"{hs['Avg Win']:.3f}%", "Average positive daily return."],
        ["Average Loss", f"{hs['Avg Loss']:.3f}%", "Average absolute negative daily return."],
        ["Payoff Ratio", f"{hs['Payoff']:.2f}", "Average win divided by average loss."],
        ["Expectancy", f"{hs['Expectancy']:.4f}%", "Probability-weighted average daily return."],
        ["Sharpe", f"{hs['Sharpe']:.2f}", "Return relative to total volatility."],
        ["Sortino", f"{hs['Sortino']:.2f}", "Return relative to downside volatility."],
        ["Max Drawdown", f"{hs['Max Drawdown']:.2f}%", "Largest historical peak-to-trough decline."],
    ],
    columns=["Metric", "Value", "One-Line Meaning"],
)

st.dataframe(
    stats_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# EXPECTED VALUE
# ============================================================

st.subheader("🧮 Expected Value")

p = win_probability
r = payoff

ev = (p * r) - (1 - p)

st.markdown(
    f"""
**EV = (Win Probability × Payoff) − (Loss Probability × 1)**

**EV = ({p:.3f} × {r:.3f}) − ({1-p:.3f} × 1) = `{ev:.3f}`**

One-line interpretation: **{"positive historical expectancy" if ev > 0 else "negative historical expectancy"} under these assumptions.**
"""
)


# ============================================================
# MONTE CARLO
# ============================================================

st.subheader("🎲 Monte Carlo Risk Simulation")

mc = monte_carlo(
    data["DailyReturn"].dropna(),
    simulations=3000,
    days=252,
)

if mc:

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Median 1Y Growth Factor",
        f"{mc['median_final']:.2f}x",
    )

    m2.metric(
        "5th Percentile Final",
        f"{mc['p05_final']:.2f}x",
    )

    m3.metric(
        "95th Percentile Final",
        f"{mc['p95_final']:.2f}x",
    )

    st.write(
        f"Median simulated maximum drawdown: "
        f"**{mc['median_drawdown'] * 100:.2f}%**"
    )

    st.write(
        f"5th-percentile drawdown boundary: "
        f"**{mc['p95_drawdown'] * 100:.2f}%**"
    )

    st.caption(
        "Monte Carlo resamples historical returns; it is a scenario distribution, "
        "not a forecast of the actual future."
    )


# ============================================================
# NEWS / RUMOURS
# ============================================================

st.subheader("📰 Recent News / Market Information")

news = get_news(ticker)

if news:

    for item in news:

        title = item["Title"]
        publisher = item["Publisher"]
        link = item["Link"]

        if link:
            st.markdown(
                f"**{title}**  \n"
                f"{publisher}  \n"
                f"[Open source]({link})"
            )
        else:
            st.markdown(
                f"**{title}**  \n{publisher}"
            )

        st.divider()

else:
    st.info(
        "No news items were returned by the available Yahoo Finance feed."
    )

st.warning(
    "News/rumour headlines are information inputs, not verified facts or trading signals. "
    "The application does not treat rumours as confirmed events."
)


# ============================================================
# DATA QUALITY / TRUST
# ============================================================

st.subheader("🛡️ How much should you trust this?")

st.markdown(
    """
### What is mathematically reliable?

**High confidence**
- Indicator arithmetic calculated from downloaded prices.
- Moving averages.
- RSI calculation.
- ATR calculation.
- Volatility calculation.
- Z-score calculation.
- Historical return statistics.

**Not guaranteed**
- Future price.
- Future news.
- Future market regime.
- Probability estimates outside the historical sample.
- Profit.
- Execution price.
- Slippage.
- Market liquidity.

### The correct interpretation

The application should **never say "100% BUY" or "99.99% guaranteed profit."**

Instead:

> **The mathematics can be exact while the prediction remains uncertain.**

A 78/100 score means the current collection of measured factors is favorable under this model; it does **not** mean there is a 78% probability that the stock will rise.
"""
)


# ============================================================
# DATA TABLE
# ============================================================

with st.expander("🔬 Raw Indicator Data"):

    display_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SMA20",
        "SMA50",
        "SMA200",
        "RSI",
        "MACD",
        "MACDSignal",
        "ATR",
        "ATR_pct",
        "BBPosition",
        "ROC20",
        "StochK",
        "CCI",
        "RelativeVolume",
        "Volatility20",
        "ZScore20",
        "ADX",
    ]

    available = [
        c for c in display_cols
        if c in data.columns
    ]

    st.dataframe(
        data[available].tail(100).round(4),
        use_container_width=True,
    )


# ============================================================
# FINAL DECISION SUMMARY
# ============================================================

st.subheader("🧠 One-Line Quant Conclusion")

if regime == "BULLISH":

    conclusion = (
        f"{ticker}: BULLISH bias — score {score:.1f}/100; "
        "trend/momentum evidence is currently stronger than bearish evidence."
    )

elif regime == "BEARISH":

    conclusion = (
        f"{ticker}: BEARISH bias — score {score:.1f}/100; "
        "trend/momentum evidence is currently weaker than bearish evidence."
    )

else:

    conclusion = (
        f"{ticker}: SIDEWAYS/MIXED — score {score:.1f}/100; "
        "the mathematical evidence is not strong enough for a directional conclusion."
    )

st.info(conclusion)

st.caption(
    "Research tool only — not financial advice. No model can guarantee future market direction."
)
