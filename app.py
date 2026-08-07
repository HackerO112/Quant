import sys
import subprocess
import time

# Auto-install essential runtime dependencies if missing
REQUIRED_PACKAGES = {
    'yfinance': 'yfinance',
    'plotly': 'plotly',
    'xgboost': 'xgboost',
    'lightgbm': 'lightgbm',
    'catboost': 'catboost',
    'sklearn': 'scikit-learn',
    'scipy': 'scipy',
    'statsmodels': 'statsmodels'
}

for module_name, package_name in REQUIRED_PACKAGES.items():
    try:
        __import__(module_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# =========================================================
# 1. BALANCED MONOCHROMATIC (MEDIUM GREY / HIGH VISIBILITY)
# =========================================================
st.set_page_config(
    page_title="Black-Box Quant AI Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Balanced Medium Monochromatic Theme (Neither Dark nor Light) */
    html, body, [class*="css"], .stApp {
        background-color: #1E293B !important; /* Medium Slate Grey */
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }

    /* Force All Text, Labels, Headers to Crisp Pure White */
    p, span, div, label, h1, h2, h3, h4, h5, h6, .stMarkdown {
        color: #FFFFFF !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #334155 !important;
        border-right: 2px solid #475569 !important;
    }

    /* Metric Cards - Mid Monochromatic Contrast */
    .stMetric, .metric-card {
        background-color: #334155 !important;
        border: 2px solid #64748B !important;
        border-radius: 10px !important;
        padding: 18px !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25) !important;
    }

    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: #F1F5F9 !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    /* Black-Box AI Single Signal Container */
    .blackbox-card {
        background: #334155 !important;
        border: 3px solid #64748B;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3);
    }

    /* Tabs Styling for Monochromatic Theme */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #334155 !important;
        border-radius: 8px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #475569 !important;
        border-radius: 6px;
    }

    /* High Visibility Signal Accents */
    .buy-signal { color: #34D399 !important; font-weight: 900; }  /* Vivid Emerald */
    .sell-signal { color: #F87171 !important; font-weight: 900; } /* Vivid Rose */
    .hold-signal { color: #FBBF24 !important; font-weight: 900; } /* Vivid Amber */
</style>
""", unsafe_allow_html=True)

CURRENCY_SYMBOLS = {
    'USD': '$', 'EUR': '€', 'GBP': '£', 'JPY': '¥', 'INR': '₹', 'CAD': 'CA$', 'AUD': 'A$'
}

# ==========================================
# 2. HIGH-PRECISION DATA PIPELINE
# ==========================================
def fetch_high_precision_market_data(ticker_symbol: str, period="3y"):
    """Fetches market data with exact sub-millisecond execution timing."""
    start_time_ns = time.time_ns()
    try:
        t = yf.Ticker(ticker_symbol)
        df = t.history(period=period, auto_adjust=False)
        
        exec_latency_ms = (time.time_ns() - start_time_ns) / 1e6
        
        if df.empty or len(df) < 100:
            return None, {}, None, exec_latency_ms
        
        info = t.info or {}
        financials = {
            "income": t.financials,
            "balance": t.balance_sheet,
            "cashflow": t.cashflow,
            "dividends": t.dividends,
            "splits": t.splits
        }
        return df, info, financials, exec_latency_ms
    except Exception:
        return None, {}, None, 0.0

def build_precision_features(df: pd.DataFrame, horizon=3):
    """Generates precise technical and volatility indicators."""
    data = df.copy()
    
    data['Returns'] = data['Close'].pct_change()
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    
    log_hl = np.log(data['High'] / data['Low'].replace(0, np.nan)) ** 2
    log_co = np.log(data['Close'] / data['Open'].replace(0, np.nan)) ** 2
    data['Garman_Klass'] = np.sqrt(0.5 * log_hl - (2 * np.log(2) - 1) * log_co)
    data['Hist_Vol_20'] = data['Returns'].rolling(20).std() * np.sqrt(252)
    
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = tr.rolling(14).mean()
    data['ATR_Pct'] = data['ATR'] / data['Close']
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    raw_rsi = 100 - (100 / (1 + rs))
    data['RSI'] = raw_rsi
    data['RSI_Z'] = (raw_rsi - raw_rsi.rolling(50).mean()) / (raw_rsi.rolling(50).std() + 1e-9)
    
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['SMA_200'] = data['Close'].rolling(200).mean()
    
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd_diff = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    data['MACD_Z'] = (macd_diff - macd_diff.rolling(50).mean()) / (macd_diff.rolling(50).std() + 1e-9)
    
    data['Vol_Ratio_Z'] = (data['Volume'] - data['Volume'].rolling(20).mean()) / (data['Volume'].rolling(20).std() + 1e-9)
    
    future_max = data['High'].shift(-horizon).rolling(horizon).max()
    future_min = data['Low'].shift(-horizon).rolling(horizon).min()
    data['Target'] = ((future_max >= data['Close'] + (1.2 * data['ATR'])) & (future_min > data['Close'] - (1.2 * data['ATR']))).astype(float)
    data.loc[future_max.isna(), 'Target'] = np.nan
    
    return data

# ==========================================
# 3. BLACK-BOX SINGLE CONSENSUS SIGNAL ENGINE
# ==========================================
def train_blackbox_ai_ensemble(df: pd.DataFrame):
    """Combines 7 ML algorithms into ONE consensus decision."""
    feature_cols = ['Garman_Klass', 'Hist_Vol_20', 'ATR_Pct', 'RSI_Z', 'MACD_Z', 'Vol_Ratio_Z']
    
    df_clean = df.dropna(subset=feature_cols).copy()
    train_df = df_clean.dropna(subset=['Target']).copy()
    
    X_train_full = train_df[feature_cols]
    y_train_full = train_df['Target'].astype(int)
    
    split_idx = int(len(X_train_full) * 0.8)
    X_tr, X_te = X_train_full.iloc[:split_idx], X_train_full.iloc[split_idx:]
    y_tr, y_te = y_train_full.iloc[:split_idx], y_train_full.iloc[split_idx:]
    
    selector = SelectKBest(score_func=mutual_info_classif, k=min(5, len(feature_cols)))
    X_tr_sel = selector.fit_transform(X_tr, y_tr)
    X_te_sel = selector.transform(X_te)
    
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_sel)
    X_te_scaled = scaler.transform(X_te_sel)
    
    models = [
        ('XGBoost', XGBClassifier(n_estimators=30, max_depth=2, learning_rate=0.01, random_state=42, eval_metric='logloss')),
        ('LightGBM', LGBMClassifier(n_estimators=30, max_depth=2, learning_rate=0.01, random_state=42, verbose=-1)),
        ('CatBoost', CatBoostClassifier(iterations=30, depth=2, learning_rate=0.01, verbose=0, random_state=42)),
        ('RandomForest', RandomForestClassifier(n_estimators=40, max_depth=2, random_state=42)),
        ('ExtraTrees', ExtraTreesClassifier(n_estimators=40, max_depth=2, random_state=42)),
        ('GradBoost', GradientBoostingClassifier(n_estimators=30, max_depth=2, learning_rate=0.01, random_state=42)),
        ('LogReg', LogisticRegression(C=0.01, random_state=42))
    ]
    
    model_probs = {}
    for name, model in models:
        model.fit(X_tr_scaled, y_tr)
        model_probs[name] = model.predict_proba(X_te_scaled)[:, 1]
        
    meta_model = LogisticRegression(C=0.05)
    meta_X_tr = np.column_stack([m.predict_proba(X_tr_scaled)[:, 1] for _, m in models])
    meta_X_te = np.column_stack([model_probs[name] for name, _ in models])
    meta_model.fit(meta_X_tr, y_tr)
    
    ensemble_test_probs = meta_model.predict_proba(meta_X_te)[:, 1]
    oos_acc = np.mean((ensemble_test_probs >= 0.50).astype(int) == y_te)
    
    # --- LIVE INFERENCE FIX (Reshaping to 1 sample x 7 features) ---
    latest_feature_row = df_clean[feature_cols].iloc[[-1]]
    latest_scaled = scaler.transform(selector.transform(latest_feature_row))
    latest_model_probs = [m.predict_proba(latest_scaled)[0][1] for _, m in models]
    
    latest_meta_X = np.array(latest_model_probs).reshape(1, -1)
    master_bull_prob = float(meta_model.predict_proba(latest_meta_X)[0][1])
    
    # ONE Master Signal Decision
    if master_bull_prob >= 0.55:
        signal = "BUY"
        confidence = master_bull_prob
    elif master_bull_prob <= 0.45:
        signal = "SELL"
        confidence = 1.0 - master_bull_prob
    else:
        signal = "HOLD"
        confidence = 1.0 - abs(master_bull_prob - 0.5) * 2

    return signal, confidence, master_bull_prob, oos_acc, ensemble_test_probs, y_te

# ==========================================
# 4. SIDEBAR CONTROLS
# ==========================================
st.sidebar.markdown("## ⚙️ Configuration")

preset_ticker = st.sidebar.selectbox("Select Asset", ["NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "RELIANCE.NS", "Custom"])
ticker_symbol = st.sidebar.text_input("Custom Symbol:", value="NVDA").upper() if preset_ticker == "Custom" else preset_ticker

portfolio_capital = st.sidebar.number_input("Portfolio Capital ($)", value=100000, step=5000)
max_trade_risk_pct = st.sidebar.slider("Max Risk per Trade (% Capital)", 0.5, 5.0, 2.0, 0.5)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data Feed"):
    st.rerun()

# ==========================================
# 5. DATA PROCESSING & EXECUTION
# ==========================================
raw_df, info_dict, financials_dict, latency_ms = fetch_high_precision_market_data(ticker_symbol)

if raw_df is None or len(raw_df) < 100:
    st.error(f"Insufficient market data for symbol **{ticker_symbol}**.")
    st.stop()

processed_df = build_precision_features(raw_df)
master_signal, signal_conf, live_prob, oos_acc, test_probs, y_test = train_blackbox_ai_ensemble(processed_df)

curr_code = str(info_dict.get('currency', 'USD')).upper()
curr_sym = CURRENCY_SYMBOLS.get(curr_code, f"{curr_code} ")

latest_price = float(processed_df['Close'].iloc[-1])
prev_close = float(processed_df['Close'].iloc[-2])
change_amt = latest_price - prev_close
change_pct = (change_amt / prev_close) * 100

now_utc = datetime.datetime.now(datetime.timezone.utc)
formatted_time_ms = now_utc.strftime('%H:%M:%S') + f".{int(now_utc.microsecond / 1000):03d} UTC"

# ==========================================
# 6. HEADER & BLACK-BOX SINGLE SIGNAL
# ==========================================
st.title(f"⚡ {info_dict.get('longName', ticker_symbol)} ({ticker_symbol})")
st.caption(f"Latency: **{latency_ms:.2f} ms** | Timestamp: **{formatted_time_ms}** | Currency: **{curr_code}**")

# --- ONE MASTER BLACK-BOX AI SIGNAL BANNER ---
if master_signal == "BUY":
    sig_class = "buy-signal"
    border_color = "#34D399"
    badge = "🚀 CONSENSUS BULLISH"
elif master_signal == "SELL":
    sig_class = "sell-signal"
    border_color = "#F87171"
    badge = "🔻 CONSENSUS BEARISH"
else:
    sig_class = "hold-signal"
    border_color = "#FBBF24"
    badge = "⚖️ CONSENSUS NEUTRAL"

st.markdown(f"""
<div class="blackbox-card" style="border-color: {border_color};">
    <div style="font-size: 0.95rem; color: #CBD5E1; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
        Black-Box AI Single Signal ({badge})
    </div>
    <div style="font-size: 3.8rem; line-height: 1;" class="{sig_class}">
        {master_signal}
    </div>
    <div style="font-size: 1.15rem; color: #FFFFFF; margin-top: 10px;">
        Signal Confidence: <b style="color: #60A5FA;">{signal_conf:.1%}</b> | Master Probability: <b>{live_prob:.1%}</b>
    </div>
</div>
""", unsafe_allow_html=True)

# Price Key Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Live Price", f"{curr_sym}{latest_price:,.3f}", f"{change_pct:+.2f}%")
c2.metric("Day High", f"{curr_sym}{processed_df['High'].iloc[-1]:,.3f}")
c3.metric("Day Low", f"{curr_sym}{processed_df['Low'].iloc[-1]:,.3f}")
c4.metric("Volume", f"{processed_df['Volume'].iloc[-1]:,.0f}")

st.divider()

# ==========================================
# 7. WORKSTATION TABS
# ==========================================
tab_sizer, tab_backtest, tab_chart, tab_financials = st.tabs([
    "🧮 Position Sizer",
    "📈 Backtest Performance",
    "📊 Precision Chart",
    "📋 Financial Statements"
])

# ------------------------------------------
# TAB 1: KELLY POSITION SIZER
# ------------------------------------------
with tab_sizer:
    st.subheader("🧮 Risk & Position Sizing")
    
    col_s1, col_s2 = st.columns(2)
    atr_val = float(processed_df['ATR'].dropna().iloc[-1])
    
    with col_s1:
        st.markdown("#### Trade Setup")
        rr_ratio = st.number_input("Target Reward / Risk Ratio", value=2.0, step=0.5)
        
        win_prob = max(min(oos_acc, 0.80), 0.51)
        kelly_fraction = max(0.0, (win_prob * rr_ratio - (1 - win_prob)) / rr_ratio) * 0.5
        
        stop_dist = float(1.5 * atr_val)
        target_dist = stop_dist * rr_ratio
        
        max_risk_dollars = portfolio_capital * (max_trade_risk_pct / 100.0)
        shares_by_risk = int(max_risk_dollars / (stop_dist + 1e-9))
        shares_by_kelly = int((portfolio_capital * kelly_fraction) / latest_price) if latest_price > 0 else 0
        
        recommended_shares = max(0, min(shares_by_risk, shares_by_kelly)) if master_signal != "HOLD" else 0
        allocated_capital = recommended_shares * latest_price

        st.metric("Recommended Allocation", f"{recommended_shares} Shares", f"{curr_sym}{allocated_capital:,.2f} Invested")

    with col_s2:
        st.markdown("#### Scenario Targets")
        target_price = latest_price + target_dist
        stop_price = max(0.01, latest_price - stop_dist)
        
        max_profit = recommended_shares * target_dist
        max_loss = recommended_shares * stop_dist

        st.metric("🎯 Target Price", f"{curr_sym}{target_price:,.3f}", f"+{curr_sym}{max_profit:,.2f} Profit")
        st.metric("🛑 Stop Loss", f"{curr_sym}{stop_price:,.3f}", f"-{curr_sym}{max_loss:,.2f} Loss")

# ------------------------------------------
# TAB 2: BACKTESTING
# ------------------------------------------
with tab_backtest:
    st.subheader("📈 Out-of-Sample Strategy Backtest")
    
    test_data = processed_df.dropna(subset=['Target']).iloc[-len(test_probs):].copy()
    if not test_data.empty:
        test_data['Signal'] = (test_probs >= 0.50).astype(int)
        test_data['Strat_Returns'] = test_data['Signal'].shift(1) * test_data['Returns']
        
        test_data['Cum_Benchmark'] = (1 + test_data['Returns'].fillna(0)).cumprod()
        test_data['Cum_Strategy'] = (1 + test_data['Strat_Returns'].fillna(0)).cumprod()
        
        strat_ret = (test_data['Cum_Strategy'].iloc[-1] - 1)
        bench_ret = (test_data['Cum_Benchmark'].iloc[-1] - 1)

        b1, b2, b3 = st.columns(3)
        b1.metric("Strategy Return", f"{strat_ret:+.1%}")
        b2.metric("Benchmark Return", f"{bench_ret:+.1%}")
        b3.metric("OOS Accuracy", f"{oos_acc:.1%}")

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=test_data.index, y=test_data['Cum_Strategy'], name="Strategy", line=dict(color='#34D399', width=2.5)))
        fig_bt.add_trace(go.Scatter(x=test_data.index, y=test_data['Cum_Benchmark'], name="Buy & Hold", line=dict(color='#94A3B8', dash='dash')))
        
        fig_bt.update_layout(
            paper_bgcolor='#334155',
            plot_bgcolor='#1E293B',
            font=dict(color='#FFFFFF'),
            height=400,
            title="Strategy Equity Curve",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_bt, use_container_width=True)

# ------------------------------------------
# TAB 3: CHARTING
# ------------------------------------------
with tab_chart:
    st.subheader("📊 Price Action Chart")
    
    fig_chart = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig_chart.add_trace(go.Candlestick(
        x=processed_df.index[-180:], open=processed_df['Open'][-180:], high=processed_df['High'][-180:],
        low=processed_df['Low'][-180:], close=processed_df['Close'][-180:], name="Price"
    ), row=1, col=1)
    
    fig_chart.add_trace(go.Scatter(x=processed_df.index[-180:], y=processed_df['SMA_20'][-180:], name="20 SMA", line=dict(color='#FBBF24', width=1.5)), row=1, col=1)
    fig_chart.add_trace(go.Scatter(x=processed_df.index[-180:], y=processed_df['SMA_50'][-180:], name="50 SMA", line=dict(color='#60A5FA', width=1.5)), row=1, col=1)
    
    colors = ['#34D399' if c >= o else '#F87171' for c, o in zip(processed_df['Close'][-180:], processed_df['Open'][-180:])]
    fig_chart.add_trace(go.Bar(x=processed_df.index[-180:], y=processed_df['Volume'][-180:], marker_color=colors, name="Volume"), row=2, col=1)
    
    fig_chart.update_layout(
        paper_bgcolor='#334155',
        plot_bgcolor='#1E293B',
        font=dict(color='#FFFFFF'),
        height=500,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_chart, use_container_width=True)

# ------------------------------------------
# TAB 4: FINANCIAL STATEMENTS
# ------------------------------------------
with tab_financials:
    st.subheader("📋 Financial Statements")
    stmt_choice = st.radio("Statement View:", ["Income Statement", "Balance Sheet", "Cash Flow"], horizontal=True)
    
    stmt_map = {
        "Income Statement": financials_dict.get("income"),
        "Balance Sheet": financials_dict.get("balance"),
        "Cash Flow": financials_dict.get("cashflow")
    }
    
    df_stmt = stmt_map.get(stmt_choice)
    if df_stmt is not None and not df_stmt.empty:
        st.dataframe(df_stmt.style.format(precision=2), use_container_width=True)
    else:
        st.info("Statement data unavailable for this asset.")

st.caption("⚠️ **Disclaimer:** For educational and quantitative research purposes only. Not financial advice.")