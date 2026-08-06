import os
import csv
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
import streamlit as st
import yfinance as yf

# ==========================================
# PAGE CONFIGURATION & MOBILE CSS
# ==========================================
st.set_page_config(
    page_title="Verified Institutional Quant Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Mobile Optimization and High-Contrast UI
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
    }
    .main .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }
    .signal-card {
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .audit-badge {
        background-color: #1e222d;
        border: 1px solid #363a45;
        padding: 10px 14px;
        border-radius: 8px;
        font-size: 0.88rem;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Institutional AI Quant Engine & Beginner Terminal")
st.caption("Walk-Forward Cross-Validated | Leak-Proof | Explicit UTF-8 Windows Compatible")

# ==========================================
# DATA PIPELINE (FIXES MULTIINDEX & CLOSED MARKET)
# ==========================================
def fetch_robust_market_data(symbol, interval, period):
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        
        # Closed market fallback: If intraday data is empty, revert to daily
        if data.empty and interval != "1d":
            data = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
            interval = "1d"
            
        if data.empty:
            return pd.DataFrame(), interval, False

        # Flatten yfinance MultiIndex columns tuple
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
            
        df = data.reset_index()
        
        # Standardize Date column name
        date_col = next((col for col in df.columns if str(col).lower() in ["date", "datetime", "index"]), df.columns[0])
        df.rename(columns={date_col: "Date"}, inplace=True)

        # Standardize OHLCV capitalization
        column_mapping = {}
        for col in df.columns:
            c_lower = str(col).lower()
            if "open" in c_lower: column_mapping[col] = "Open"
            elif "high" in c_lower: column_mapping[col] = "High"
            elif "low" in c_lower: column_mapping[col] = "Low"
            elif "close" in c_lower: column_mapping[col] = "Close"
            elif "vol" in c_lower: column_mapping[col] = "Volume"

        df.rename(columns=column_mapping, inplace=True)
        
        req = ["Date", "Open", "High", "Low", "Close"]
        if not all(c in df.columns for c in req):
            return pd.DataFrame(), interval, False

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        df["Volume"] = pd.to_numeric(df.get("Volume", 1.0), errors="coerce").fillna(1.0)
        df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"]).sort_values("Date").reset_index(drop=True)
        
        return df, interval, True
    except Exception:
        return pd.DataFrame(), interval, False

# ==========================================
# TECHNICAL INDICATORS CALCULATOR ENGINE
# ==========================================
def compute_indicators(data):
    df = data.copy()
    
    # 1. RSI (14)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # 2. MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # 3. EMAs & Bollinger Bands
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

    # 4. ATR (14) - Dynamic Volatility
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    true_range = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1)
    df["ATR_14"] = true_range.rolling(window=14).mean()

    # 5. Volume Ratio
    df["Vol_Ratio"] = df["Volume"] / (df["Volume"].rolling(window=20).mean() + 1e-9)

    # Features & Machine Learning Targets
    df["Prev_Close"] = df["Close"].shift(1)
    df["Momentum_1D"] = df["Close"].shift(1) - df["Open"].shift(1)
    df["Target_Return"] = (df["Close"].shift(-1) - df["Close"]) / df["Close"]

    return df

# ==========================================
# UTF-8 SAFE CSV AUDIT LOGGER (WINDOWS FIX)
# ==========================================
def log_prediction_to_csv(symbol, current_price, target_price, signal, score, fee_barrier):
    log_file = "quant_live_audit_log.csv"
    file_exists = os.path.isfile(log_file)
    # Explicit encoding="utf-8" prevents Windows cp1252 character map errors
    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp_UTC", "Symbol", "LivePrice", "PredictedTarget", "Signal", "ConfidenceScore", "FeeBarrierUsed"])
        writer.writerow([
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            symbol,
            round(current_price, 2),
            round(target_price, 2),
            signal,
            score,
            fee_barrier
        ])

# ==========================================
# SIDEBAR CONTROL & CONFIGURATION
# ==========================================
st.sidebar.header("⚙️ Market & Risk Controls")

preset_ticker = st.sidebar.selectbox(
    "Asset Presets:",
    ("TATSILV.NS", "RELIANCE.NS", "AAPL", "NVDA", "BTC-USD", "GC=F (Gold)", "Custom")
)

if preset_ticker == "Custom":
    ticker_symbol = st.sidebar.text_input("Enter Yahoo Finance Ticker:", "TATSILV.NS")
elif "Gold" in preset_ticker:
    ticker_symbol = "GC=F"
else:
    ticker_symbol = preset_ticker

data_granularity = st.sidebar.radio("Horizon:", ("Intraday 5-Min", "Intraday 15-Min", "Daily Timeline"))

if data_granularity == "Intraday 5-Min":
    requested_interval, requested_period = "5m", "5d"
elif data_granularity == "Intraday 15-Min":
    requested_interval, requested_period = "15m", "5d"
else:
    requested_interval, requested_period = "1d", "2y"

fee_barrier_pct = st.sidebar.slider("Slippage/Fee Barrier (%):", 0.01, 0.20, 0.05, step=0.01) / 100.0
live_auto_refresh = st.sidebar.checkbox("🔄 Auto-Refresh (60s)", value=False)

# Fetch Data
df_raw, active_interval, success = fetch_robust_market_data(ticker_symbol, requested_interval, requested_period)

if not success or df_raw.empty:
    st.error(f"Failed to fetch market data for `{ticker_symbol}`. Please verify ticker validity.")
    st.stop()

df = compute_indicators(df_raw)
dataset = df.dropna().copy()

if len(dataset) < 30:
    st.warning("Building technical history data bars...")
    st.stop()

# ==========================================
# WALK-FORWARD CROSS-VALIDATION ENGINE
# ==========================================
feature_cols = ["Prev_Close", "Momentum_1D", "RSI_14", "MACD_Hist", "EMA_20", "EMA_50", "ATR_14", "Vol_Ratio"]
X = dataset[feature_cols]
y = dataset["Target_Return"]

tscv = TimeSeriesSplit(n_splits=5)
oos_mae_scores = []
oos_directional_hits = []

for train_idx, val_idx in tscv.split(X):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    # Scaler fitted strictly on training fold to prevent data leakage
    fold_scaler = RobustScaler()
    X_train_scaled = fold_scaler.fit_transform(X_train_fold)
    X_val_scaled = fold_scaler.transform(X_val_fold)

    fold_model = HistGradientBoostingRegressor(max_iter=150, max_depth=5, learning_rate=0.03, random_state=42)
    fold_model.fit(X_train_scaled, y_train_fold)

    val_preds = fold_model.predict(X_val_scaled)
    oos_mae_scores.append(mean_absolute_error(y_val_fold, val_preds))
    
    directional_correct = np.sign(val_preds) == np.sign(y_val_fold.values)
    oos_directional_hits.append(np.mean(directional_correct))

avg_oos_mae = np.mean(oos_mae_scores)
avg_oos_win_rate = np.mean(oos_directional_hits) * 100.0

# Final Model Fit
final_scaler = RobustScaler()
X_full_scaled = final_scaler.fit_transform(X)
final_model = HistGradientBoostingRegressor(max_iter=200, max_depth=5, learning_rate=0.03, random_state=42)
final_model.fit(X_full_scaled, y)

# Live Bar Evaluation
terminal = df.iloc[-1]
live_vector = pd.DataFrame([[
    terminal["Close"],
    (terminal["Close"] - terminal["Open"]),
    terminal["RSI_14"],
    terminal["MACD_Hist"],
    terminal["EMA_20"],
    terminal["EMA_50"],
    terminal["ATR_14"],
    terminal["Vol_Ratio"]
]], columns=feature_cols)

live_scaled = final_scaler.transform(live_vector)
predicted_raw_return = float(final_model.predict(live_scaled).item())
latest_close = float(terminal["Close"])

# Fee & Slippage Barrier Deduction
if predicted_raw_return > 0:
    net_predicted_return = max(0.0, predicted_raw_return - fee_barrier_pct)
else:
    net_predicted_return = min(0.0, predicted_raw_return + fee_barrier_pct)

target_price = latest_close * (1 + net_predicted_return)

# ==========================================
# STRICT CONFLUENCE SIGNAL MATRIX
# ==========================================
live_rsi = float(terminal["RSI_14"])
live_macd = float(terminal["MACD_Hist"])
live_ema50 = float(terminal["EMA_50"])
live_atr = float(terminal["ATR_14"])

confidence_score = 50

if net_predicted_return > 0.001: confidence_score += 20
elif net_predicted_return < -0.001: confidence_score -= 20

if live_rsi < 35: confidence_score += 15
elif live_rsi > 65: confidence_score -= 15

if live_macd > 0: confidence_score += 10
else: confidence_score -= 10

if latest_close > live_ema50: confidence_score += 10
else: confidence_score -= 10

confidence_score = max(5, min(95, confidence_score))

if avg_oos_win_rate < 48.0:
    master_signal = "UNVERIFIED / HIGH NOISE ⚠️"
    signal_color = "#ffaa00"
    reasoning = "Model Walk-Forward Win Rate is below 48%. Trade signal suppressed to protect capital."
elif net_predicted_return == 0.0:
    master_signal = "HOLD / FEE BARRIER BLOCKED ⏸️"
    signal_color = "#cccccc"
    reasoning = f"Expected movement cannot clear the {fee_barrier_pct*100:.2f}% broker fee/slippage cost."
elif confidence_score >= 70 and net_predicted_return > 0:
    master_signal = "VERIFIED STRONG BUY 🟢"
    signal_color = "#00ff00"
    reasoning = "AI return forecast, RSI oversold/momentum, and EMA trend cross-confirm upward vector."
elif confidence_score <= 30 and net_predicted_return < 0:
    master_signal = "VERIFIED STRONG SELL 🔴"
    signal_color = "#ff0000"
    reasoning = "AI return forecast and technical indicators cross-confirm downward pressure."
else:
    master_signal = "NEUTRAL / HOLD ⏸️"
    signal_color = "#888888"
    reasoning = "Mixed technical indicators. No high-probability edge detected."

# Dynamic Risk Bounds
stop_loss = latest_close - (1.5 * live_atr)
take_profit = latest_close + (2.0 * live_atr)

# Log Prediction to CSV using UTF-8 Safe Handler
log_prediction_to_csv(ticker_symbol, latest_close, target_price, master_signal, confidence_score, fee_barrier_pct)

# ==========================================
# DASHBOARD UI LAYOUT
# ==========================================
last_time_str = terminal["Date"].strftime("%Y-%m-%d %H:%M UTC")

st.markdown(f"""
<div class="audit-badge">
    🛡️ <b>Walk-Forward Audit Verification:</b> Out-of-Sample Win Rate: <b>{avg_oos_win_rate:.1f}%</b> | Out-of-Sample MAE: <b>{avg_oos_mae*100:.3f}%</b>
</div>
""", unsafe_allow_html=True)

# Metrics Grid
m1, m2, m3, m4 = st.columns(4)
m1.metric("Live Market Price", f"{latest_close:.2f}")
m2.metric(f"Net Target ({active_interval})", f"{target_price:.2f}", f"{net_predicted_return*100:.2f}%")
m3.metric("AI Confidence", f"{confidence_score}%")
m4.metric("Volatility (ATR)", f"{live_atr:.2f}")

# Signal Banner Card
st.markdown(f"""
<div class="signal-card" style="border-left: 6px solid {signal_color}; background-color: #1e222d;">
    <h3 style="margin:0; color:{signal_color}; font-size:1.3rem;">{master_signal}</h3>
    <p style="margin-top:6px; margin-bottom:0; font-size:0.9rem; color:#d0d0d0;"><b>Logic:</b> {reasoning}</p>
</div>
""", unsafe_allow_html=True)

# Risk Bounds
r1, r2 = st.columns(2)
with r1:
    st.error(f"🛑 **Stop Loss:** `{stop_loss:.2f}` (Exit if price drops here)")
with r2:
    st.success(f"🎯 **Take Profit:** `{take_profit:.2f}` (Target ceiling)")

st.markdown("---")

# ==========================================
# BEGINNER INDICATOR BREAKDOWN
# ==========================================
st.subheader("💡 Technical Indicator Readings")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**1. Momentum (RSI 14)**")
    if live_rsi < 35: st.success(f"🟢 Oversold ({live_rsi:.1f}): Rebound candidate.")
    elif live_rsi > 65: st.error(f"🔴 Overbought ({live_rsi:.1f}): Resistance risk.")
    else: st.info(f"🔵 Neutral ({live_rsi:.1f}): Balanced momentum.")

with c2:
    st.markdown("**2. Trend (EMA 50)**")
    if latest_close > live_ema50: st.success(f"🟢 Bullish: Price > 50-EMA ({live_ema50:.2f})")
    else: st.error(f"🔴 Bearish: Price < 50-EMA ({live_ema50:.2f})")

with c3:
    st.markdown("**3. Volume Confirmation**")
    vol_r = float(terminal["Vol_Ratio"])
    if vol_r > 1.3: st.success(f"🟢 Institutional Spike ({vol_r:.1f}x avg)")
    else: st.info(f"🔵 Standard Retail ({vol_r:.1f}x avg)")

st.markdown("---")

# ==========================================
# RESPONSIVE PLOTLY CHART
# ==========================================
st.subheader("📈 Multi-Indicator Chart")

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    vertical_spacing=0.06, row_heights=[0.7, 0.3]
)

plot_df = df.tail(80)

fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Close"], name="Price", line=dict(color="#00bfff", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["EMA_20"], name="EMA 20", line=dict(color="#ffaa00", width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["EMA_50"], name="EMA 50", line=dict(color="#ff00ff", width=1)), row=1, col=1)

future_stamp = plot_df["Date"].iloc[-1] + pd.Timedelta(minutes=5 if active_interval=="5m" else 15 if active_interval=="15m" else 1440)
fig.add_trace(go.Scatter(
    x=[plot_df["Date"].iloc[-1], future_stamp], y=[latest_close, target_price],
    mode="lines+markers", name="Net AI Vector",
    line=dict(color="#00ff00" if net_predicted_return >= 0 else "#ff0000", width=2, dash="dash"),
    marker=dict(size=8, color="#ffaa00")
), row=1, col=1)

fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["RSI_14"], name="RSI", line=dict(color="#ab63fa", width=1.5)), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig.update_layout(
    template="plotly_dark", height=450,
    margin=dict(l=10, r=10, t=20, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True, config={'responsive': True})

# ==========================================
# UTF-8 AUDIT LOG READOUT
# ==========================================
with st.expander("📄 Open Live Audit Prediction CSV Log"):
    if os.path.exists("quant_live_audit_log.csv"):
        log_df = pd.read_csv("quant_live_audit_log.csv", encoding="utf-8")
        st.dataframe(log_df.tail(20), use_container_width=True)
        st.download_button(
            "Download Audit Log CSV",
            data=log_df.to_csv(index=False).encode("utf-8"),
            file_name="quant_live_audit_log.csv",
            mime="text/csv"
        )
    else:
        st.info("No prediction logs recorded yet.")

if live_auto_refresh:
    time.sleep(60)
    st.rerun()