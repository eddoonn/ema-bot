# ============================================================
#         EMA Multi-Factor Scanner Bot — Pro Edition
# ============================================================
# Scans S&P500 + NASDAQ100 + Dow30 + Biotech + Sector ETFs
# Sends Discord alerts for confirmed EMA(13/21) crossovers
# Uses 4H EMA200 trend filter + RSI/ADX/OBV/ATR confirmations
# Includes 5-day backtest performance summary
# ============================================================

import yfinance as yf
import pandas as pd
import requests
import time
import datetime
import logging
import os
import pytz
from io import StringIO
import ta
import numpy as np
from flask import Flask
from threading import Thread
import urllib.request

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

DISCORD_WEBHOOK = "https://discord.com/api/webhooks/REPLACE_WITH_YOUR_WEBHOOK"

EMA_FAST = 13
EMA_SLOW = 21
EMA_TREND = 200

TIMEFRAME_DAILY = "1d"
TIMEFRAME_4H = "4h"
CHECK_INTERVAL = 900  # 15 minutes
HOLD_DAYS = 5
CAPITAL_PER_TRADE = 500
LOG_FILE = "trades_log.csv"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M"
)

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------

def send_discord_message(content: str):
    """Send a message to Discord webhook."""
    if not DISCORD_WEBHOOK:
        logging.warning("⚠️ No webhook URL configured — alert not sent.")
        return
    try:
        requests.post(DISCORD_WEBHOOK, json={"content": content})
        logging.info("✅ Discord alert sent.")
    except Exception as e:
        logging.error(f"❌ Discord send failed: {e}")

def to_float(x):
    """Safely convert anything to float or NaN."""
    try:
        if isinstance(x, (list, np.ndarray)):
            x = x[0]
        return float(x)
    except Exception:
        return np.nan

# ----------------------------------------------------------------------
# Ticker universe fetchers
# ----------------------------------------------------------------------

def safe_read_html(url):
    """Fetch HTML with custom User-Agent to avoid 403."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        html = resp.read()
    return pd.read_html(html)

def get_sp500_tickers():
    try:
        tables = safe_read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        tickers = df["Symbol"].tolist()
        logging.info(f"Loaded {len(tickers)} S&P 500 tickers from Wikipedia.")
        return tickers
    except Exception as e:
        logging.error(f"S&P500 fetch failed: {e}")
        return ["AAPL","MSFT","TSLA","NVDA","AMZN","GOOG","META"]

def get_biotech_tickers():
    try:
        tables = safe_read_html("https://en.wikipedia.org/wiki/List_of_biotechnology_companies")
        bio = tables[0]
        tickers = bio.iloc[:,0].dropna().tolist()[:100]
        logging.info(f"Loaded {len(tickers)} biotech tickers from Wikipedia.")
        return tickers
    except Exception as e:
        logging.error(f"Biotech list error: {e}")
        return ["BIIB","REGN","VRTX","GILD","ALNY","ILMN"]

def get_nasdaq100_tickers():
    try:
        tables = safe_read_html("https://en.wikipedia.org/wiki/NASDAQ-100")
        df = tables[4]
        tickers = df["Ticker"].dropna().tolist()
        logging.info(f"Loaded {len(tickers)} NASDAQ-100 tickers from Wikipedia.")
        return tickers
    except Exception as e:
        logging.error(f"NASDAQ100 fetch failed: {e}")
        return ["AAPL","MSFT","NVDA","META","AMZN","GOOG","TSLA"]

def get_dow30_tickers():
    try:
        tables = safe_read_html("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")
        for t in tables:
            cols = [c.lower() for c in t.columns.astype(str)]
            possible_cols = [c for c in cols if "symbol" in c or "ticker" in c]
            if possible_cols:
                tickers = t[t.columns[cols.index(possible_cols[0])]].dropna().astype(str).tolist()
                logging.info(f"Loaded {len(tickers)} Dow 30 tickers from Wikipedia.")
                return tickers
        raise ValueError("No symbol/ticker column found.")
    except Exception as e:
        logging.error(f"Dow30 fetch failed: {e}")
        return ["AAPL", "MSFT", "CAT", "BA", "JNJ", "PG", "V"]

def get_sector_tickers():
    return ["XLE","XLF","XLK","XLV","XLI","XLY","XLU","XLB","XLC","XBI","SMH","SOXX"]

# ----------------------------------------------------------------------
# Technical indicator helpers
# ----------------------------------------------------------------------

def fetch_indicators(df):
    """Add RSI, ADX, OBV, ATR columns."""
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
    df["adx"] = adx.adx()
    obv = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"])
    df["obv"] = obv.on_balance_volume()
    df["atr"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    return df

# ----------------------------------------------------------------------
# Core EMA + multi-factor logic
# ----------------------------------------------------------------------

def fetch_emas_and_filters(symbol):
    """Download data and compute indicators safely."""
    try:
        df_daily = yf.download(symbol, period="120d", interval=TIMEFRAME_DAILY,
                               progress=False, auto_adjust=False, threads=False)
        df_4h = yf.download(symbol, period="180d", interval=TIMEFRAME_4H,
                            progress=False, auto_adjust=False, threads=False)

        if isinstance(df_daily.columns, pd.MultiIndex):
            df_daily.columns = [col[0] for col in df_daily.columns]
        if isinstance(df_4h.columns, pd.MultiIndex):
            df_4h.columns = [col[0] for col in df_4h.columns]

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df_daily and isinstance(df_daily[col].iloc[0], (np.ndarray, list)):
                df_daily[col] = df_daily[col].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
            if col in df_4h and isinstance(df_4h[col].iloc[0], (np.ndarray, list)):
                df_4h[col] = df_4h[col].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)

        if len(df_daily) < 50 or len(df_4h) < 200:
            raise ValueError("Insufficient data")

        df_daily = fetch_indicators(df_daily)
        df_daily["ema_fast"] = df_daily["Close"].ewm(span=EMA_FAST).mean()
        df_daily["ema_slow"] = df_daily["Close"].ewm(span=EMA_SLOW).mean()
        df_4h["ema_trend"] = df_4h["Close"].ewm(span=EMA_TREND).mean()

        ema_trend = float(df_4h["ema_trend"].iloc[-1])
        return df_daily, ema_trend
    except Exception as e:
        raise ValueError(f"{symbol}: {e}")

# ----------------------------------------------------------------------
# Scanner
# ----------------------------------------------------------------------

def scan_tickers(tickers):
    """Main scanner with full numeric safety and indicator filters."""
    conf_signals = []

    for sym in tickers:
        try:
            df, ema_trend = fetch_emas_and_filters(sym)
            for col in df.columns:
                df[col] = df[col].apply(to_float)
            ema_trend = to_float(ema_trend)

            prev_fast = to_float(df["ema_fast"].iloc[-2])
            prev_slow = to_float(df["ema_slow"].iloc[-2])
            ema_fast  = to_float(df["ema_fast"].iloc[-1])
            ema_slow  = to_float(df["ema_slow"].iloc[-1])
            close     = to_float(df["Close"].iloc[-1])
            rsi       = to_float(df["rsi"].iloc[-1])
            adx       = to_float(df["adx"].iloc[-1])
            atr       = to_float(df["atr"].iloc[-1])
            atr_mean  = to_float(df["atr"].rolling(100).mean().iloc[-1])
            obv       = df["obv"].copy()

            vals = [prev_fast, prev_slow, ema_fast, ema_slow,
                    close, rsi, adx, atr, atr_mean, ema_trend]
            if any(map(lambda x: pd.isna(x) or np.isnan(x), vals)):
                logging.warning(f"Skipping {sym}: NaN or invalid numeric data.")
                continue

            # --- Diagnostic block ---
            try:
                _ = [float(v) for v in vals]
            except Exception as e:
                logging.error(f"Type error for {sym}: {e} | "
                              f"types={[type(v) for v in vals]}")
                continue

            crossed_up   = prev_fast < prev_slow and ema_fast > ema_slow
            crossed_down = prev_fast > prev_slow and ema_fast < ema_slow
            is_above_trend = close > ema_trend
            is_below_trend = close < ema_trend
            obv_rising = obv.iloc[-1] > obv.iloc[-5] if len(obv) > 5 else True

            if crossed_up and is_above_trend and rsi > 55 and adx > 20 and atr > 0.8 * atr_mean and obv_rising:
                msg = f"📈 {sym} BUY @ {close:.2f} | RSI {rsi:.1f}, ADX {adx:.1f}"
                conf_signals.append(msg)
                record_signal("BUY", sym, close)

            elif crossed_down and is_below_trend and rsi < 45 and adx > 20 and atr > 0.8 * atr_mean and not obv_rising:
                msg = f"🔻 {sym} SELL @ {close:.2f} | RSI {rsi:.1f}, ADX {adx:.1f}"
                conf_signals.append(msg)
                record_signal("SELL", sym, close)

        except Exception as e:
            logging.error(f"{sym}: {e}")

    return conf_signals

# ----------------------------------------------------------------------
# Backtest Tracker
# ----------------------------------------------------------------------

def record_signal(signal_type, sym, price):
    date = datetime.date.today().isoformat()
    df = pd.DataFrame([[date, signal_type, sym, price]],
                      columns=["date","signal","symbol","price"])
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)

def evaluate_old_signals():
    if not os.path.exists(LOG_FILE):
        return None
    df = pd.read_csv(LOG_FILE)
    df["date"] = pd.to_datetime(df["date"])
    cutoff = datetime.date.today() - datetime.timedelta(days=HOLD_DAYS)
    eligible = df[df["date"].dt.date <= cutoff]
    if eligible.empty:
        return None

    results = []
    for _, row in eligible.iterrows():
        sym, entry, side, entry_date = row["symbol"], row["price"], row["signal"], row["date"]
        try:
            df_hist = yf.download(sym, start=entry_date, period="10d",
                                  interval="1d", progress=False)
            if len(df_hist) < HOLD_DAYS:
                continue
            exit_price = float(df_hist["Close"].iloc[HOLD_DAYS - 1])
            ret = (exit_price - entry) / entry
            if side == "SELL":
                ret = -ret
            profit = ret * CAPITAL_PER_TRADE
            results.append((sym, side, entry, exit_price, profit))
        except Exception as e:
            logging.error(f"Backtest {sym}: {e}")

    if not results:
        return None

    total = sum(p[4] for p in results)
    winrate = sum(1 for p in results if p[4] > 0) / len(results) * 100
    avg_trade = total / len(results)

    msg = "**📊 Weekly Performance Report**\n"
    for sym, side, entry, exitp, prof in results:
        emoji = "✅" if prof > 0 else "❌"
        msg += f"{emoji} {sym} {side} {entry:.2f} → {exitp:.2f} | {prof:+.2f} USD\n"
    msg += f"\n**Total:** {total:+.2f} USD | **Winrate:** {winrate:.1f}% | **Avg:** {avg_trade:+.2f} USD/trade"

    df = df[~df["symbol"].isin([r[0] for r in results])]
    df.to_csv(LOG_FILE, index=False)
    return msg

# ----------------------------------------------------------------------
# Main Loop
# ----------------------------------------------------------------------

def main():
    tickers = sorted(set(
        get_sp500_tickers() +
        get_biotech_tickers() +
        get_nasdaq100_tickers() +
        get_dow30_tickers() +
        get_sector_tickers()
    ))
    logging.info(f"Total tickers to scan: {len(tickers)}")
    logging.info("🚀 EMA Multi-Factor Scanner Started")

    while True:
        conf_signals = scan_tickers(tickers)

        if conf_signals:
            msg = f"**✅ EMA Multi-Factor Alerts ({datetime.datetime.now():%Y-%m-%d %H:%M})**\n" + "\n".join(conf_signals)
            send_discord_message(msg)
        else:
            logging.info(f"{datetime.datetime.now():%H:%M} — No signals")

        report = evaluate_old_signals()
        if report:
            send_discord_message(report)

        time.sleep(CHECK_INTERVAL)

# ----------------------------------------------------------------------
# Flask server to keep Render alive
# ----------------------------------------------------------------------

app = Flask(__name__)

@app.route('/')
def home():
    return "EMA Scanner running."

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------

if __name__ == "__main__":
    try:
        Thread(target=run_flask, daemon=True).start()
        main()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
