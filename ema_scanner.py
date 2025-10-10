# ============================================================
#         EMA Multi-Factor Scanner Bot — Pro Edition
# ============================================================
# Scans S&P500 + NASDAQ100 + Dow30 + Biotech + Sector ETFs
# Sends Discord alerts for confirmed EMA(13/21) crossovers
# Uses 4H EMA200 trend filter + RSI/ADX/OBV/ATR confirmations
# Includes 5-day backtest performance summary
# ============================================================

import re
import os
import ta
import time
import logging
import datetime
import traceback
import urllib.request

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from flask import Flask
from threading import Thread
from pandas.api.types import is_numeric_dtype

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "https://discord.com/api/webhooks/REPLACE_WITH_YOUR_WEBHOOK")

EMA_FAST = int(os.getenv("EMA_FAST", 13))
EMA_SLOW = int(os.getenv("EMA_SLOW", 21))
EMA_TREND = int(os.getenv("EMA_TREND", 200))

TIMEFRAME_DAILY = os.getenv("TIMEFRAME_DAILY", "1d")
TIMEFRAME_4H = os.getenv("TIMEFRAME_4H", "4h")

CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", 900))  # 15 minutes
HOLD_DAYS = int(os.getenv("HOLD_DAYS", 5))
CAPITAL_PER_TRADE = float(os.getenv("CAPITAL_PER_TRADE", 500))
LOG_FILE = os.getenv("LOG_FILE", "trades_log.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M"
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def send_discord_message(content: str):
    if (not DISCORD_WEBHOOK) or ("REPLACE_WITH_YOUR_WEBHOOK" in DISCORD_WEBHOOK):
        logging.warning("⚠️ No valid webhook URL configured — alert not sent.")
        return
    try:
        resp = requests.post(DISCORD_WEBHOOK, json={"content": content}, timeout=10)
        if resp.status_code >= 300:
            logging.error(f"❌ Discord send failed: {resp.status_code} {resp.text}")
        else:
            logging.info("✅ Discord alert sent.")
    except Exception as e:
        logging.error(f"❌ Discord send failed: {e}")

def to_float(x):
    try:
        if isinstance(x, (list, np.ndarray)) and len(x) > 0:
            x = x[0]
        return float(x)
    except Exception:
        return np.nan

def _normalize_ticker(x):
    if x is None:
        return None
    s = str(x).strip().upper()
    if not s:
        return None
    s = s.split()[0]
    s = s.replace(".", "-")
    if not re.fullmatch(r"[A-Z\-]+", s):
        return None
    if len(s) > 12:
        return None
    return s

def normalize_tickers(seq):
    out = []
    for x in seq:
        s = _normalize_ticker(x)
        if s:
            out.append(s)
    return out

def safe_read_html(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as resp:
        html = resp.read()
    return pd.read_html(html)

# ----------------------------------------------------------------------
# Ticker universes
# ----------------------------------------------------------------------

def get_sp500_tickers():
    try:
        tables = safe_read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        tickers = df["Symbol"].astype(str).tolist()
        logging.info(f"Loaded {len(tickers)} S&P 500 tickers from Wikipedia.")
        return tickers
    except Exception as e:
        logging.error(f"S&P500 fetch failed: {e}")
        return ["AAPL","MSFT","TSLA","NVDA","AMZN","GOOG","META"]

def get_biotech_tickers():
    try:
        tables = safe_read_html("https://en.wikipedia.org/wiki/List_of_biotechnology_companies")
        bio = tables[0]
        tickers = bio.iloc[:, 0].dropna().astype(str).tolist()[:100]
        logging.info(f"Loaded {len(tickers)} biotech tickers from Wikipedia.")
        return tickers
    except Exception as e:
        logging.error(f"Biotech list error: {e}")
        return ["BIIB","REGN","VRTX","GILD","ALNY","ILMN"]

def get_nasdaq100_tickers():
    try:
        tables = safe_read_html("https://en.wikipedia.org/wiki/NASDAQ-100")
        df = None
        for t in tables:
            if "Ticker" in t.columns:
                df = t
                break
        if df is None:
            df = tables[4]
        tickers = df["Ticker"].dropna().astype(str).tolist()
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
# Indicators
# ----------------------------------------------------------------------

def fetch_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
    df["adx"] = adx.adx()
    obv = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"])
    df["obv"] = obv.on_balance_volume()
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14)
    df["atr"] = atr.average_true_range()
    return df

def _ensure_flat_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def _unbox_arraylike_cells(df: pd.DataFrame, cols):
    for col in cols:
        if col in df.columns and len(df[col]) > 0 and isinstance(df[col].iloc[0], (np.ndarray, list)):
            df[col] = df[col].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x)
    return df

def _to_numeric_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ----------------------------------------------------------------------
# Data fetch + EMA logic
# ----------------------------------------------------------------------

def fetch_emas_and_filters(symbol: str):
    try:
        df_daily = yf.download(
            symbol, period="120d", interval=TIMEFRAME_DAILY,
            progress=False, auto_adjust=False, threads=False
        )
        df_4h = yf.download(
            symbol, period="180d", interval=TIMEFRAME_4H,
            progress=False, auto_adjust=False, threads=False
        )

        if df_daily is None or df_daily.empty or df_4h is None or df_4h.empty:
            raise ValueError("No data returned")

        df_daily = _ensure_flat_columns(df_daily)
        df_4h = _ensure_flat_columns(df_4h)

        cols = ["Open", "High", "Low", "Close", "Volume"]
        df_daily = _unbox_arraylike_cells(df_daily, cols)
        df_4h = _unbox_arraylike_cells(df_4h, cols)

        if len(df_daily) < max(EMA_SLOW, 50) or len(df_4h) < max(EMA_TREND, 200):
            raise ValueError("Insufficient data")

        df_daily = _to_numeric_cols(df_daily, cols)
        df_4h = _to_numeric_cols(df_4h, cols)

        df_daily = fetch_indicators(df_daily)
        df_daily["ema_fast"] = df_daily["Close"].ewm(span=EMA_FAST, adjust=False).mean()
        df_daily["ema_slow"] = df_daily["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
        df_4h["ema_trend"] = df_4h["Close"].ewm(span=EMA_TREND, adjust=False).mean()

        ema_trend = to_float(df_4h["ema_trend"].iloc[-1])
        return df_daily, ema_trend

    except Exception as e:
        raise ValueError(f"{symbol}: {e}")

# ----------------------------------------------------------------------
# Scanner
# ----------------------------------------------------------------------

def scan_tickers(tickers):
    conf_signals = []

    for sym in tickers:
        try:
            df, ema_trend = fetch_emas_and_filters(sym)

            # Convert non-numeric columns when possible (no deprecated errors="ignore")
            for col in df.columns:
                if is_numeric_dtype(df[col]):
                    continue
                try:
                    df[col] = pd.to_numeric(df[col])
                except Exception:
                    pass

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

            if any(pd.isna(v) or (isinstance(v, float) and np.isnan(v)) for v in vals):
                logging.debug(f"Skipping {sym}: NaN or invalid numeric data.")
                continue

            crossed_up     = (prev_fast < prev_slow) and (ema_fast > ema_slow)
            crossed_down   = (prev_fast > prev_slow) and (ema_fast < ema_slow)
            is_above_trend = (close > ema_trend)
            is_below_trend = (close < ema_trend)

            try:
                obv_rising = float(obv.iloc[-1]) > float(obv.iloc[-5]) if len(obv) > 5 else True
            except Exception:
                obv_rising = True

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
# Backtest
# ----------------------------------------------------------------------

def record_signal(signal_type, sym, price):
    date = datetime.date.today().isoformat()
    df = pd.DataFrame([[date, signal_type, sym, float(price)]],
                      columns=["date","signal","symbol","price"])
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)

def evaluate_old_signals():
    if not os.path.exists(LOG_FILE):
        return None

    try:
        df = pd.read_csv(LOG_FILE)
    except Exception as e:
        logging.error(f"Failed to read log file '{LOG_FILE}': {e}")
        return None

    if df.empty:
        return None

    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    except Exception:
        pass

    cutoff = datetime.date.today() - datetime.timedelta(days=HOLD_DAYS)
    eligible = df[df["date"].dt.date <= cutoff]
    if eligible.empty:
        return None

    results = []
    for _, row in eligible.iterrows():
        sym  = str(row["symbol"])
        side = str(row["signal"])
        entry = to_float(row["price"])
        entry_date = row["date"]
        try:
            df_hist = yf.download(sym, start=entry_date, period="10d",
                                  interval="1d", progress=False)
            if df_hist is None or len(df_hist) < HOLD_DAYS:
                continue
            exit_price = float(df_hist["Close"].iloc[HOLD_DAYS - 1])
            ret = (exit_price - entry) / entry if entry else 0.0
            if side.upper() == "SELL":
                ret = -ret
            profit = ret * CAPITAL_PER_TRADE
            results.append((sym, side, float(entry), float(exit_price), float(profit)))
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

    df_remaining = df[~df["symbol"].isin([r[0] for r in results])]
    try:
        df_remaining.to_csv(LOG_FILE, index=False)
    except Exception as e:
        logging.error(f"Failed to update log file: {e}")

    return msg

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _build_universe():
    raw = (
        get_sp500_tickers()
        + get_biotech_tickers()
        + get_nasdaq100_tickers()
        + get_dow30_tickers()
        + get_sector_tickers()
    )
    tickers = sorted(set(normalize_tickers(raw)))
    return tickers

def main():
    try:
        tickers = _build_universe()
        logging.info(f"Total tickers to scan: {len(tickers)}")
        logging.info("🚀 EMA Multi-Factor Scanner Started")
    except Exception as e:
        logging.error(f"Failed to build ticker universe: {e}")
        raise

    while True:
        try:
            conf_signals = scan_tickers(tickers)

            if conf_signals:
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                msg = f"**✅ EMA Multi-Factor Alerts ({now})**\n" + "\n".join(conf_signals)
                send_discord_message(msg)
            else:
                logging.info(f"{datetime.datetime.now():%H:%M} — No signals")

            report = evaluate_old_signals()
            if report:
                send_discord_message(report)

        except Exception as e:
            logging.error(f"⚠️ Global loop error: {type(e).__name__} — {e}")
            traceback.print_exc()

        time.sleep(CHECK_INTERVAL)

# ----------------------------------------------------------------------
# Flask (keep-alive)
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
        traceback.print_exc()
