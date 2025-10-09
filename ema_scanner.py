import yfinance as yf
import pandas as pd
import requests
import time
import datetime
import pytz
import logging
from io import StringIO
import os

# === CONFIG ===
EMA_FAST = 13
EMA_SLOW = 21
EMA_TREND = 200
TIMEFRAME = "1d"
CHECK_INTERVAL = 900       # 15 minutes
HOLD_DAYS = 5
CAPITAL_PER_TRADE = 500
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1425616478601871400/AMbiCffNSI7lOsqLPBZ5UDPOStNW0UgcAJAqMU0D1QxDzD2EymlnrbTQxN44XErNkaXm"
TEST_ALERT_ON_START = False
LOG_FILE = "signals_log.csv"

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ----------------------------------------------------------------------
#                         DATA SOURCES
# ----------------------------------------------------------------------

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0.0.0 Safari/537.36"
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        tables = pd.read_html(StringIO(r.text))
        tickers = tables[0]["Symbol"].tolist()
        logging.info(f"✅ Loaded {len(tickers)} S&P 500 tickers from Wikipedia.")
        return tickers
    except Exception as e:
        logging.warning(f"⚠️ Error loading S&P 500 list: {e}")
        return [
            "AAPL", "MSFT", "AMZN", "GOOG", "META", "NVDA", "TSLA", "JPM", "V", "HD",
            "MA", "UNH", "KO", "PEP", "DIS", "NFLX", "BAC", "PFE", "ADBE", "COST"
        ]

def get_biotech_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_biotechnology_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        html_tables = pd.read_html(StringIO(r.text))
        symbols = []
        for df in html_tables:
            for col in df.columns:
                if "Ticker" in col or "Symbol" in col:
                    symbols.extend(df[col].dropna().astype(str).tolist())
        symbols = [s for s in symbols if s.isalpha()]
        if symbols:
            logging.info(f"✅ Loaded {len(symbols)} biotech tickers from Wikipedia.")
            return symbols
    except Exception as e:
        logging.warning(f"⚠️ Error loading biotech list: {e}")
    fallback = [
        "BIIB","REGN","VRTX","GILD","ILMN","MRNA","BMRN","ALNY","EXEL","NBIX",
        "INCY","SRPT","RPRX","IONS","TECH","CYT","CRSP","DXCM","VIR","AMGN"
    ]
    logging.info(f"Using fallback biotech list of {len(fallback)} tickers.")
    return fallback

def get_all_tickers():
    tickers = list(set(get_sp500_tickers() + get_biotech_tickers()))
    logging.info(f"Total tickers to scan: {len(tickers)}")
    return tickers

# ----------------------------------------------------------------------
#                         DISCORD ALERT
# ----------------------------------------------------------------------

def notify_discord(message: str):
    if not DISCORD_WEBHOOK:
        logging.warning("⚠️ No webhook URL configured — alert not sent.")
        return
    try:
        requests.post(DISCORD_WEBHOOK, json={"content": message})
    except Exception as e:
        logging.error(f"Error sending to Discord: {e}")

# ----------------------------------------------------------------------
#                         CORE SCANNER
# ----------------------------------------------------------------------

def scan_symbol(sym):
    try:
        df_d = yf.download(sym, period="200d", interval="1d", progress=False, auto_adjust=False)
        if df_d.empty or len(df_d) < EMA_SLOW + 2:
            return None

        df_d["ema_fast"] = df_d["Close"].ewm(span=EMA_FAST, adjust=False).mean()
        df_d["ema_slow"] = df_d["Close"].ewm(span=EMA_SLOW, adjust=False).mean()

        cross_up = (
            df_d["ema_fast"].shift(2).iloc[-1] < df_d["ema_slow"].shift(2).iloc[-1]
        ) and (
            df_d["ema_fast"].shift(1).iloc[-1] > df_d["ema_slow"].shift(1).iloc[-1]
        )

        cross_dn = (
            df_d["ema_fast"].shift(2).iloc[-1] > df_d["ema_slow"].shift(2).iloc[-1]
        ) and (
            df_d["ema_fast"].shift(1).iloc[-1] < df_d["ema_slow"].shift(1).iloc[-1]
        )

        df_4h = yf.download(sym, period="200d", interval="4h", progress=False, auto_adjust=False)
        if df_4h.empty:
            return None
        df_4h["ema_trend"] = df_4h["Close"].ewm(span=EMA_TREND, adjust=False).mean()

        ema_4h_last = float(df_4h["ema_trend"].iloc[-1])
        last_close = float(df_d["Close"].iloc[-1])
        trend_up = last_close > ema_4h_last
        trend_dn = last_close < ema_4h_last

        if cross_up and trend_up:
            return ("BUY", sym, last_close)
        elif cross_dn and trend_dn:
            return ("SELL", sym, last_close)
        else:
            return None
    except Exception as e:
        logging.error(f"Error scanning {sym}: {e}")
        return None

# ----------------------------------------------------------------------
#                         BACKTEST TRACKER
# ----------------------------------------------------------------------

def record_signal(signal_type, sym, price):
    date = datetime.date.today().isoformat()
    df = pd.DataFrame([[date, signal_type, sym, price]], columns=["date", "signal", "symbol", "price"])
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)

def evaluate_old_
