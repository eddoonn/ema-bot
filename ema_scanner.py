import yfinance as yf
import pandas as pd
import time
import requests
import datetime
import pytz
import logging
from urllib.request import Request, urlopen

# === CONFIG ===
EMA_FAST = 13
EMA_SLOW = 21
TIMEFRAME = "1h"           # "1h", "4h", "1d"
CHECK_INTERVAL = 15 * 60   # seconds
TIMEZONE = "America/New_York"
DISCORD_WEBHOOK = "PASTE_YOUR_DISCORD_WEBHOOK_HERE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# === HELPERS ===
def safe_html_read(url):
    """Load HTML with custom headers to avoid 403 errors."""
    try:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(req) as r:
            return pd.read_html(r.read())
    except Exception as e:
        logging.error("Error fetching %s: %s", url, e)
        return None


def get_sp500_tickers():
    """Scrape S&P 500 tickers from Wikipedia or fallback to GitHub list."""
    try:
        table = safe_html_read("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        if table and len(table) > 0:
            df = table[0]
            tickers = df["Symbol"].tolist()
            logging.info("Loaded %d S&P 500 tickers.", len(tickers))
            return tickers
    except Exception as e:
        logging.error("Error loading S&P 500 list: %s", e)

    # Fallback: GitHub static CSV of S&P 500 symbols
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv")
        return df["Symbol"].tolist()
    except Exception as e:
        logging.error("Fallback S&P list failed: %s", e)
        return ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "GOOG"]


def get_biotech_tickers():
    """Fetch or define top biotech tickers."""
    try:
        table = safe_html_read("https://en.wikipedia.org/wiki/NASDAQ_Biotechnology_Index")
        if table and len(table) > 1:
            df = table[1]
        else:
            df = table[0]
        tickers = df["Ticker"].dropna().tolist()
        logging.info("Loaded %d biotech tickers.", len(tickers))
        return tickers
    except Exception as e:
        logging.error("Error loading biotech list: %s", e)
        # fallback static list
        return [
            "AMGN","BIIB","VRTX","REGN","GILD","MRNA","NBIX","EXEL","INCY","SGEN",
            "TECH","BGNE","ILMN","CRSP","NTLA","BMRN","XLRN","ALNY","RPRX","NBIX",
            "HALO","IONS","ARWR","SRPT","NKTR","ACAD","ARGX","BNTX","NVCR","VERV"
        ]


def build_symbol_list():
    sp500 = get_sp500_tickers()
    biotech = get_biotech_tickers()
    all_syms = sorted(set(sp500 + biotech))
    logging.info("Total tickers to scan: %d", len(all_syms))
    return all_syms


SYMBOLS = build_symbol_list()


# === NOTIFICATIONS ===
def send_alert(msg: str):
    if DISCORD_WEBHOOK and DISCORD_WEBHOOK != "PASTE_YOUR_DISCORD_WEBHOOK_HERE":
        try:
            requests.post(DISCORD_WEBHOOK, json={"content": msg})
            logging.info("✅ Sent alert to Discord.")
        except Exception as e:
            logging.error("❌ Error sending alert: %s", e)
    else:
        logging.warning("⚠️ No webhook URL configured — alert not sent.")


# === SCANNER CORE ===
def scan_once():
    tz = pytz.timezone(TIMEZONE)
    now = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M")
    signals = []

    for sym in SYMBOLS:
        try:
            df = yf.download(sym, period="60d", interval=TIMEFRAME,
                             progress=False, auto_adjust=False)
            if len(df) < 2:
                continue

            df["ema_fast"] = df["C]()_]()

