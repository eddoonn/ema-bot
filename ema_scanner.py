import yfinance as yf
import pandas as pd
import time
import requests
import datetime
import pytz
import logging

# === CONFIG ===
EMA_FAST = 13
EMA_SLOW = 21
TIMEFRAME = "1h"           # "1h", "4h", "1d"
CHECK_INTERVAL = 15 * 60   # seconds
TIMEZONE = "America/New_York"
DISCORD_WEBHOOK = "PASTE_YOUR_DISCORD_WEBHOOK_HERE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# === FETCH SYMBOL LISTS ===
def get_sp500_tickers():
    """Scrape S&P 500 tickers from Wikipedia."""
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = table[0]
        tickers = df["Symbol"].tolist()
        logging.info("Loaded %d S&P 500 tickers.", len(tickers))
        return tickers
    except Exception as e:
        logging.error("Error loading S&P 500 list: %s", e)
        # fallback minimal list
        return ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "GOOG"]


def get_biotech_tickers():
    """Fetch or define top 100 biotech tickers."""
    try:
        # Example source (NASDAQ Biotech Index)
        table = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ_Biotechnology_Index")
        df = table[1] if len(table) > 1 else table[0]
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


# === COMBINE SYMBOL LISTS ===
def build_symbol_list():
    sp500 = get_sp500_tickers()
    biotech = get_biotech_tickers()
    # combine and remove duplicates
    all_syms = sorted(set(sp500 + biotech))
    logging.info("Total tickers to scan: %d", len(all_syms))
    return all_syms


SYMBOLS = build_symbol_list()


# === DISCORD NOTIFICATION ===
def send_alert(msg: str):
    if DISCORD_WEBHOOK and DISCORD_WEBHOOK != "PASTE_YOUR_DISCORD_WEBHOOK_HERE":
        try:
            requests.post(DISCORD_WEBHOOK, json={"content": msg})
            logging.info("✅ Sent alert to Discord.")
        except Exception as e:
            logging.error("❌ Error sending alert: %s", e)
    else:
        logging.warning("⚠️ No webhook URL configured — alert not sent.")


# === CORE SCANNER ===
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

            df["ema_fast"] = df["Close"].ewm(span=EMA_FAST).mean()
            df["ema_slow"] = df["Close"].ewm(span=EMA_SLOW).mean()

            prev_fast, prev_slow = df["ema_fast"].iloc[-2], df["ema_slow"].iloc[-2]
            last_fast, last_slow = df["ema_fast"].iloc[-1], df["ema_slow"].iloc[-1]

            cross_up = prev_fast < prev_slow and last_fast > last_slow
            cross_dn = prev_fast > prev_slow and last_fast < last_slow

            if cross_up:
                signals.append(f"📈 {sym} BUY @ {df['Close'].iloc[-1]:.2f}")
            elif cross_dn:
                signals.append(f"🔻 {sym} SELL @ {df['Close'].iloc[-1]:.2f}")

        except Exception as e:
            logging.error("Error scanning %s: %s", sym, e)

    if signals:
        msg = f"**EMA Cross Alerts — {now}**\n" + "\n".join(signals)
        send_alert(msg)
    else:
        logging.info("%s — No signals", now)


# === MAIN LOOP ===
if __name__ == "__main__":
    logging.info("🚀 EMA Scanner Bot Started — scanning every %d seconds", CHECK_INTERVAL)
    while True:
        scan_once()
        time.sleep(CHECK_INTERVAL)

