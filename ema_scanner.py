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

# Optional: send a one-time test message at startup
TEST_ALERT_ON_START = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# === HELPERS ===
def safe_html_read(url: str):
    """Load HTML with a browser-like user agent to avoid 403, return list[tables] or None."""
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req) as r:
            html = r.read().decode("utf-8", errors="ignore")
        return pd.read_html(html)
    except Exception as e:
        logging.error("Error fetching %s: %s", url, e)
        return None


def _normalize_for_yahoo(tickers):
    """Yahoo uses '-' where many lists use '.' (e.g., BRK.B -> BRK-B)."""
    out = []
    for t in tickers:
        t = str(t).strip()
        if not t:
            continue
        out.append(t.replace(".", "-"))
    return out


def get_sp500_tickers():
    """Scrape S&P 500 tickers from Wikipedia; fallback to a public CSV."""
    # Primary: Wikipedia
    tables = safe_html_read("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    if tables:
        try:
            df = tables[0]
            if "Symbol" in df.columns:
                syms = _normalize_for_yahoo(df["Symbol"].tolist())
                logging.info("Loaded %d S&P 500 tickers from Wikipedia.", len(syms))
                return syms
        except Exception as e:
            logging.error("Parse S&P 500 wiki failed: %s", e)

    # Fallback: GitHub dataset CSV
    try:
        csv_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        df = pd.read_csv(csv_url)
        syms = _normalize_for_yahoo(df["Symbol"].tolist())
        logging.info("Loaded %d S&P 500 tickers from GitHub CSV.", len(syms))
        return syms
    except Exception as e:
        logging.error("Fallback S&P list failed: %s", e)
        # final minimal fallback
        return ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "GOOG"]


def get_biotech_tickers():
    """Try NASDAQ Biotech Index page; fallback to a static short list."""
    tables = safe_html_read("https://en.wikipedia.org/wiki/NASDAQ_Biotechnology_Index")
    if tables:
        # Find a table that contains a 'Ticker' or 'Symbol' column
        for df in tables:
            cols = {c.lower(): c for c in df.columns}
            if "ticker" in cols:
                syms = df[cols["ticker"]].dropna().tolist()
                syms = _normalize_for_yahoo(syms)
                logging.info("Loaded %d biotech tickers from Wikipedia.", len(syms))
                return syms
            if "symbol" in cols:
                syms = df[cols["symbol"]].dropna().tolist()
                syms = _normalize_for_yahoo(syms)
                logging.info("Loaded %d biotech tickers from Wikipedia.", len(syms))
                return syms

    # Fallback static set (representative, not exhaustive)
    fallback = [
        "AMGN","BIIB","VRTX","REGN","GILD","MRNA","NBIX","EXEL","INCY","SGEN",
        "TECH","BGNE","ILMN","CRSP","NTLA","BMRN","XLRN","ALNY","RPRX","HALO",
        "IONS","ARWR","SRPT","ACAD","ARGX","BNTX","NVCR","VERV","REGN","VRTX"
    ]
    logging.warning("Using fallback biotech list of %d tickers.", len(fallback))
    return fallback


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
            requests.post(DISCORD_WEBHOOK, json={"content": msg}, timeout=10)
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
            df = yf.download(
                sym,
                period="60d",
                interval=TIMEFRAME,
                progress=False,
                auto_adjust=False,
            )
            if df is None or len(df) < 2:
                continue

            # Calculate EMAs
            df["ema_fast"] = df["Close"].ewm(span=EMA_FAST).mean()
            df["ema_slow"] = df["Close"].ewm(span=EMA_SLOW).mean()

            # Use scalar comparisons (avoid Series ambiguity)
            prev_fast, prev_slow = float(df["ema_fast"].iloc[-2]), float(df["ema_slow"].iloc[-2])
            last_fast, last_slow = float(df["ema_fast"].iloc[-1]), float(df["ema_slow"].iloc[-1])
            last_close = float(df["Close"].iloc[-1])

            cross_up = (prev_fast < prev_slow) and (last_fast > last_slow)
            cross_dn = (prev_fast > prev_slow) and (last_fast < last_slow)

            if cross_up:
                signals.append(f"📈 {sym} BUY @ {last_close:.2f}")
            elif cross_dn:
                signals.append(f"🔻 {sym} SELL @ {last_close:.2f}")

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
    if TEST_ALERT_ON_START:
        send_alert("✅ Startup test: Discord webhook is working.")
    while True:
        scan_once()
        time.sleep(CHECK_INTERVAL)

