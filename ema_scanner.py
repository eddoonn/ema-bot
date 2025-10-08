import yfinance as yf
import pandas as pd
import requests
import time
import datetime
import pytz
import logging
from io import StringIO

# === CONFIG ===
EMA_FAST = 13          # fast EMA (daily)
EMA_SLOW = 21          # slow EMA (daily)
EMA_TREND = 200        # trend EMA (4H)
TIMEFRAME = "1d"       # scan timeframe
CHECK_INTERVAL = 900   # 900 sec = 15 min between scans
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1425616478601871400/AMbiCffNSI7lOsqLPBZ5UDPOStNW0UgcAJAqMU0D1QxDzD2EymlnrbTQxN44XErNkaXm"
TEST_ALERT_ON_START = False

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ======================================================================
#                         TICKER SOURCES
# ======================================================================

def get_sp500_tickers():
    """Load S&P 500 tickers from Wikipedia with browser headers and fallback."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
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
        # Minimal reliable fallback set
        return [
            "AAPL", "MSFT", "AMZN", "GOOG", "META", "NVDA", "TSLA", "BRK.B", "JPM", "V",
            "PG", "JNJ", "XOM", "HD", "MA", "UNH", "KO", "PEP", "DIS", "NFLX",
            "CSCO", "BAC", "PFE", "ADBE", "AVGO", "INTC", "VZ", "CRM", "ABT", "COST"
        ]

def get_biotech_tickers():
    """Fetch biotech tickers from Wikipedia with headers and fallback."""
    url = "https://en.wikipedia.org/wiki/List_of_biotechnology_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) "
            "Version/17.0 Safari/605.1.15"
        )
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
    # Reliable fallback
    fallback = [
        "BIIB","REGN","VRTX","GILD","ILMN","MRNA","BMRN","ALNY","EXEL","NBIX",
        "INCY","SRPT","RPRX","IONS","TECH","CYT","CRSP","DXCM","VIR","AMGN",
        "MDGL","TWST","BEAM","NTLA","HALO","ACAD","SGMO","BLUE","ARWR","SGEN"
    ]
    logging.info(f"Using fallback biotech list of {len(fallback)} tickers.")
    return fallback

def get_all_tickers():
    tickers = list(set(get_sp500_tickers() + get_biotech_tickers()))
    logging.info(f"Total tickers to scan: {len(tickers)}")
    return tickers

# ======================================================================
#                         ALERT HELPERS
# ======================================================================

def notify_discord(message: str):
    """Send message to Discord webhook."""
    if not DISCORD_WEBHOOK:
        logging.warning("⚠️ No webhook URL configured — alert not sent.")
        return
    try:
        requests.post(DISCORD_WEBHOOK, json={"content": message})
    except Exception as e:
        logging.error(f"Error sending to Discord: {e}")

# ======================================================================
#                         CORE SCANNER
# ======================================================================

def scan_symbol(sym: str):
    """Scan a single ticker for EMA crossovers with 4H trend filter."""
    try:
        # --- DAILY DATA ---
        df_d = yf.download(sym, period="200d", interval="1d", progress=False, auto_adjust=False)
        if df_d.empty or len(df_d) < EMA_SLOW + 2:
            return None

        df_d["ema_fast"] = df_d["Close"].ewm(span=EMA_FAST, adjust=False).mean()
        df_d["ema_slow"] = df_d["Close"].ewm(span=EMA_SLOW, adjust=False).mean()

        # --- Confirmed crossover (TradingView-style, wait 1 bar) ---
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

        # --- 4H TREND FILTER ---
        df_4h = yf.download(sym, period="200d", interval="4h", progress=False, auto_adjust=False)
        if df_4h.empty:
            return None
        df_4h["ema_trend"] = df_4h["Close"].ewm(span=EMA_TREND, adjust=False).mean()
        ema_4h_last = float(df_4h["ema_trend"].iloc[-1])
        last_close = float(df_d["Close"].iloc[-1])

        trend_up = last_close > ema_4h_last
        trend_dn = last_close < ema_4h_last

        if cross_up and trend_up:
            return f"📈 {sym} BUY @ {last_close:.2f} (Daily EMA13>EMA21, above 4H EMA200)"
        elif cross_dn and trend_dn:
            return f"🔻 {sym} SELL @ {last_close:.2f} (Daily EMA13<EMA21, below 4H EMA200)"
        else:
            return None

    except Exception as e:
        logging.error(f"Error scanning {sym}: {e}")
        return None

# ======================================================================
#                         MAIN LOOP
# ======================================================================

def main():
    tickers = get_all_tickers()
    logging.info("🚀 EMA Scanner Bot Started — scanning every 900 seconds")

    if TEST_ALERT_ON_START:
        notify_discord("✅ EMA Scanner is online and ready to scan markets.")

    while True:
        try:
            signals = []
            for sym in tickers:
                result = scan_symbol(sym)
                if result:
                    signals.append(result)

            if signals:
                message = (
                    f"**EMA Cross Alerts — {datetime.datetime.now():%Y-%m-%d %H:%M}**\n"
                    + "\n".join(signals)
                )
                notify_discord(message)
                logging.info(message)
            else:
                logging.info(f"{datetime.datetime.now():%H:%M} — No signals")

            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            logging.error(f"Fatal loop error: {e}")
            time.sleep(60)

# ======================================================================
#                         ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()
