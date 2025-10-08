import yfinance as yf
import pandas as pd
import requests
import time
import datetime
import pytz
import logging
from io import StringIO

# === CONFIG ===
EMA_FAST = 13
EMA_SLOW = 21
EMA_TREND = 200
TIMEFRAME = "1d"        # daily
CHECK_INTERVAL = 900     # 15 min
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1425616478601871400/AMbiCffNSI7lOsqLPBZ5UDPOStNW0UgcAJAqMU0D1QxDzD2EymlnrbTQxN44XErNkaXm"
TEST_ALERT_ON_START = False

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# === Functions ===
def get_sp500_tickers():
    """Load S&P 500 tickers from Wikipedia (fallback to local list if blocked)."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
        tickers = tables[0]["Symbol"].tolist()
        logging.info(f"Loaded {len(tickers)} S&P 500 tickers from Wikipedia.")
        return tickers
    except Exception as e:
        logging.warning(f"Error loading S&P 500 list: {e}")
        # fallback sample
        return ["AAPL", "MSFT", "AMZN", "NVDA", "GOOG", "META", "TSLA", "AMD", "NFLX", "INTC"]

def get_biotech_tickers():
    """Fetch biotech tickers from NASDAQ screener or fallback list."""
    try:
        html = pd.read_html("https://en.wikipedia.org/wiki/List_of_biotechnology_companies")
        all_symbols = []
        for df in html:
            if "Ticker" in df.columns:
                all_symbols += df["Ticker"].dropna().tolist()
        all_symbols = [s for s in all_symbols if isinstance(s, str) and s.isalpha()]
        if all_symbols:
            logging.info(f"Loaded {len(all_symbols)} biotech tickers from Wikipedia.")
            return all_symbols
    except Exception as e:
        logging.warning(f"Error loading biotech list: {e}")
    # fallback
    fallback = [
        "BIIB","REGN","VRTX","GILD","ILMN","MRNA","BMRN","ALNY","EXEL","NBIX",
        "INCY","SRPT","RPRX","IONS","TECH","CYT","CRSP","SGEN","DXCM","VIR",
        "AMGN","MDGL","TWST","BEAM","NTLA","HALO","ACAD","SGMO","BLUE","ARWR"
    ]
    logging.info(f"Using fallback biotech list of {len(fallback)} tickers.")
    return fallback

def get_all_tickers():
    tickers = list(set(get_sp500_tickers() + get_biotech_tickers()))
    logging.info(f"Total tickers to scan: {len(tickers)}")
    return tickers

def notify_discord(message: str):
    """Send message to Discord webhook."""
    if not DISCORD_WEBHOOK:
        logging.warning("⚠️ No webhook URL configured — alert not sent.")
        return
    try:
        requests.post(DISCORD_WEBHOOK, json={"content": message})
    except Exception as e:
        logging.error(f"Error sending to Discord: {e}")

def scan_symbol(sym):
    """Scan a single ticker for EMA crossovers with 4H trend filter."""
    try:
        # === DAILY DATA ===
        df_d = yf.download(sym, period="200d", interval="1d", progress=False)
        if df_d.empty or len(df_d) < EMA_SLOW + 2:
            return None

        df_d["ema_fast"] = df_d["Close"].ewm(span=EMA_FAST).mean()
        df_d["ema_slow"] = df_d["Close"].ewm(span=EMA_SLOW).mean()

        # --- TradingView-style confirmed crossover (Option B) ---
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

        # === 4H TREND FILTER ===
        df_4h = yf.download(sym, period="200d", interval="4h", progress=False)
        if df_4h.empty:
            return None
        df_4h["ema_trend"] = df_4h["Close"].ewm(span=EMA_TREND).mean()
        ema_4h_last = float(df_4h["ema_trend"].iloc[-1])
        last_close = float(df_d["Close"].iloc[-1])

        # === FILTERS ===
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

# === Main loop ===
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
                message = f"**EMA Cross Alerts — {datetime.datetime.now():%Y-%m-%d %H:%M}**\n" + "\n".join(signals)
                notify_discord(message)
                logging.info(message)
            else:
                logging.info(f"{datetime.datetime.now():%H:%M} — No signals")

            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            logging.error(f"Fatal loop error: {e}")
            time.sleep(60)

# === Run ===
if __name__ == "__main__":
    main()
