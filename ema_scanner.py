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
        sym = row["symbol"]
        entry_price = row["price"]
        signal_type = row["signal"]
        entry_date = row["date"]
        try:
            df_hist = yf.download(sym, start=entry_date, period="10d", interval="1d", progress=False)
            if len(df_hist) < HOLD_DAYS:
                continue
            exit_price = df_hist["Close"].iloc[HOLD_DAYS - 1]
            ret = (exit_price - entry_price) / entry_price
            if signal_type == "SELL":
                ret = -ret
            profit = ret * CAPITAL_PER_TRADE
            results.append((sym, signal_type, entry_price, exit_price, profit))
        except Exception as e:
            logging.error(f"Error evaluating {sym}: {e}")

    if not results:
        return None

    total = sum(p[4] for p in results)
    winrate = sum(1 for p in results if p[4] > 0) / len(results) * 100
    avg_trade = total / len(results)

    msg = "**📊 Weekly Performance Report**\n"
    for sym, side, entry, exitp, prof in results:
        emoji = "✅" if prof > 0 else "❌"
        msg += f"{emoji} {sym} {side} from {entry:.2f} → {exitp:.2f} | {prof:+.2f} USD\n"
    msg += f"\n**Total:** {total:+.2f} USD | **Winrate:** {winrate:.1f}% | **Avg:** {avg_trade:+.2f} USD/trade"

    # remove evaluated rows so each is only reported once
    df = df[~df["symbol"].isin([r[0] for r in results])]
    df.to_csv(LOG_FILE, index=False)

    return msg


