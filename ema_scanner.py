# ============================================================
#             EMA Scanner Bot — Render Worker Edition
# ============================================================
# Scans S&P 500 + Top Biotech stocks
# Sends Discord alerts for confirmed EMA (13/21) crossovers
# Uses 4H EMA 200 trend filter + daily bar confirmation
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

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1425616478601871400/AMbiCffNSI7lOsqLPBZ5UDPOStNW0UgcAJAqMU0D1QxDzD2EymlnrbTQxN44XErNkaXm"

EMA_FAST = 13
EMA_SLOW = 21
EMA_TREND = 200

TIMEFRAME_DAILY = "1d"
TIMEFRAME_4H = "4h"
CHECK_INTERVAL = 900  # seconds (15 min)
HOLD_DAYS = 5
CAPITAL_PER_TRADE = 500
LOG_FILE = "trades_log.csv"

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M")

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

def get_sp500_tickers():
    """Fetch S&P 500 tickers from Wikipedia (fallback to static list)."""
    try:
        import urllib.request
        html = urllib.request.urlopen("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies").read()
        tables = pd.read_html(StringIO(html.decode()))
        df = tables[0]
        tickers = df["Symbol"].tolist()
        logging.info(f"Loaded {len(tickers)} S&P 500 tickers from Wikipedia.")
        return tickers
    except Exception as e:
        logging.error(f"Error loading S&P 500 list: {e}")
        # fallback minimal list
        return ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOG", "META"]

def get_biotech_tickers():
    """Fetch top biotech tickers (fallback static)."""
    try:
        bio = pd.read_html("https://en.wikipedia.org/wiki/List_of_biotechnology_companies")[0]
        tickers = bio.iloc[:,0].dropna().tolist()[:100]
        logging.info(f"Loaded {len(tickers)} biotech tickers from Wikipedia.")
        return tickers
    except Exception as e:
        logging.error(f"Error loading biotech list: {e}")
        return ["BIIB", "REGN", "VRTX", "GILD", "ALNY", "ILMN", "CRSP", "EXEL", "BMRN", "SGEN",
                "NBIX", "TECH", "HALO", "ARGX", "RGEN", "UTHR", "IONS", "GMAB", "INCY", "BLUE",
                "SRPT", "ACAD", "PBYI", "PRTA", "MDGL", "MRTX", "CYT", "AGEN", "VCYT", "EVGN"]

# ----------------------------------------------------------------------
# Core EMA Scanner
# ----------------------------------------------------------------------

def fetch_emas(symbol):
    """Return EMAs (13, 21 daily; 200 on 4h) and latest Close."""
    df_daily = yf.download(symbol, period="60d", interval=TIMEFRAME_DAILY, progress=False)
    df_4h    = yf.download(symbol, period="180d", interval=TIMEFRAME_4H, progress=False)

    if len(df_daily) < 21 or len(df_4h) < 200:
        raise ValueError("Insufficient data")

    df_daily["ema_fast"] = df_daily["Close"].ewm(span=EMA_FAST).mean()
    df_daily["ema_slow"] = df_daily["Close"].ewm(span=EMA_SLOW).mean()
    df_4h["ema_trend"]   = df_4h["Close"].ewm(span=EMA_TREND).mean()

    last_close = float(df_daily["Close"].iloc[-1])
    ema_fast   = float(df_daily["ema_fast"].iloc[-1])
    ema_slow   = float(df_daily["ema_slow"].iloc[-1])
    ema_trend  = float(df_4h["ema_trend"].iloc[-1])
    prev_fast  = float(df_daily["ema_fast"].iloc[-2])
    prev_slow  = float(df_daily["ema_slow"].iloc[-2])

    return prev_fast, prev_slow, ema_fast, ema_slow, ema_trend, last_close

def scan_tickers(tickers):
    """Check all tickers for EMA crossovers with trend filter."""
    signals = []
    for sym in tickers:
        try:
            prev_fast, prev_slow, ema_fast, ema_slow, ema_trend, close = fetch_emas(sym)

            crossed_up   = prev_fast < prev_slow and ema_fast > ema_slow
            crossed_down = prev_fast > prev_slow and ema_fast < ema_slow
            is_above_trend = close > ema_trend
            is_below_trend = close < ema_trend

            if crossed_up and is_above_trend:
                signals.append(f"📈 {sym} BUY @ {close:.2f} (Daily EMA13>EMA21 & above 4H EMA200)")
                record_signal("BUY", sym, close)
            elif crossed_down and is_below_trend:
                signals.append(f"🔻 {sym} SELL @ {close:.2f} (Daily EMA13<EMA21 & below 4H EMA200)")
                record_signal("SELL", sym, close)

        except Exception as e:
            logging.error(f"Error scanning {sym}: {e}")
    return signals

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
    """After HOLD_DAYS, evaluate past trades."""
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
            if side == "SELL": ret = -ret
            profit = ret * CAPITAL_PER_TRADE
            results.append((sym, side, entry, exit_price, profit))
        except Exception as e:
            logging.error(f"Error evaluating {sym}: {e}")

    if not results: return None

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
    sp500 = get_sp500_tickers()
    bio = get_biotech_tickers()
    tickers = sorted(set(sp500 + bio))
    logging.info(f"Total tickers to scan: {len(tickers)}")
    logging.info("🚀 EMA Scanner Bot Started — scanning every 900 seconds")

    while True:
        signals = scan_tickers(tickers)
        if signals:
            msg = f"**EMA Cross Alerts — {datetime.datetime.now():%Y-%m-%d %H:%M}**\n" + "\n".join(signals)
            send_discord_message(msg)
        else:
            logging.info(f"{datetime.datetime.now():%H:%M} — No signals")

        # periodic performance check
        report = evaluate_old_signals()
        if report:
            send_discord_message(report)

        time.sleep(CHECK_INTERVAL)

# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
