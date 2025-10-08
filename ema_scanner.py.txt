import yfinance as yf
import pandas as pd
import time
import requests
import datetime
import pytz
import logging

# === CONFIG ===
SYMBOLS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "GOOG"]
EMA_FAST = 13
EMA_SLOW = 21
TIMEFRAME = "1h"           # "1h", "4h", "1d"
CHECK_INTERVAL = 15 * 60   # seconds
TIMEZONE = "America/New_York"

# --- NOTIFICATION ---
DISCORD_WEBHOOK = "PASTE_YOUR_DISCORD_WEBHOOK_HERE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def send_alert(msg: str):
    if DISCORD_WEBHOOK:
        requests.post(DISCORD_WEBHOOK, json={"content": msg})
    logging.info("ALERT SENT: %s", msg)

def scan_once():
    tz = pytz.timezone(TIMEZONE)
    now = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M")
    signals = []

    for sym in SYMBOLS:
        try:
            df = yf.download(sym, period="60d", interval=TIMEFRAME, progress=False)
            if len(df) < 2: continue
            df["ema_fast"] = df["Close"].ewm(span=EMA_FAST).mean()
            df["ema_slow"] = df["Close"].ewm(span=EMA_SLOW).mean()

            prev, last = df.iloc[-2], df.iloc[-1]
            cross_up = prev.ema_fast < prev.ema_slow and last.ema_fast > last.ema_slow
            cross_dn = prev.ema_fast > prev.ema_slow and last.ema_fast < last.ema_slow

            if cross_up:
                signals.append(f"📈 {sym} BUY @ {last.Close:.2f}")
            elif cross_dn:
                signals.append(f"🔻 {sym} SELL @ {last.Close:.2f}")

        except Exception as e:
            logging.error("Error scanning %s: %s", sym, e)

    if signals:
        msg = f"**EMA Cross Alerts — {now}**\\n" + "\\n".join(signals)
        send_alert(msg)
    else:
        logging.info("%s — No signals", now)

if __name__ == "__main__":
    while True:
        scan_once()
        time.sleep(CHECK_INTERVAL)
