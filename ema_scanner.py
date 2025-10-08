import yfinance as yf
import pandas as pd
import time, requests, datetime, pytz, logging, io

# -------------------- CONFIG --------------------
EMA_FAST = 13
EMA_SLOW = 21
EMA_TREND = 200

TIMEFRAME_SIGNAL = "1d"     # Daily signals
TIMEFRAME_TREND  = "4h"     # 4H trend filter
CHECK_INTERVAL   = 15 * 60  # every 15 minutes
TIMEZONE         = "America/New_York"

DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1425616478601871400/AMbiCffNSI7lOsqLPBZ5UDPOStNW0UgcAJAqMU0D1QxDzD2EymlnrbTQxN44XErNkaXm"

# -------------------- LOGGING --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
tz = pytz.timezone(TIMEZONE)

# -------------------- DISCORD --------------------
def send_alert(message: str):
    """Send message to Discord webhook."""
    try:
        if not DISCORD_WEBHOOK:
            logging.warning("⚠️ No webhook URL configured — alert not sent.")
            return
        r = requests.post(DISCORD_WEBHOOK, json={"content": message})
        if r.status_code == 204:
            logging.info("✅ Sent alert to Discord.")
        else:
            logging.warning(f"⚠️ Discord response {r.status_code}: {r.text}")
    except Exception as e:
        logging.error(f"Error sending alert: {e}")

# -------------------- LOAD TICKERS --------------------
def load_tickers():
    """Load S&P500 from Wikipedia and add top biotech tickers."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        sp500 = tables[0]["Symbol"].tolist()
        logging.info(f"✅ Loaded {len(sp500)} S&P 500 tickers.")
    except Exception as e:
        logging.error(f"⚠️ Could not load S&P500 list: {e}")
        sp500 = []

    biotech = [
        "AMGN","REGN","VRTX","GILD","BIIB","ILMN","NBIX","EXEL","ALNY","INCY","SGEN",
        "CRSP","BGNE","BMRN","NBIX","KRTX","XLRN","MRNA","NTLA","BEAM","EDIT","SRPT",
        "RNA","TMDX","CYT","ARWR","VERV","TWST","EVGN","DNLI"
    ]
    logging.info(f"✅ Added {len(biotech)} biotech tickers.")

    tickers = list(set(sp500 + biotech))
    logging.info(f"Total tickers to scan: {len(tickers)}")
    return tickers

SYMBOLS = load_tickers()

# -------------------- DATA HELPERS --------------------
def get_data(symbol: str, period: str = "90d", interval: str = "1d"):
    """Download historical data for a symbol."""
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError("No data returned")
    return df

# -------------------- CORE SCAN LOGIC --------------------
def scan_symbol(sym: str):
    """Check one symbol for EMA crossover signals with 4H trend filter."""
    df_d = get_data(sym, period="120d", interval=TIMEFRAME_SIGNAL)
    df_4h = get_data(sym, period="60d", interval=TIMEFRAME_TREND)

    # --- Compute EMAs ---
    df_d["ema_fast"] = df_d["Close"].ewm(span=EMA_FAST).mean()
    df_d["ema_slow"] = df_d["Close"].ewm(span=EMA_SLOW).mean()
    df_4h["ema_trend"] = df_4h["Close"].ewm(span=EMA_TREND).mean()

    # --- Crossovers on daily timeframe ---
    prev_fast = df_d["ema_fast"].iloc[-2]
    prev_slow = df_d["ema_slow"].iloc[-2]
    last_fast = df_d["ema_fast"].iloc[-1]
    last_slow = df_d["ema_slow"].iloc[-1]
    last_close = float(df_d["Close"].iloc[-1])

    cross_up  = prev_fast < prev_slow and last_fast > last_slow
    cross_dn  = prev_fast > prev_slow and last_fast < last_slow

    # --- Trend filter from 4H timeframe ---
    ema_trend_4h = float(df_4h["ema_trend"].iloc[-1])
    trend_up = last_close > ema_trend_4h
    trend_dn = last_close < ema_trend_4h

    # --- Combine signals with filter ---
    signal = None
    if cross_up and trend_up:
        signal = f"📈 {sym} BUY @ {last_close:.2f} (Daily EMA13>EMA21, above 4H EMA200)"
    elif cross_dn and trend_dn:
        signal = f"🔻 {sym} SELL @ {last_close:.2f} (Daily EMA13<EMA21, below 4H EMA200)"

    return signal

# -------------------- MAIN LOOP --------------------
def run_scanner():
    logging.info(f"🚀 EMA Scanner Bot Started — scanning every {CHECK_INTERVAL//60} minutes")
    last_daily_scan_date = None

    while True:
        try:
            now = datetime.datetime.now(tz)

            # only run once per completed daily candle
            if last_daily_scan_date == now.date():
                time.sleep(CHECK_INTERVAL)
                continue

            signals = []
            for sym in SYMBOLS:
                try:
                    sig = scan_symbol(sym)
                    if sig:
                        signals.append(sig)
                except Exception as e:
                    logging.error(f"Error scanning {sym}: {e}")

            if signals:
                msg = f"**EMA Cross Alerts — {now.strftime('%Y-%m-%d %H:%M')}**\n" + "\n".join(signals)
                send_alert(msg)
            else:
                logging.info(f"{now.strftime('%H:%M')} — No signals today")

            last_daily_scan_date = now.date()
            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            logging.error(f"Main loop error: {e}")
            time.sleep(60)

# -------------------- START --------------------
if __name__ == "__main__":
    run_scanner()
