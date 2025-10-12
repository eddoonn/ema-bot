# ============================================================
#         EMA Multi-Factor Scanner Bot — Pro Edition
# ============================================================
# Scans S&P500 + NASDAQ100 + Dow30 + Biotech + Sector ETFs
# Sends Discord alerts for confirmed EMA(13/21) crossovers
# Uses 4H EMA200 trend filter + RSI/ADX/OBV/ATR confirmations
# Includes 5-day backtest performance summary
# Batched downloads to reduce Yahoo rate limits
# ============================================================

import os
import re
import time
import math
import json
import random
import logging
import datetime
import traceback
import urllib.request
from threading import Thread

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import ta

from flask import Flask, jsonify
from pandas.api.types import is_numeric_dtype

# ----------------------------------------------------------------------
# Configuration (override via env vars on Render)
# ----------------------------------------------------------------------

DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "https://discord.com/api/webhooks/1425616478601871400/AMbiCffNSI7lOsqLPBZ5UDPOStNW0UgcAJAqMU0D1QxDzD2EymlnrbTQxN44XErNkaXm")

EMA_FAST  = int(os.getenv("EMA_FAST", 13))
EMA_SLOW  = int(os.getenv("EMA_SLOW", 21))
EMA_TREND = int(os.getenv("EMA_TREND", 200))

TIMEFRAME_DAILY = os.getenv("TIMEFRAME_DAILY", "1d")
TIMEFRAME_4H    = os.getenv("TIMEFRAME_4H", "4h")

CHECK_INTERVAL     = int(os.getenv("CHECK_INTERVAL", 120))       # sleep between loops
HOLD_DAYS          = int(os.getenv("HOLD_DAYS", 5))
CAPITAL_PER_TRADE  = float(os.getenv("CAPITAL_PER_TRADE", 500))
LOG_FILE           = os.getenv("LOG_FILE", "trades_log.csv")

# ---- Confirmation scoring tunables (for "any 2 of 3") ----
# You can override these in Render → Environment
RSI_MIN_BUY   = float(os.getenv("RSI_MIN_BUY", 50))    # default 50 (was 55)
RSI_MAX_SELL  = float(os.getenv("RSI_MAX_SELL", 50))   # default 50 (was 45; 48–50 is common)
ADX_MIN       = float(os.getenv("ADX_MIN", 15))        # default 15 (was 20)
ATR_RATIO_MIN = float(os.getenv("ATR_RATIO_MIN", 0.7)) # default 0.7 (was 0.8)

# Allow a small buffer around the 4h EMA200 trend
TREND_BUF     = float(os.getenv("TREND_BUF", 0.99))    # 0.99 ≈ allow ~1% below for buys

# Require "any N of 3" confirmations (RSI, ADX, ATR)
CONFIRM_SCORE_BUY  = int(os.getenv("CONFIRM_SCORE_BUY", 2))
CONFIRM_SCORE_SELL = int(os.getenv("CONFIRM_SCORE_SELL", 2))

# Optional: count OBV as an extra vote (False by default)
USE_OBV = os.getenv("USE_OBV", "0") == "1"


# scanning cadence
SCAN_BATCH_SIZE    = int(os.getenv("SCAN_BATCH_SIZE", 150))       # tickers per rotating batch
BATCHES_PER_LOOP   = int(os.getenv("BATCHES_PER_LOOP", 1))        # how many batches per loop
BATCH_PAUSE        = float(os.getenv("BATCH_PAUSE", 3.0))         # seconds pause between batches in a loop

# rate-limit / resiliency
MAX_RETRIES        = int(os.getenv("MAX_RETRIES", 3))
BACKOFF_BASE       = float(os.getenv("BACKOFF_BASE", 1.7))
BACKOFF_JITTER_MAX = float(os.getenv("BACKOFF_JITTER_MAX", 0.35))
RATE_LIMIT_DELAY   = float(os.getenv("RATE_LIMIT_DELAY", 0.0))    # small pause after each chunk request

# yfinance chunking inside a batch
YF_BATCH_CHUNK     = int(os.getenv("YF_BATCH_CHUNK", 50))         # 40–60 recommended

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M"
)
logger = logging.getLogger(__name__)

# quiet yfinance’s own "Failed download" lines
logging.getLogger("yfinance").setLevel(logging.ERROR)

# ----------------------------------------------------------------------
# Globals for health reporting
# ----------------------------------------------------------------------

LAST_SCAN_SUMMARY = {
    "universe_size": 0,
    "last_batch_start": None,
    "last_batch_size": 0,
    "signals_in_batch": 0,
    "coverage_pct": 0.0,
    "last_error": None,
}

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def send_discord_message(content: str):
    if (not DISCORD_WEBHOOK) or ("REPLACE_WITH_YOUR_WEBHOOK" in DISCORD_WEBHOOK):
        logging.warning("⚠️ No valid webhook URL configured — alert not sent.")
        return
    try:
        resp = requests.post(DISCORD_WEBHOOK, json={"content": content}, timeout=10)
        if resp.status_code >= 300:
            logging.error(f"❌ Discord send failed: {resp.status_code} {resp.text}")
        else:
            logging.info("✅ Discord alert sent.")
    except Exception as e:
        logging.error(f"❌ Discord send failed: {e}")

def to_float(x):
    try:
        if isinstance(x, (list, np.ndarray)) and len(x) > 0:
            x = x[0]
        return float(x)
    except Exception:
        return np.nan

def _normalize_ticker(x):
    if x is None:
        return None
    s = str(x).strip().upper()
    if not s:
        return None
    s = s.split()[0]
    s = s.replace(".", "-")  # BRK.B -> BRK-B for Yahoo
    if not re.fullmatch(r"[A-Z\-]+", s):
        return None
    if len(s) > 12:
        return None
    return s

def normalize_tickers(seq):
    out = []
    for x in seq:
        s = _normalize_ticker(x)
        if s:
            out.append(s)
    return out

def safe_read_html(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as resp:
        html = resp.read()
    return pd.read_html(html)

def _sleep_backoff(attempt: int):
    wait = (BACKOFF_BASE ** attempt) + random.random() * BACKOFF_JITTER_MAX
    time.sleep(wait)

# ----------------------------------------------------------------------
# Universe fetchers
# ----------------------------------------------------------------------

def get_sp500_tickers():
    try:
        tables = safe_read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        tickers = df["Symbol"].astype(str).tolist()
        logging.info(f"Loaded {len(tickers)} S&P 500 tickers from Wikipedia.")
        return tickers
    except Exception as e:
        logging.error(f"S&P500 fetch failed: {e}")
        return ["AAPL","MSFT","TSLA","NVDA","AMZN","GOOG","META"]

def get_biotech_tickers():
    try:
        tables = safe_read_html("https://en.wikipedia.org/wiki/List_of_biotechnology_companies")
        bio = tables[0]
        tickers = bio.iloc[:, 0].dropna().astype(str).tolist()[:100]
        logging.info(f"Loaded {len(tickers)} biotech tickers from Wikipedia.")
        return tickers
    except Exception as e:
        logging.error(f"Biotech list error: {e}")
        return ["BIIB","REGN","VRTX","GILD","ALNY","ILMN"]

def get_nasdaq100_tickers():
    try:
        tables = safe_read_html("https://en.wikipedia.org/wiki/NASDAQ-100")
        df = None
        for t in tables:
            if "Ticker" in t.columns:
                df = t
                break
        if df is None and len(tables) > 4:
            df = tables[4]
        if df is None:
            raise ValueError("No NASDAQ-100 table with 'Ticker' column found.")
        tickers = df["Ticker"].dropna().astype(str).tolist()
        logging.info(f"Loaded {len(tickers)} NASDAQ-100 tickers from Wikipedia.")
        return tickers
    except Exception as e:
        logging.error(f"NASDAQ100 fetch failed: {e}")
        return ["AAPL","MSFT","NVDA","META","AMZN","GOOG","TSLA"]

def get_dow30_tickers():
    try:
        tables = safe_read_html("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")
        for t in tables:
            cols = [c.lower() for c in t.columns.astype(str)]
            possible_cols = [c for c in cols if "symbol" in c or "ticker" in c]
            if possible_cols:
                tickers = t[t.columns[cols.index(possible_cols[0])]].dropna().astype(str).tolist()
                logging.info(f"Loaded {len(tickers)} Dow 30 tickers from Wikipedia.")
                return tickers
        raise ValueError("No symbol/ticker column found.")
    except Exception as e:
        logging.error(f"Dow30 fetch failed: {e}")
        return ["AAPL", "MSFT", "CAT", "BA", "JNJ", "PG", "V"]

def get_sector_tickers():
    return ["XLE","XLF","XLK","XLV","XLI","XLY","XLU","XLB","XLC","XBI","SMH","SOXX"]

# ----------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------

def fetch_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
    df["adx"] = adx.adx()
    obv = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"])
    df["obv"] = obv.on_balance_volume()
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14)
    df["atr"] = atr.average_true_range()
    return df

def _to_numeric_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ----------------------------------------------------------------------
# Batch downloads (CHUNKED) to reduce rate limits
# ----------------------------------------------------------------------

def _download_batch_chunked(tickers, *, period, interval, label):
    """
    Download many tickers by splitting into chunks to avoid rate limits.
    Returns dict[ticker] -> DataFrame (or None).
    """
    if not tickers:
        return {}

    chunks = [tickers[i:i+YF_BATCH_CHUNK] for i in range(0, len(tickers), YF_BATCH_CHUNK)]
    merged = {}

    for ci, chunk in enumerate(chunks, 1):
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                df = yf.download(
                    chunk,
                    period=period,
                    interval=interval,
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                    group_by='ticker'
                )
                if isinstance(df, pd.DataFrame) and not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        for t in chunk:
                            if t in df.columns.get_level_values(0):
                                sub = df[t].dropna(how="all")
                                merged[t] = sub if not sub.empty else None
                            else:
                                merged[t] = None
                    else:
                        # Rare case: single wide frame — assign to first symbol
                        merged[chunk[0]] = df
                else:
                    # empty — likely rate-limited
                    raise RuntimeError("Empty DataFrame (possibly rate-limited)")

                if RATE_LIMIT_DELAY > 0:
                    time.sleep(RATE_LIMIT_DELAY)
                break  # chunk done

            except Exception as e:
                last_exc = e
                msg = str(e)
                if any(s in msg for s in ["Rate limited", "Too Many Requests", "429", "rate-limit"]):
                    logging.warning(f"{label} chunk {ci}/{len(chunks)}: rate-limited, backoff attempt {attempt+1}/{MAX_RETRIES}")
                    _sleep_backoff(attempt)
                    continue
                if any(s in msg.lower() for s in ["timed out", "timeout", "temporary failure", "connection reset"]):
                    logging.warning(f"{label} chunk {ci}/{len(chunks)}: transient error, backoff attempt {attempt+1}/{MAX_RETRIES}")
                    _sleep_backoff(attempt)
                    continue
                # non-transient — give up this chunk
                break

        if chunk and (chunk[0] not in merged):
            if last_exc:
                logging.warning(f"{label} chunk {ci}/{len(chunks)} failed: {last_exc}")
            for t in chunk:
                merged.setdefault(t, None)

    return merged

def _download_single_symbol(sym: str, *, period, interval, label):
    out = _download_batch_chunked([sym], period=period, interval=interval, label=label)
    return out.get(sym)

# ----------------------------------------------------------------------
# Core scan helpers
# ----------------------------------------------------------------------

def _prepare_daily_df(df_daily: pd.DataFrame) -> pd.DataFrame:
    if df_daily is None or df_daily.empty:
        return None
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df_daily = _to_numeric_cols(df_daily.copy(), cols)
    try:
        df_daily = fetch_indicators(df_daily)
        df_daily["ema_fast"] = df_daily["Close"].ewm(span=EMA_FAST, adjust=False).mean()
        df_daily["ema_slow"] = df_daily["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    except Exception:
        return None
    return df_daily

def _extract_ema_trend(df_4h: pd.DataFrame) -> float:
    if df_4h is None or df_4h.empty:
        return np.nan
    try:
        df_4h = _to_numeric_cols(df_4h.copy(), ["Close"])
        df_4h["ema_trend"] = df_4h["Close"].ewm(span=EMA_TREND, adjust=False).mean()
        return to_float(df_4h["ema_trend"].iloc[-1])
    except Exception:
        return np.nan

def _compute_signal_for_df(df_daily: pd.DataFrame, ema_trend_value: float):
    """
    Evaluate EMA(13/21) crossovers with multi-factor confirmation and
    compute a continuous probabilistic confidence score (0.00–1.00).
    """

    from pandas.api.types import is_numeric_dtype

    # --- Ensure numeric columns ---
    for col in df_daily.columns:
        if not is_numeric_dtype(df_daily[col]):
            try:
                df_daily[col] = pd.to_numeric(df_daily[col])
            except Exception:
                pass

    if len(df_daily) < max(EMA_SLOW, 50) or pd.isna(ema_trend_value):
        return None

    # --- Extract latest indicator values ---
    prev_fast = to_float(df_daily["ema_fast"].iloc[-2])
    prev_slow = to_float(df_daily["ema_slow"].iloc[-2])
    ema_fast  = to_float(df_daily["ema_fast"].iloc[-1])
    ema_slow  = to_float(df_daily["ema_slow"].iloc[-1])
    close     = to_float(df_daily["Close"].iloc[-1])
    rsi       = to_float(df_daily["rsi"].iloc[-1])
    adx       = to_float(df_daily["adx"].iloc[-1])
    atr       = to_float(df_daily["atr"].iloc[-1])
    atr_mean  = to_float(df_daily["atr"].rolling(100).mean().iloc[-1])

    vals = [prev_fast, prev_slow, ema_fast, ema_slow, close, rsi, adx, atr, atr_mean, ema_trend_value]
    if any(pd.isna(v) or (isinstance(v, float) and np.isnan(v)) for v in vals):
        return None

    # --- Determine crossover direction and trend filter ---
    crossed_up     = (prev_fast < prev_slow) and (ema_fast > ema_slow)
    crossed_down   = (prev_fast > prev_slow) and (ema_fast < ema_slow)
    is_above_trend = (close > ema_trend_value * TREND_BUF)
    is_below_trend = (close < ema_trend_value / TREND_BUF)

    # --- OBV confirmation (optional) ---
    try:
        obv = df_daily["obv"]
        if len(obv) > 5:
            obv_rising  = float(obv.iloc[-1]) > float(obv.iloc[-5])
            obv_falling = float(obv.iloc[-1]) < float(obv.iloc[-5])
        else:
            obv_rising = obv_falling = True
    except Exception:
        obv_rising = obv_falling = True

    # --- Continuous confidence scoring helper ---
    def _compute_confidence_score(direction, rsi, adx, atr, atr_mean, obv_rising=None, obv_falling=None):
        """Compute a continuous confidence score (0–1) based on normalized RSI, ADX, ATR, and optional OBV."""
        scores = []

        # RSI: distance beyond threshold (0–1 over ~20 points)
        if direction == "BUY":
            rsi_score = np.clip((rsi - RSI_MIN_BUY) / 20.0, 0, 1)
        else:
            rsi_score = np.clip((RSI_MAX_SELL - rsi) / 20.0, 0, 1)
        scores.append(rsi_score)

        # ADX: normalized 15–40 range
        adx_score = np.clip((adx - ADX_MIN) / 25.0, 0, 1)
        scores.append(adx_score)

        # ATR: volatility ratio relative to mean
        atr_ratio = atr / max(atr_mean, 1e-8)
        atr_score = np.clip((atr_ratio - ATR_RATIO_MIN) / (1.5 - ATR_RATIO_MIN), 0, 1)
        scores.append(atr_score)

        # OBV: optional binary bonus
        if USE_OBV:
            if direction == "BUY":
                scores.append(1.0 if obv_rising else 0.0)
            else:
                scores.append(1.0 if obv_falling else 0.0)

        return round(float(np.nanmean(scores)), 2)

    # --- BUY logic ---
    if crossed_up and is_above_trend:
        confidence = _compute_confidence_score("BUY", rsi, adx, atr, atr_mean, obv_rising=obv_rising)
        rsi_pass = rsi > RSI_MIN_BUY
        adx_pass = adx > ADX_MIN
        atr_pass = atr > ATR_RATIO_MIN * atr_mean
        obv_pass = obv_rising if USE_OBV else True
        confirm_score = sum([rsi_pass, adx_pass, atr_pass, obv_pass])

        if confirm_score >= CONFIRM_SCORE_BUY:
            return ("BUY", close, rsi, adx, confidence)

    # --- SELL logic ---
    if crossed_down and is_below_trend:
        confidence = _compute_confidence_score("SELL", rsi, adx, atr, atr_mean, obv_falling=obv_falling)
        rsi_pass = rsi < RSI_MAX_SELL
        adx_pass = adx > ADX_MIN
        atr_pass = atr > ATR_RATIO_MIN * atr_mean
        obv_pass = obv_falling if USE_OBV else True
        confirm_score = sum([rsi_pass, adx_pass, atr_pass, obv_pass])

        if confirm_score >= CONFIRM_SCORE_SELL:
            return ("SELL", close, rsi, adx, confidence)

    return None

# ----------------------------------------------------------------------
# Scanner (batched + chunked)
# ----------------------------------------------------------------------

def scan_tickers_batched(tickers, *, offset=0, batch_size=SCAN_BATCH_SIZE):
    """Scan a rotating batch using chunked multi-ticker downloads."""
    conf_signals = []

    n = len(tickers)
    if n == 0:
        return conf_signals, offset

    start = offset % n
    end = start + batch_size
    if end <= n:
        batch = tickers[start:end]
        next_offset = end % n
    else:
        batch = tickers[start:] + tickers[:(end % n)]
        next_offset = end % n

    LAST_SCAN_SUMMARY.update({
        "universe_size": n,
        "last_batch_start": start,
        "last_batch_size": len(batch),
        "signals_in_batch": 0,
        "coverage_pct": (next_offset / n) * 100.0 if n else 0.0,
        "last_error": None,
    })

    logging.info(
        f"After this batch: next_offset={next_offset} "
        f"(~{(next_offset / n) * 100.0:.1f}% of universe covered)"
    )

    # --- Download price data in chunks ---
    try:
        daily_map = _download_batch_chunked(batch, period="120d", interval=TIMEFRAME_DAILY, label="daily")
        if RATE_LIMIT_DELAY > 0:
            time.sleep(RATE_LIMIT_DELAY)
        h4_map = _download_batch_chunked(batch, period="180d", interval=TIMEFRAME_4H, label="4h")
        if RATE_LIMIT_DELAY > 0:
            time.sleep(RATE_LIMIT_DELAY)
    except Exception as e:
        LAST_SCAN_SUMMARY["last_error"] = f"Batch download failed: {e}"
        logging.warning(f"Batch download failed: {e}")
        return conf_signals, next_offset

    # --- Evaluate each symbol in the batch ---
    for sym in batch:
        df_daily = daily_map.get(sym)
        df_4h = h4_map.get(sym)

        # Fallback single-symbol downloads if missing
        if (df_daily is None) or df_daily.empty:
            df_daily = _download_single_symbol(sym, period="120d", interval=TIMEFRAME_DAILY, label=f"daily:{sym}")
        if (df_4h is None) or df_4h.empty:
            df_4h = _download_single_symbol(sym, period="180d", interval=TIMEFRAME_4H, label=f"4h:{sym}")

        if df_daily is None or df_daily.empty or df_4h is None or df_4h.empty:
            continue

        df_daily = _prepare_daily_df(df_daily)
        if df_daily is None or len(df_daily) < max(EMA_SLOW, 50):
            continue

        ema_trend_value = _extract_ema_trend(df_4h)
        if pd.isna(ema_trend_value):
            continue

        sig = _compute_signal_for_df(df_daily, ema_trend_value)
        if sig is None:
            continue

        # --- Unpack the new return structure ---
        side, close, rsi, adx, conf = sig

        # --- Format and filter message by confidence threshold ---
        MIN_CONFIDENCE_ALERT = float(os.getenv("MIN_CONFIDENCE_ALERT", 0.40))  # adjustable via env var

        if conf < MIN_CONFIDENCE_ALERT:
            logging.info(f"Skipping low-confidence {sym} signal ({conf:.2f})")
            continue

        if side == "BUY":
            msg = f"📈 {sym} BUY @ {close:.2f} | RSI {rsi:.1f}, ADX {adx:.1f}, CONF {conf:.2f}"
            record_signal("BUY", sym, close)
        elif side == "SELL":
            msg = f"🔻 {sym} SELL @ {close:.2f} | RSI {rsi:.1f}, ADX {adx:.1f}, CONF {conf:.2f}"
            record_signal("SELL", sym, close)
        else:
            continue

        conf_signals.append(msg)

    LAST_SCAN_SUMMARY["signals_in_batch"] = len(conf_signals)
    return conf_signals, next_offset

# ----------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------

def record_signal(signal_type, sym, price):
    date = datetime.date.today().isoformat()
    df = pd.DataFrame([[date, signal_type, sym, float(price)]],
                      columns=["date","signal","symbol","price"])
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)

def _download_single_symbol_for_backtest(sym: str):
    return _download_single_symbol(sym, period="10d", interval="1d", label=f"backtest:{sym}")

def evaluate_old_signals():
    if not os.path.exists(LOG_FILE):
        return None

    try:
        df = pd.read_csv(LOG_FILE)
    except Exception as e:
        logging.error(f"Failed to read log file '{LOG_FILE}': {e}")
        return None

    if df.empty:
        return None

    try:
        df["date"]  = pd.to_datetime(df["date"], errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    except Exception:
        pass

    cutoff = datetime.date.today() - datetime.timedelta(days=HOLD_DAYS)
    eligible = df[df["date"].dt.date <= cutoff]
    if eligible.empty:
        return None

    results = []
    for _, row in eligible.iterrows():
        sym  = str(row["symbol"])
        side = str(row["signal"])
        entry = to_float(row["price"])
        try:
            df_hist = _download_single_symbol_for_backtest(sym)
            if df_hist is None or len(df_hist) < HOLD_DAYS:
                continue
            exit_price = float(df_hist["Close"].iloc[HOLD_DAYS - 1])
            ret = (exit_price - entry) / entry if entry else 0.0
            if side.upper() == "SELL":
                ret = -ret
            profit = ret * CAPITAL_PER_TRADE
            results.append((sym, side, float(entry), float(exit_price), float(profit)))
        except Exception as e:
            logging.error(f"Backtest {sym}: {e}")

    if not results:
        return None

    total = sum(p[4] for p in results)
    winrate = sum(1 for p in results if p[4] > 0) / len(results) * 100
    avg_trade = total / len(results)

    msg = "**📊 Weekly Performance Report**\n"
    for sym, side, entry, exitp, prof in results:
        emoji = "✅" if prof > 0 else "❌"
        msg += f"{emoji} {sym} {side} {entry:.2f} → {exitp:.2f} | {prof:+.2f} USD\n"
    msg += f"\n**Total:** {total:+.2f} USD | **Winrate:** {winrate:.1f}% | **Avg:** {avg_trade:+.2f} USD/trade"

    df_remaining = df[~df["symbol"].isin([r[0] for r in results])]
    try:
        df_remaining.to_csv(LOG_FILE, index=False)
    except Exception as e:
        logging.error(f"Failed to update log file: {e}")

    return msg

# ----------------------------------------------------------------------
# Universe
# ----------------------------------------------------------------------

def _build_universe():
    raw = (
        get_sp500_tickers()
        + get_biotech_tickers()
        + get_nasdaq100_tickers()
        + get_dow30_tickers()
        + get_sector_tickers()
    )
    tickers = sorted(set(normalize_tickers(raw)))
    return tickers

# ----------------------------------------------------------------------
# Flask (keep-alive + health)
# ----------------------------------------------------------------------

app = Flask(__name__)

@app.route('/')
def home():
    return "EMA Scanner running."

@app.route('/healthz')
def healthz():
    return jsonify(LAST_SCAN_SUMMARY)

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    try:
        tickers = _build_universe()
        logging.info(f"Total tickers to scan: {len(tickers)}")
        logging.info("🚀 EMA Multi-Factor Scanner Started")
        send_discord_message("🟢 Bot online and scanning…")
    except Exception as e:
        logging.error(f"Failed to build ticker universe: {e}")
        raise

    random.shuffle(tickers)
    LAST_SCAN_SUMMARY["universe_size"] = len(tickers)

    offset = 0
    while True:
        try:
            total_signals = 0
            batches = max(1, BATCHES_PER_LOOP)

            for i in range(batches):
                conf_signals, offset = scan_tickers_batched(
                    tickers, offset=offset, batch_size=SCAN_BATCH_SIZE
                )
                total_signals += len(conf_signals)

                if conf_signals:
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                    msg = f"**✅ EMA Multi-Factor Alerts ({now})**\n" + "\n".join(conf_signals)
                    send_discord_message(msg)

                if i < batches - 1 and BATCH_PAUSE > 0:
                    time.sleep(BATCH_PAUSE)

            if total_signals == 0:
                logging.info(f"{datetime.datetime.now():%H:%M} — No signals in these {batches} batch(es)")

            report = evaluate_old_signals()
            if report:
                send_discord_message(report)

        except Exception as e:
            LAST_SCAN_SUMMARY["last_error"] = f"{type(e).__name__}: {e}"
            logging.error(f"⚠️ Global loop error: {type(e).__name__} — {e}")
            traceback.print_exc()

        time.sleep(CHECK_INTERVAL)

# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------

if __name__ == "__main__":
    try:
        Thread(target=run_flask, daemon=True).start()
        main()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        traceback.print_exc()
