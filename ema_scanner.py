# ============================================================
#         EMA Multi-Factor Scanner Bot — Pro Edition
# ============================================================
# Scans S&P500 + NASDAQ100 + Dow30 + Biotech + Sector ETFs
# Sends Discord alerts for confirmed EMA(13/21) crossovers
# Uses 4H EMA200 trend filter + ADX/OBV/ATR/MACD confirmations
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

# ---- Confirmation scoring tunables (legacy RSI vars remain defined but unused) ----
RSI_MIN_BUY   = float(os.getenv("RSI_MIN_BUY", 50))    # (unused in current logic)
RSI_MAX_SELL  = float(os.getenv("RSI_MAX_SELL", 50))   # (unused in current logic)
ADX_MIN       = float(os.getenv("ADX_MIN", 15))        # still used (optional ADX vote)
ATR_RATIO_MIN = float(os.getenv("ATR_RATIO_MIN", 0.7)) # (unused in current logic)

# Allow a small buffer around the 4h EMA200 trend
TREND_BUF     = float(os.getenv("TREND_BUF", 0.99))    # 0.99 ≈ allow ~1% below for buys

# Require "any N of K" confirmations
CONFIRM_SCORE_BUY  = int(os.getenv("CONFIRM_SCORE_BUY", 2))
CONFIRM_SCORE_SELL = int(os.getenv("CONFIRM_SCORE_SELL", 2))

# Optional: count OBV as an extra vote (False by default)
USE_OBV = os.getenv("USE_OBV", "0") == "1"

# ---- NEW: EMA(21) slope + ATR-z distance thresholds ----
SLOPE_W          = int(os.getenv("SLOPE_W", 5))             # bars for EMA21 slope
SLOPE_MIN_BUY    = float(os.getenv("SLOPE_MIN_BUY", 0.004)) # ≈ +0.3% over SLOPE_W bars
SLOPE_MIN_SELL   = float(os.getenv("SLOPE_MIN_SELL", -0.004)) # ≈ -0.3% over SLOPE_W bars

Z_MIN_BUY        = float(os.getenv("Z_MIN_BUY", -0.4))      # (Close-EMA21)/ATR lower bound (long)
Z_MAX_BUY        = float(os.getenv("Z_MAX_BUY",  0.4))      # upper bound (avoid chases)
Z_MIN_SELL       = float(os.getenv("Z_MIN_SELL", -0.4))     # short-side band
Z_MAX_SELL       = float(os.getenv("Z_MAX_SELL",  0.4))

# ---- NEW: MACD histogram acceleration ----
MACD_FAST        = int(os.getenv("MACD_FAST", 12))
MACD_SLOW        = int(os.getenv("MACD_SLOW", 26))
MACD_SIGNAL      = int(os.getenv("MACD_SIGNAL", 9))
MACD_ACCEL_BARS  = int(os.getenv("MACD_ACCEL_BARS", 2))   # require rising/falling last N bars

# ---- NEW: Optional ADX as an additional vote ----
USE_ADX_CONFIRM  = os.getenv("USE_ADX_CONFIRM", "1") == "1"

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
    # Removed RSI: no longer used in logic

    # ADX
    adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
    df["adx"] = adx.adx()

    # OBV (optional vote)
    obv = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"])
    df["obv"] = obv.on_balance_volume()

    # ATR (for z-distance and risk/confidence)
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14)
    df["atr"] = atr.average_true_range()

    # NEW: MACD and histogram (acceleration)
    macd = ta.trend.MACD(
        close=df["Close"],
        window_slow=MACD_SLOW,
        window_fast=MACD_FAST,
        window_sign=MACD_SIGNAL
    )
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()

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
    EMA(13/21) crossover with:
      - Slope+Distance confirmation (EMA21 slope over SLOPE_W bars + ATR-z distance band)
      - MACD histogram acceleration confirmation
      - Optional OBV vote (if USE_OBV=1)
      - Optional ADX vote (if USE_ADX_CONFIRM=1)
    Returns: ("BUY"/"SELL", close, adx, confidence) or None
    """

    # Ensure numeric
    for col in df_daily.columns:
        if not is_numeric_dtype(df_daily[col]):
            try:
                df_daily[col] = pd.to_numeric(df_daily[col])
            except Exception:
                pass

    if len(df_daily) < max(EMA_SLOW, 50) or pd.isna(ema_trend_value):
        return None

    # --- Latest values ---
    prev_fast = to_float(df_daily["ema_fast"].iloc[-2])
    prev_slow = to_float(df_daily["ema_slow"].iloc[-2])
    ema_fast  = to_float(df_daily["ema_fast"].iloc[-1])
    ema_slow  = to_float(df_daily["ema_slow"].iloc[-1])
    close     = to_float(df_daily["Close"].iloc[-1])
    adx       = to_float(df_daily["adx"].iloc[-1])
    atr       = to_float(df_daily["atr"].iloc[-1])
    # rolling(100) may be NaN on short histories; handle safely in normalization
    atr_mean  = to_float(df_daily["atr"].rolling(100).mean().iloc[-1])
    macd_hist = df_daily["macd_hist"].values

    vals = [prev_fast, prev_slow, ema_fast, ema_slow, close, adx, atr, ema_trend_value]
    if any(pd.isna(v) or (isinstance(v, float) and np.isnan(v)) for v in vals):
        return None
    if len(macd_hist) < (MACD_ACCEL_BARS + 2) or len(df_daily) <= SLOPE_W:
        return None

    # --- Trend (regime) gates vs "trend EMA" ---
    is_above_trend = (close > ema_trend_value * TREND_BUF)
    is_below_trend = (close < ema_trend_value / TREND_BUF)

    # --- Cross directions ---
    crossed_up   = (prev_fast <= prev_slow) and (ema_fast > ema_slow)
    crossed_down = (prev_fast >= prev_slow) and (ema_fast < ema_slow)

    # --- Slope+Distance (EMA21 slope over SLOPE_W bars + ATR-z) ---
    ema_slow_t   = float(df_daily["ema_slow"].iloc[-1])
    ema_slow_tmW = float(df_daily["ema_slow"].iloc[-SLOPE_W])
    slope_21     = (ema_slow_t / ema_slow_tmW - 1.0) if ema_slow_tmW else 0.0
    z_dist       = (close - ema_slow_t) / max(atr, 1e-8)

    slope_dist_buy  = (slope_21 >  SLOPE_MIN_BUY)  and (Z_MIN_BUY  <= z_dist <= Z_MAX_BUY)
    slope_dist_sell = (slope_21 <  SLOPE_MIN_SELL) and (Z_MIN_SELL <= z_dist <= Z_MAX_SELL)

    # --- MACD histogram acceleration ---
    def _macd_accel_ok(long_side: bool):
        n = MACD_ACCEL_BARS
        for i in range(1, n + 1):
            h0 = float(macd_hist[-i])
            h1 = float(macd_hist[-i - 1])
            if long_side:
                if not (h0 > 0 and h0 > h1):
                    return False
            else:
                if not (h0 < 0 and h0 < h1):
                    return False
        return True

    macd_accel_buy  = _macd_accel_ok(True)
    macd_accel_sell = _macd_accel_ok(False)

    # --- Optional ADX vote ---
    adx_pass = (adx > ADX_MIN) if USE_ADX_CONFIRM else None

    # --- Confidence scoring helpers ---
    def _slope_score(slope, min_thr, side):
        if side == "BUY":
            return float(np.clip((slope - min_thr) / max(2*abs(min_thr), 1e-8), 0, 1))
        else:
            # more negative is better
            return float(np.clip((abs(slope) - abs(min_thr)) / max(2*abs(min_thr), 1e-8), 0, 1))

    def _z_score(z, lo, hi):
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        if half <= 0:
            return 0.0
        return float(np.clip(1.0 - abs(z - mid) / half, 0, 1))

    def _macd_score(long_side: bool):
        n = MACD_ACCEL_BARS
        deltas = []
        for i in range(1, n + 1):
            h0 = float(macd_hist[-i]); h1 = float(macd_hist[-i-1])
            d = h0 - h1
            deltas.append(max(d, 0) if long_side else max(-d, 0))
        # Safe normalization independent of ATR scale; fallback if atr_mean is NaN/0
        denom_base = atr_mean if (atr_mean is not None and np.isfinite(atr_mean) and atr_mean > 0) else 1.0
        denom = 0.5 * denom_base
        base = max(float(np.mean(deltas)), 0.0)
        return float(np.clip(base / max(denom, 1e-8), 0, 1))

    def _adx_score(a):
        return float(np.clip((a - ADX_MIN) / 25.0, 0, 1))

    # -------------- BUY logic --------------
    if crossed_up and is_above_trend:
        votes = [slope_dist_buy, macd_accel_buy]
        if USE_OBV:
            try:
                obv = df_daily["obv"]
                obv_rising = float(obv.iloc[-1]) > float(obv.iloc[-5]) if len(obv) > 5 else True
            except Exception:
                obv_rising = True
            votes.append(obv_rising)
        if USE_ADX_CONFIRM:
            votes.append(adx_pass)

        confirm_score = sum(bool(v) for v in votes)

        if confirm_score >= CONFIRM_SCORE_BUY:
            c_parts = [
                _slope_score(slope_21, SLOPE_MIN_BUY, "BUY"),
                _z_score(z_dist, Z_MIN_BUY, Z_MAX_BUY),
                _macd_score(True),
            ]
            if USE_ADX_CONFIRM:
                c_parts.append(_adx_score(adx))
            confidence = round(float(np.nanmean(c_parts)), 2)
            return ("BUY", close, adx, confidence)

    # -------------- SELL logic --------------
    if crossed_down and is_below_trend:
        votes = [slope_dist_sell, macd_accel_sell]
        if USE_OBV:
            try:
                obv = df_daily["obv"]
                obv_falling = float(obv.iloc[-1]) < float(obv.iloc[-5]) if len(obv) > 5 else True
            except Exception:
                obv_falling = True
            votes.append(obv_falling)
        if USE_ADX_CONFIRM:
            votes.append(adx_pass)

        confirm_score = sum(bool(v) for v in votes)

        if confirm_score >= CONFIRM_SCORE_SELL:
            c_parts = [
                _slope_score(slope_21, abs(SLOPE_MIN_SELL), "SELL"),
                _z_score(z_dist, Z_MIN_SELL, Z_MAX_SELL),
                _macd_score(False),
            ]
            if USE_ADX_CONFIRM:
                c_parts.append(_adx_score(adx))
            confidence = round(float(np.nanmean(c_parts)), 2)
            return ("SELL", close, adx, confidence)

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
        side, close, adx, conf = sig

        # --- Format and filter message by confidence threshold ---
        MIN_CONFIDENCE_ALERT = float(os.getenv("MIN_CONFIDENCE_ALERT", 0.40))  # adjustable via env var

        if conf < MIN_CONFIDENCE_ALERT:
            logging.info(f"Skipping low-confidence {sym} signal ({conf:.2f})")
            continue

        if side == "BUY":
            msg = f"📈 {sym} BUY @ {close:.2f} | ADX {adx:.1f}, CONF {conf:.2f}"
            record_signal("BUY", sym, close)
        elif side == "SELL":
            msg = f"🔻 {sym} SELL @ {close:.2f} | ADX {adx:.1f}, CONF {conf:.2f}"
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
    prev_offset = 0  # NEW: to detect wrap-around (full coverage)
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
            
        # --- NEW: exit after one full pass if requested ---
        if RUN_ONCE:
            # When offset wraps (becomes smaller than previous), we've covered the full universe once
            if offset < prev_offset:
                logging.info("✅ Completed one full pass through the universe (RUN_ONCE=1). Exiting.")
                break
            prev_offset = offset

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


