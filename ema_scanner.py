# ============================================================
#  EMA Multi-Factor Scanner Bot — Precision Continuation Edition
# ============================================================
# Scans S&P500 + NASDAQ100 + Dow30 + Biotech + Sector ETFs
# Sends Discord alerts for high-selectivity pullback continuation setups
# Uses TRUE 4H EMA200 trend filter via 1H -> 4H resampling
# Adds market regime, sector regime, event filters, relative strength,
# and optional OpenInsider confirmation for BUY setups.
# ============================================================

import os
import re
import time
import random
import logging
import datetime
import traceback
import urllib.request
from io import StringIO
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

DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")
RUN_ONCE = os.getenv("RUN_ONCE", "0") == "1"

EMA_FAST = int(os.getenv("EMA_FAST", 13))
EMA_SLOW = int(os.getenv("EMA_SLOW", 21))
EMA_TREND = int(os.getenv("EMA_TREND", 200))

TIMEFRAME_DAILY = os.getenv("TIMEFRAME_DAILY", "1d")
TIMEFRAME_4H_BASE = os.getenv("TIMEFRAME_4H_BASE", "1h")
RESAMPLE_4H_RULE = os.getenv("RESAMPLE_4H_RULE", "4h")
TREND_PERIOD = os.getenv("TREND_PERIOD", "180d")

CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", 120))
HOLD_DAYS = int(os.getenv("HOLD_DAYS", 5))
CAPITAL_PER_TRADE = float(os.getenv("CAPITAL_PER_TRADE", 500))
LOG_FILE = os.getenv("LOG_FILE", "trades_log.csv")

# Stricter defaults for precision mode
ADX_MIN = float(os.getenv("ADX_MIN", 22))
TREND_BUF = float(os.getenv("TREND_BUF", 0.995))
CONFIRM_SCORE_BUY = int(os.getenv("CONFIRM_SCORE_BUY", 4))
CONFIRM_SCORE_SELL = int(os.getenv("CONFIRM_SCORE_SELL", 4))
USE_OBV = os.getenv("USE_OBV", "0") == "1"

SLOPE_W = int(os.getenv("SLOPE_W", 5))
SLOPE_MIN_BUY = float(os.getenv("SLOPE_MIN_BUY", 0.006))
SLOPE_MIN_SELL = float(os.getenv("SLOPE_MIN_SELL", -0.006))

Z_MIN_BUY = float(os.getenv("Z_MIN_BUY", -0.20))
Z_MAX_BUY = float(os.getenv("Z_MAX_BUY", 0.35))
Z_MIN_SELL = float(os.getenv("Z_MIN_SELL", -0.35))
Z_MAX_SELL = float(os.getenv("Z_MAX_SELL", 0.20))

MACD_FAST = int(os.getenv("MACD_FAST", 12))
MACD_SLOW = int(os.getenv("MACD_SLOW", 26))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", 9))
MACD_ACCEL_BARS = int(os.getenv("MACD_ACCEL_BARS", 3))
USE_ADX_CONFIRM = os.getenv("USE_ADX_CONFIRM", "1") == "1"

SCAN_BATCH_SIZE = int(os.getenv("SCAN_BATCH_SIZE", 125))
BATCHES_PER_LOOP = int(os.getenv("BATCHES_PER_LOOP", 1))
BATCH_PAUSE = float(os.getenv("BATCH_PAUSE", 3.0))

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
BACKOFF_BASE = float(os.getenv("BACKOFF_BASE", 1.7))
BACKOFF_JITTER_MAX = float(os.getenv("BACKOFF_JITTER_MAX", 0.35))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", 0.0))
YF_BATCH_CHUNK = int(os.getenv("YF_BATCH_CHUNK", 40))

# Precision-mode controls
MARKET_PROXY = os.getenv("MARKET_PROXY", "SPY")
GROWTH_PROXY = os.getenv("GROWTH_PROXY", "QQQ")
MIN_DOLLAR_VOL_M = float(os.getenv("MIN_DOLLAR_VOL_M", 25))
EXTRA_TICKERS = os.getenv("EXTRA_TICKERS", "")
PULLBACK_LOOKBACK = int(os.getenv("PULLBACK_LOOKBACK", 3))
PULLBACK_TOUCH_PCT = float(os.getenv("PULLBACK_TOUCH_PCT", 0.006))
CLOSE_IN_RANGE_MIN = float(os.getenv("CLOSE_IN_RANGE_MIN", 0.60))
TREND_ESTABLISHED_BARS = int(os.getenv("TREND_ESTABLISHED_BARS", 3))
FOURH_SLOPE_BARS = int(os.getenv("FOURH_SLOPE_BARS", 5))
FOURH_SLOPE_MIN = float(os.getenv("FOURH_SLOPE_MIN", 0.002))
FOURH_MAX_STRETCH_ATR = float(os.getenv("FOURH_MAX_STRETCH_ATR", 4.5))
MIN_CONFIDENCE_ALERT = float(os.getenv("MIN_CONFIDENCE_ALERT", 0.70))

REQUIRE_MARKET_REGIME = os.getenv("REQUIRE_MARKET_REGIME", "1") == "1"
REQUIRE_SECTOR_REGIME = os.getenv("REQUIRE_SECTOR_REGIME", "1") == "1"
REQUIRE_RELATIVE_STRENGTH = os.getenv("REQUIRE_RELATIVE_STRENGTH", "1") == "1"

RS_LOOKBACK = int(os.getenv("RS_LOOKBACK", 10))
RS_EMA = int(os.getenv("RS_EMA", 20))

ENABLE_EVENT_FILTER = os.getenv("ENABLE_EVENT_FILTER", "1") == "1"
ENABLE_EARNINGS_FILTER = os.getenv("ENABLE_EARNINGS_FILTER", "1") == "1"
EARNINGS_SKIP_DAYS_BEFORE = int(os.getenv("EARNINGS_SKIP_DAYS_BEFORE", 3))
EARNINGS_SKIP_DAYS_AFTER = int(os.getenv("EARNINGS_SKIP_DAYS_AFTER", 2))
GAP_ATR_MAX = float(os.getenv("GAP_ATR_MAX", 1.20))
ALLOW_BIOTECH_WITHOUT_INSIDERS = os.getenv("ALLOW_BIOTECH_WITHOUT_INSIDERS", "0") == "1"
BIOTECH_WHITELIST = set(
    s.strip().upper() for s in os.getenv("BIOTECH_WHITELIST", "").split(",") if s.strip()
)

ENABLE_OPENINSIDER = os.getenv("ENABLE_OPENINSIDER", "1") == "1"
OPENINSIDER_LOOKBACK_DAYS = int(os.getenv("OPENINSIDER_LOOKBACK_DAYS", 30))
OPENINSIDER_MIN_BUY_VALUE = float(os.getenv("OPENINSIDER_MIN_BUY_VALUE", 100000))
OPENINSIDER_CLUSTER_MIN_COUNT = int(os.getenv("OPENINSIDER_CLUSTER_MIN_COUNT", 2))
OPENINSIDER_BULLISH_SCORE_MIN = float(os.getenv("OPENINSIDER_BULLISH_SCORE_MIN", 0.55))
OPENINSIDER_CACHE_MINUTES = int(os.getenv("OPENINSIDER_CACHE_MINUTES", 60))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M"
)
logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.ERROR)

# ----------------------------------------------------------------------
# Globals
# ----------------------------------------------------------------------

LAST_SCAN_SUMMARY = {
    "universe_size": 0,
    "last_batch_start": None,
    "last_batch_size": 0,
    "signals_in_batch": 0,
    "coverage_pct": 0.0,
    "last_error": None,
}

SECTOR_PROXY_BY_SYMBOL = {}
BIOTECH_SYMBOLS = set()
EARNINGS_CACHE = {}
OPENINSIDER_CACHE = {}

GICS_TO_ETF = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def send_discord_message(content: str):
    if not DISCORD_WEBHOOK:
        logging.warning("No valid webhook URL configured. Alert not sent.")
        return
    try:
        resp = requests.post(DISCORD_WEBHOOK, json={"content": content}, timeout=10)
        if resp.status_code >= 300:
            logging.error(f"Discord send failed: {resp.status_code} {resp.text}")
        else:
            logging.info("Discord alert sent.")
    except Exception as e:
        logging.error(f"Discord send failed: {e}")

def to_float(x):
    try:
        if isinstance(x, (list, np.ndarray)) and len(x) > 0:
            x = x[0]
        return float(x)
    except Exception:
        return np.nan

def to_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

def _normalize_ticker(x):
    if x is None:
        return None
    s = str(x).strip().upper()
    if not s:
        return None
    s = s.split()[0]
    s = s.replace(".", "-")
    if not re.fullmatch(r"[A-Z0-9\-]+", s):
        return None
    if not re.search(r"[A-Z]", s):
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
        html = resp.read().decode("utf-8", errors="ignore")
    return pd.read_html(StringIO(html))

def _sleep_backoff(attempt: int):
    wait = (BACKOFF_BASE ** attempt) + random.random() * BACKOFF_JITTER_MAX
    time.sleep(wait)

def _to_numeric_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def parse_money(x):
    try:
        s = str(x).replace("$", "").replace(",", "").replace("+", "").strip()
        return float(s)
    except Exception:
        return np.nan

def parse_pct(x):
    try:
        s = str(x).replace("%", "").replace("+", "").replace(",", "").strip()
        return float(s)
    except Exception:
        return np.nan

def close_in_range(high, low, close):
    rng = max(high - low, 1e-8)
    return (close - low) / rng

def recent_bar_time():
    return datetime.datetime.utcnow()

# ----------------------------------------------------------------------
# Universe fetchers
# ----------------------------------------------------------------------

def get_sp500_tickers():
    try:
        tables = safe_read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        tickers = normalize_tickers(df["Symbol"].astype(str).tolist())

        symbol_col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        sector_col = "GICS Sector" if "GICS Sector" in df.columns else None
        if sector_col:
            for _, row in df[[symbol_col, sector_col]].dropna().iterrows():
                sym = _normalize_ticker(row[symbol_col])
                sector = str(row[sector_col]).strip()
                etf = GICS_TO_ETF.get(sector)
                if sym and etf:
                    SECTOR_PROXY_BY_SYMBOL[sym] = etf

        logging.info(f"Loaded {len(tickers)} S&P 500 tickers from Wikipedia.")
        return tickers
    except Exception as e:
        logging.error(f"S&P500 fetch failed: {e}")
        fallback = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOG", "META"]
        for sym in fallback:
            SECTOR_PROXY_BY_SYMBOL.setdefault(sym, "XLK")
        return fallback

def get_sp400_tickers():
    try:
        tables = safe_read_html("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies")
        df = tables[0]

        symbol_col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        tickers = normalize_tickers(df[symbol_col].astype(str).tolist())

        sector_col = "GICS Sector" if "GICS Sector" in df.columns else None
        if sector_col:
            for _, row in df[[symbol_col, sector_col]].dropna().iterrows():
                sym = _normalize_ticker(row[symbol_col])
                sector = str(row[sector_col]).strip()
                etf = GICS_TO_ETF.get(sector)
                if sym and etf:
                    SECTOR_PROXY_BY_SYMBOL[sym] = etf

        logging.info(f"Loaded {len(tickers)} S&P 400 tickers from Wikipedia.")
        return tickers
    except Exception as e:
        logging.error(f"S&P400 fetch failed: {e}")
        return ["DKS", "WSM", "FND", "ONTO", "RPM", "ALGN", "SCI"]


def get_sp600_tickers():
    try:
        tables = safe_read_html("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies")
        df = tables[0]

        symbol_col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        tickers = normalize_tickers(df[symbol_col].astype(str).tolist())

        sector_col = "GICS Sector" if "GICS Sector" in df.columns else None
        if sector_col:
            for _, row in df[[symbol_col, sector_col]].dropna().iterrows():
                sym = _normalize_ticker(row[symbol_col])
                sector = str(row[sector_col]).strip()
                etf = GICS_TO_ETF.get(sector)
                if sym and etf:
                    SECTOR_PROXY_BY_SYMBOL[sym] = etf

        logging.info(f"Loaded {len(tickers)} S&P 600 tickers from Wikipedia.")
        return tickers
    except Exception as e:
        logging.error(f"S&P600 fetch failed: {e}")
        return ["BOOT", "INSP", "AXON", "IBP", "CROX", "SHAK", "SMPL"]

def get_biotech_tickers():
    try:
        tables = safe_read_html("https://en.wikipedia.org/wiki/List_of_biotechnology_companies")

        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            for wanted in ["ticker", "symbol", "nyse", "nasdaq"]:
                matches = [i for i, c in enumerate(cols) if wanted in c]
                if matches:
                    idx = matches[0]
                    tickers = t.iloc[:, idx].dropna().astype(str).tolist()
                    tickers = normalize_tickers(tickers)
                    BIOTECH_SYMBOLS.update(tickers[:100])
                    for sym in tickers[:100]:
                        SECTOR_PROXY_BY_SYMBOL.setdefault(sym, "XBI")
                    logging.info(f"Loaded {len(tickers)} biotech tickers from Wikipedia.")
                    return tickers[:100]
        raise ValueError("No ticker-like column found in biotech table.")
    except Exception as e:
        logging.error(f"Biotech list error: {e}")
        fallback = ["BIIB", "REGN", "VRTX", "GILD", "ALNY", "ILMN"]
        BIOTECH_SYMBOLS.update(fallback)
        for sym in fallback:
            SECTOR_PROXY_BY_SYMBOL.setdefault(sym, "XBI")
        return fallback

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
        tickers = normalize_tickers(df["Ticker"].dropna().astype(str).tolist())
        logging.info(f"Loaded {len(tickers)} NASDAQ-100 tickers from Wikipedia.")
        return tickers
    except Exception as e:
        logging.error(f"NASDAQ100 fetch failed: {e}")
        fallback = ["AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOG", "TSLA"]
        for sym in fallback:
            SECTOR_PROXY_BY_SYMBOL.setdefault(sym, "XLK")
        return fallback

def get_dow30_tickers():
    try:
        tables = safe_read_html("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            possible_cols = [c for c in cols if "symbol" in c or "ticker" in c]
            if possible_cols:
                idx = cols.index(possible_cols[0])
                tickers = normalize_tickers(t[t.columns[idx]].dropna().astype(str).tolist())
                logging.info(f"Loaded {len(tickers)} Dow 30 tickers from Wikipedia.")
                return tickers
        raise ValueError("No symbol/ticker column found.")
    except Exception as e:
        logging.error(f"Dow30 fetch failed: {e}")
        fallback = ["AAPL", "MSFT", "CAT", "BA", "JNJ", "PG", "V"]
        return fallback

def get_sector_tickers():
    sector_list = [
        "XLE", "XLF", "XLK", "XLV", "XLI", "XLY",
        "XLU", "XLB", "XLC", "XLP", "XLRE",
        "XBI", "SMH", "SOXX", "SPY", "QQQ"
    ]
    for sym in sector_list:
        SECTOR_PROXY_BY_SYMBOL.setdefault(sym, sym)
    return sector_list
# ----------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------

def fetch_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
    df["adx"] = adx.adx()

    obv = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"])
    df["obv"] = obv.on_balance_volume()

    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14)
    df["atr"] = atr.average_true_range()

    macd = ta.trend.MACD(
        close=df["Close"],
        window_slow=MACD_SLOW,
        window_fast=MACD_FAST,
        window_sign=MACD_SIGNAL
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    return df

# ----------------------------------------------------------------------
# 1H -> TRUE 4H resampling
# ----------------------------------------------------------------------

def resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return None

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    out = _to_numeric_cols(out, ["Open", "High", "Low", "Close", "Volume"])

    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }

    out = out.resample(RESAMPLE_4H_RULE).agg(agg)
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    if out.empty:
        return None
    return out

# ----------------------------------------------------------------------
# Batch downloads
# ----------------------------------------------------------------------

def _download_batch_chunked(tickers, *, period, interval, label):
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
                    group_by="ticker"
                )

                if isinstance(df, pd.DataFrame) and not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        lvl0 = set(df.columns.get_level_values(0))
                        for t in chunk:
                            if t in lvl0:
                                sub = df[t].dropna(how="all")
                                merged[t] = sub if not sub.empty else None
                            else:
                                merged[t] = None
                    else:
                        merged[chunk[0]] = df
                        for t in chunk[1:]:
                            merged[t] = None
                else:
                    raise RuntimeError("Empty DataFrame (possibly rate-limited or unsupported interval)")

                if RATE_LIMIT_DELAY > 0:
                    time.sleep(RATE_LIMIT_DELAY)
                break

            except Exception as e:
                last_exc = e
                msg = str(e)
                msg_l = msg.lower()

                if any(s in msg for s in ["Rate limited", "Too Many Requests", "429", "rate-limit"]):
                    logging.warning(
                        f"{label} chunk {ci}/{len(chunks)}: rate-limited, backoff attempt {attempt+1}/{MAX_RETRIES}"
                    )
                    _sleep_backoff(attempt)
                    continue

                if any(s in msg_l for s in [
                    "timed out", "timeout", "temporary failure", "connection reset",
                    "nonetype", "not subscriptable", "failed download", "jsondecodeerror",
                    "expecting value", "remote end closed connection",
                ]):
                    logging.warning(
                        f"{label} chunk {ci}/{len(chunks)}: transient error, backoff attempt {attempt+1}/{MAX_RETRIES}"
                    )
                    _sleep_backoff(attempt)
                    continue

                logging.warning(f"{label} chunk {ci}/{len(chunks)} non-retryable error: {e}")
                break

        for t in chunk:
            merged.setdefault(t, None)

        if last_exc and all(merged.get(t) is None for t in chunk):
            logging.warning(f"{label} chunk {ci}/{len(chunks)} failed: {last_exc}")

    return merged

def _download_single_symbol(sym: str, *, period, interval, label):
    out = _download_batch_chunked([sym], period=period, interval=interval, label=label)
    return out.get(sym)

# ----------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------

def _prepare_daily_df(df_daily: pd.DataFrame) -> pd.DataFrame:
    if df_daily is None or df_daily.empty:
        return None

    df_daily = _to_numeric_cols(df_daily.copy(), ["Open", "High", "Low", "Close", "Volume"])
    try:
        df_daily = fetch_indicators(df_daily)
        df_daily["ema_fast"] = df_daily["Close"].ewm(span=EMA_FAST, adjust=False).mean()
        df_daily["ema_slow"] = df_daily["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
        df_daily["ema50"] = df_daily["Close"].ewm(span=50, adjust=False).mean()
        df_daily["ema200"] = df_daily["Close"].ewm(span=200, adjust=False).mean()
        df_daily["avg_dollar_vol_20"] = (df_daily["Close"] * df_daily["Volume"]).rolling(20).mean() / 1_000_000.0
        df_daily["close_in_range"] = (
            (df_daily["Close"] - df_daily["Low"]) /
            (df_daily["High"] - df_daily["Low"]).replace(0, np.nan)
        )
        df_daily["close_in_range"] = df_daily["close_in_range"].clip(0, 1)
    except Exception:
        return None

    return df_daily

def _extract_ema_trend(df_1h: pd.DataFrame):
    if df_1h is None or df_1h.empty:
        return None

    try:
        df_1h = _to_numeric_cols(df_1h.copy(), ["Open", "High", "Low", "Close", "Volume"])
        df_4h = resample_to_4h(df_1h)
        if df_4h is None or df_4h.empty or len(df_4h) < max(EMA_TREND // 3, FOURH_SLOPE_BARS + 5):
            return None

        df_4h["atr4h"] = ta.volatility.AverageTrueRange(
            df_4h["High"], df_4h["Low"], df_4h["Close"], window=14
        ).average_true_range()
        df_4h["ema_trend"] = df_4h["Close"].ewm(span=EMA_TREND, adjust=False).mean()

        ema_now = to_float(df_4h["ema_trend"].iloc[-1])
        ema_prev = to_float(df_4h["ema_trend"].iloc[-(FOURH_SLOPE_BARS + 1)])
        close_now = to_float(df_4h["Close"].iloc[-1])
        atr4h = to_float(df_4h["atr4h"].iloc[-1])

        if any(pd.isna(v) for v in [ema_now, ema_prev, close_now, atr4h]):
            return None

        slope = (ema_now / ema_prev - 1.0) if ema_prev else np.nan
        stretch_atr = (close_now - ema_now) / max(atr4h, 1e-8)

        return {
            "ema_trend": ema_now,
            "close_4h": close_now,
            "atr4h": atr4h,
            "slope_4h": slope,
            "stretch_atr": stretch_atr,
            "long_ok": bool(
                close_now > ema_now * TREND_BUF and
                slope >= FOURH_SLOPE_MIN and
                stretch_atr <= FOURH_MAX_STRETCH_ATR
            ),
            "short_ok": bool(
                close_now < ema_now / TREND_BUF and
                slope <= -FOURH_SLOPE_MIN and
                stretch_atr >= -FOURH_MAX_STRETCH_ATR
            ),
        }
    except Exception:
        return None

# ----------------------------------------------------------------------
# Regime and relative-strength helpers
# ----------------------------------------------------------------------

def infer_sector_proxy(sym: str):
    sym = _normalize_ticker(sym)
    if not sym:
        return None
    if sym in SECTOR_PROXY_BY_SYMBOL:
        return SECTOR_PROXY_BY_SYMBOL[sym]
    if sym in BIOTECH_SYMBOLS:
        return "XBI"
    return None

def build_regime_context(df_daily: pd.DataFrame, trend_ctx: dict, label: str):
    if df_daily is None or df_daily.empty or not trend_ctx:
        return {
            "label": label,
            "long_ok": False,
            "short_ok": False,
            "score_long": 0.0,
            "score_short": 0.0,
        }

    try:
        close = to_float(df_daily["Close"].iloc[-1])
        ema50 = to_float(df_daily["ema50"].iloc[-1])
        ema200 = to_float(df_daily["ema200"].iloc[-1])

        slope50 = (ema50 / to_float(df_daily["ema50"].iloc[-6]) - 1.0) if len(df_daily) > 6 else 0.0
        slope200 = (ema200 / to_float(df_daily["ema200"].iloc[-21]) - 1.0) if len(df_daily) > 21 else 0.0

        long_parts = [
            close > ema50,
            ema50 > ema200,
            slope50 > 0,
            slope200 >= 0,
            trend_ctx.get("long_ok", False),
        ]
        short_parts = [
            close < ema50,
            ema50 < ema200,
            slope50 < 0,
            slope200 <= 0,
            trend_ctx.get("short_ok", False),
        ]

        return {
            "label": label,
            "close": close,
            "ema50": ema50,
            "ema200": ema200,
            "slope50": slope50,
            "slope200": slope200,
            "long_ok": all(long_parts),
            "short_ok": all(short_parts),
            "score_long": round(sum(bool(x) for x in long_parts) / len(long_parts), 2),
            "score_short": round(sum(bool(x) for x in short_parts) / len(short_parts), 2),
        }
    except Exception:
        return {
            "label": label,
            "long_ok": False,
            "short_ok": False,
            "score_long": 0.0,
            "score_short": 0.0,
        }

def compute_relative_strength_context(stock_df: pd.DataFrame, market_df: pd.DataFrame, sector_df: pd.DataFrame | None = None):
    out = {
        "vs_market_long": False,
        "vs_market_short": False,
        "vs_sector_long": True,
        "vs_sector_short": True,
        "score": 0.0,
    }

    def _ratio_ctx(a_df, b_df):
        if a_df is None or b_df is None or a_df.empty or b_df.empty:
            return None
        joined = pd.concat(
            [a_df["Close"].rename("a"), b_df["Close"].rename("b")],
            axis=1, join="inner"
        ).dropna()
        if len(joined) < max(RS_EMA, RS_LOOKBACK) + 2:
            return None
        rs = joined["a"] / joined["b"]
        rs_ema = rs.ewm(span=RS_EMA, adjust=False).mean()
        return {
            "above_ema": bool(rs.iloc[-1] > rs_ema.iloc[-1]),
            "rising": bool(rs.iloc[-1] > rs.iloc[-(RS_LOOKBACK + 1)]),
            "falling": bool(rs.iloc[-1] < rs.iloc[-(RS_LOOKBACK + 1)]),
        }

    m = _ratio_ctx(stock_df, market_df)
    if m:
        out["vs_market_long"] = m["above_ema"] and m["rising"]
        out["vs_market_short"] = (not m["above_ema"]) and m["falling"]

    s = _ratio_ctx(stock_df, sector_df) if sector_df is not None else None
    if s:
        out["vs_sector_long"] = s["above_ema"] and s["rising"]
        out["vs_sector_short"] = (not s["above_ema"]) and s["falling"]

    bools = [
        out["vs_market_long"],
        out["vs_sector_long"] if sector_df is not None else True,
    ]
    out["score"] = round(sum(bool(x) for x in bools) / len(bools), 2)
    return out
# ----------------------------------------------------------------------
# Event filter
# ----------------------------------------------------------------------

def _extract_earnings_date_from_calendar_obj(cal):
    if cal is None:
        return None

    try:
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            for key in ["Earnings Date", "earningsDate"]:
                if key in cal.index:
                    val = cal.loc[key]
                    if isinstance(val, pd.Series):
                        val = val.iloc[0]
                    return pd.to_datetime(val).date()
                if key in cal.columns:
                    val = cal[key]
                    if isinstance(val, pd.Series):
                        val = val.iloc[0]
                    return pd.to_datetime(val).date()
        elif isinstance(cal, dict):
            for key in ["Earnings Date", "earningsDate"]:
                if key in cal:
                    val = cal[key]
                    if isinstance(val, (list, tuple)) and val:
                        val = val[0]
                    return pd.to_datetime(val).date()
    except Exception:
        return None

    return None

def get_nearby_earnings_date(sym: str):
    if not ENABLE_EARNINGS_FILTER:
        return None

    now = recent_bar_time()
    cached = EARNINGS_CACHE.get(sym)
    if cached and (now - cached["ts"]).total_seconds() < 6 * 3600:
        return cached.get("date")

    out_date = None
    try:
        tk = yf.Ticker(sym)
        cal = getattr(tk, "calendar", None)
        out_date = _extract_earnings_date_from_calendar_obj(cal)

        if out_date is None:
            try:
                ed = tk.get_earnings_dates(limit=4)
                if ed is not None and len(ed) > 0:
                    idx = pd.to_datetime(ed.index).date
                    today = datetime.date.today()
                    future_or_recent = [d for d in idx if abs((d - today).days) <= 45]
                    if future_or_recent:
                        out_date = sorted(future_or_recent, key=lambda d: abs((d - today).days))[0]
            except Exception:
                pass
    except Exception:
        out_date = None

    EARNINGS_CACHE[sym] = {"ts": now, "date": out_date}
    return out_date

def passes_event_filter(sym: str, side: str, df_daily: pd.DataFrame, insider_ctx: dict | None):
    if not ENABLE_EVENT_FILTER:
        return True, ""

    try:
        prev_close = to_float(df_daily["Close"].iloc[-2])
        today_open = to_float(df_daily["Open"].iloc[-1])
        atr = to_float(df_daily["atr"].iloc[-1])

        gap_atr = abs(today_open - prev_close) / max(atr, 1e-8)
        if gap_atr > GAP_ATR_MAX:
            return False, f"gap>{GAP_ATR_MAX:.2f} ATR"

        if sym in BIOTECH_SYMBOLS and sym not in BIOTECH_WHITELIST and not ALLOW_BIOTECH_WITHOUT_INSIDERS:
            bullish_insider = bool(insider_ctx and insider_ctx.get("bullish_buy_cluster", False))
            if side == "BUY" and not bullish_insider:
                return False, "biotech requires insider confirmation"

        ed = get_nearby_earnings_date(sym)
        if ed is not None:
            today = datetime.date.today()
            delta = (ed - today).days
            if -EARNINGS_SKIP_DAYS_AFTER <= delta <= EARNINGS_SKIP_DAYS_BEFORE:
                return False, f"earnings window ({ed.isoformat()})"

    except Exception as e:
        logging.warning(f"Event filter error for {sym}: {e}")

    return True, ""

# ----------------------------------------------------------------------
# OpenInsider integration
# ----------------------------------------------------------------------

def _find_openinsider_table(tables):
    wanted_cols = {"Filing Date", "Trade Date", "Ticker", "Insider Name", "Title", "Trade Type", "Value"}
    for t in tables:
        cols = {str(c).strip() for c in t.columns}
        if len(wanted_cols.intersection(cols)) >= 5:
            return t
    return None

def get_openinsider_context(sym: str):
    if not ENABLE_OPENINSIDER:
        return {
            "available": False,
            "bullish_buy_cluster": False,
            "bullish_score": 0.0,
            "recent_buy_count": 0,
            "recent_buy_value": 0.0,
            "recent_sell_value": 0.0,
        }

    now = recent_bar_time()
    cached = OPENINSIDER_CACHE.get(sym)
    if cached and (now - cached["ts"]).total_seconds() < OPENINSIDER_CACHE_MINUTES * 60:
        return cached["data"]

    base = {
        "available": False,
        "bullish_buy_cluster": False,
        "bullish_score": 0.0,
        "recent_buy_count": 0,
        "recent_buy_value": 0.0,
        "recent_sell_value": 0.0,
        "recent_quality_buy_count": 0,
        "notes": "",
    }

    try:
        tables = safe_read_html(f"https://openinsider.com/{sym}")
        tbl = _find_openinsider_table(tables)
        if tbl is None or tbl.empty:
            OPENINSIDER_CACHE[sym] = {"ts": now, "data": base}
            return base

        df = tbl.copy()
        if "Trade Date" not in df.columns:
            OPENINSIDER_CACHE[sym] = {"ts": now, "data": base}
            return base

        df["Trade Date"] = pd.to_datetime(df["Trade Date"], errors="coerce")
        df = df.dropna(subset=["Trade Date"])
        cutoff = pd.Timestamp(datetime.date.today() - datetime.timedelta(days=OPENINSIDER_LOOKBACK_DAYS))
        df = df[df["Trade Date"] >= cutoff]

        if df.empty:
            base["available"] = True
            OPENINSIDER_CACHE[sym] = {"ts": now, "data": base}
            return base

        for c in ["Title", "Trade Type", "Value", "ΔOwn"]:
            if c not in df.columns:
                df[c] = ""

        df["ValueNum"] = df["Value"].apply(parse_money).abs()
        df["OwnChgPct"] = df["ΔOwn"].apply(parse_pct)

        title_str = df["Title"].astype(str).str.upper()
        trade_type = df["Trade Type"].astype(str).str.upper()

        quality_title = (
            title_str.str.contains("CEO|CFO|COO|PRES|PRESIDENT|DIR|DIRECTOR|COB|CHAIR", na=False)
        )
        pure_purchase = trade_type.str.contains(r"\bP\b|\bP - PURCHASE\b", regex=True, na=False)
        sale_code = trade_type.str.contains(r"\bS\b|\bSALE\b", regex=True, na=False)
        tax_or_option = trade_type.str.contains("TAX|OPTION|OE|M -|F -", na=False)

        buy_df = df[pure_purchase & ~sale_code].copy()
        sell_df = df[sale_code & ~pure_purchase & ~tax_or_option].copy()
        quality_buy_df = buy_df[quality_title & (buy_df["ValueNum"] >= OPENINSIDER_MIN_BUY_VALUE)].copy()

        buy_value = float(buy_df["ValueNum"].sum()) if not buy_df.empty else 0.0
        sell_value = float(sell_df["ValueNum"].sum()) if not sell_df.empty else 0.0
        quality_buy_count = int(len(quality_buy_df))
        buy_count = int(len(buy_df))

        value_score = np.clip(buy_value / max(OPENINSIDER_MIN_BUY_VALUE * 3.0, 1.0), 0, 1)
        cluster_score = np.clip(quality_buy_count / max(OPENINSIDER_CLUSTER_MIN_COUNT, 1), 0, 1)
        sell_penalty = np.clip(sell_value / max(max(buy_value, 1.0), 1.0), 0, 1)

        bullish_score = float(np.clip(0.55 * cluster_score + 0.45 * value_score - 0.25 * sell_penalty, 0, 1))
        bullish_cluster = (
            quality_buy_count >= OPENINSIDER_CLUSTER_MIN_COUNT or
            bullish_score >= OPENINSIDER_BULLISH_SCORE_MIN
        )

        out = {
            "available": True,
            "bullish_buy_cluster": bool(bullish_cluster),
            "bullish_score": round(bullish_score, 2),
            "recent_buy_count": buy_count,
            "recent_quality_buy_count": quality_buy_count,
            "recent_buy_value": round(buy_value, 2),
            "recent_sell_value": round(sell_value, 2),
            "notes": f"OI buys={buy_count}, quality_buys={quality_buy_count}",
        }
        OPENINSIDER_CACHE[sym] = {"ts": now, "data": out}
        return out
    except Exception as e:
        logging.warning(f"OpenInsider fetch failed for {sym}: {e}")
        OPENINSIDER_CACHE[sym] = {"ts": now, "data": base}
        return base

# ----------------------------------------------------------------------
# Core signal engine
# ----------------------------------------------------------------------

def _pullback_reclaim_buy(df_daily: pd.DataFrame):
    if len(df_daily) < max(EMA_SLOW + 5, 40):
        return False, 0.0, 0.0

    c = df_daily["Close"]
    o = df_daily["Open"]
    h = df_daily["High"]
    l = df_daily["Low"]
    ef = df_daily["ema_fast"]
    es = df_daily["ema_slow"]
    atr = df_daily["atr"]

    trend_established = all(bool(ef.iloc[-i] > es.iloc[-i]) for i in range(1, TREND_ESTABLISHED_BARS + 1))

    recent_lows = l.iloc[-PULLBACK_LOOKBACK:]
    recent_touched = bool(
        (recent_lows.min() <= ef.iloc[-1] * (1 + PULLBACK_TOUCH_PCT)) or
        (recent_lows.min() <= es.iloc[-1] * (1 + PULLBACK_TOUCH_PCT))
    )

    prior_pullback = bool(
        (c.iloc[-2] <= ef.iloc[-2] * (1 + PULLBACK_TOUCH_PCT)) or
        (l.iloc[-2] <= es.iloc[-2] * (1 + PULLBACK_TOUCH_PCT))
    )

    reclaim = bool(
        c.iloc[-1] > ef.iloc[-1] and
        c.iloc[-1] > es.iloc[-1] and
        c.iloc[-1] > o.iloc[-1] and
        close_in_range(h.iloc[-1], l.iloc[-1], c.iloc[-1]) >= CLOSE_IN_RANGE_MIN
    )

    z_dist = (c.iloc[-1] - es.iloc[-1]) / max(atr.iloc[-1], 1e-8)
    setup_ok = trend_established and recent_touched and prior_pullback and reclaim
    return setup_ok, float(z_dist), float(close_in_range(h.iloc[-1], l.iloc[-1], c.iloc[-1]))

def _pullback_reclaim_sell(df_daily: pd.DataFrame):
    if len(df_daily) < max(EMA_SLOW + 5, 40):
        return False, 0.0, 0.0

    c = df_daily["Close"]
    o = df_daily["Open"]
    h = df_daily["High"]
    l = df_daily["Low"]
    ef = df_daily["ema_fast"]
    es = df_daily["ema_slow"]
    atr = df_daily["atr"]

    trend_established = all(bool(ef.iloc[-i] < es.iloc[-i]) for i in range(1, TREND_ESTABLISHED_BARS + 1))

    recent_highs = h.iloc[-PULLBACK_LOOKBACK:]
    recent_touched = bool(
        (recent_highs.max() >= ef.iloc[-1] * (1 - PULLBACK_TOUCH_PCT)) or
        (recent_highs.max() >= es.iloc[-1] * (1 - PULLBACK_TOUCH_PCT))
    )

    prior_pullback = bool(
        (c.iloc[-2] >= ef.iloc[-2] * (1 - PULLBACK_TOUCH_PCT)) or
        (h.iloc[-2] >= es.iloc[-2] * (1 - PULLBACK_TOUCH_PCT))
    )

    reclaim = bool(
        c.iloc[-1] < ef.iloc[-1] and
        c.iloc[-1] < es.iloc[-1] and
        c.iloc[-1] < o.iloc[-1] and
        close_in_range(h.iloc[-1], l.iloc[-1], c.iloc[-1]) <= (1.0 - CLOSE_IN_RANGE_MIN)
    )

    z_dist = (c.iloc[-1] - es.iloc[-1]) / max(atr.iloc[-1], 1e-8)
    setup_ok = trend_established and recent_touched and prior_pullback and reclaim
    return setup_ok, float(z_dist), float(close_in_range(h.iloc[-1], l.iloc[-1], c.iloc[-1]))

def _compute_signal_for_df(df_daily: pd.DataFrame, trend_ctx: dict, market_ctx: dict, sector_ctx: dict | None, rs_ctx: dict):
    for col in df_daily.columns:
        if not is_numeric_dtype(df_daily[col]):
            try:
                df_daily[col] = pd.to_numeric(df_daily[col])
            except Exception:
                pass

    if len(df_daily) < max(EMA_SLOW, 100) or not trend_ctx:
        return None

    close = to_float(df_daily["Close"].iloc[-1])
    adx = to_float(df_daily["adx"].iloc[-1])
    atr = to_float(df_daily["atr"].iloc[-1])
    atr_mean = to_float(df_daily["atr"].rolling(100).mean().iloc[-1])
    macd_hist = df_daily["macd_hist"].values

    if any(pd.isna(v) for v in [close, adx, atr]):
        return None
    if len(macd_hist) < (MACD_ACCEL_BARS + 2) or len(df_daily) <= SLOPE_W:
        return None

    ema_slow_t = float(df_daily["ema_slow"].iloc[-1])
    ema_slow_tmW = float(df_daily["ema_slow"].iloc[-SLOPE_W])
    slope_21 = (ema_slow_t / ema_slow_tmW - 1.0) if ema_slow_tmW else 0.0
    adx_prev = float(df_daily["adx"].iloc[-(MACD_ACCEL_BARS + 1)]) if len(df_daily) > MACD_ACCEL_BARS + 1 else adx
    adx_rising = adx > adx_prev and adx > ADX_MIN

    avg_dollar_vol = to_float(df_daily["avg_dollar_vol_20"].iloc[-1])
    liquid_ok = avg_dollar_vol >= MIN_DOLLAR_VOL_M

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

    def _macd_score(long_side: bool):
        n = MACD_ACCEL_BARS
        deltas = []
        for i in range(1, n + 1):
            h0 = float(macd_hist[-i])
            h1 = float(macd_hist[-i - 1])
            d = h0 - h1
            deltas.append(max(d, 0) if long_side else max(-d, 0))
        denom_base = atr_mean if (atr_mean is not None and np.isfinite(atr_mean) and atr_mean > 0) else 1.0
        denom = 0.5 * denom_base
        base = max(float(np.mean(deltas)), 0.0)
        return float(np.clip(base / max(denom, 1e-8), 0, 1))

    def _adx_score(a):
        return float(np.clip((a - ADX_MIN) / 20.0, 0, 1))

    def _slope_score(slope, min_thr, side):
        if side == "BUY":
            return float(np.clip((slope - min_thr) / max(2 * abs(min_thr), 1e-8), 0, 1))
        return float(np.clip((abs(slope) - abs(min_thr)) / max(2 * abs(min_thr), 1e-8), 0, 1))

    def _z_score(z, lo, hi):
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        if half <= 0:
            return 0.0
        return float(np.clip(1.0 - abs(z - mid) / half, 0, 1))

    market_long_ok = market_ctx.get("long_ok", False) if REQUIRE_MARKET_REGIME else True
    market_short_ok = market_ctx.get("short_ok", False) if REQUIRE_MARKET_REGIME else True
    sector_long_ok = sector_ctx.get("long_ok", False) if (REQUIRE_SECTOR_REGIME and sector_ctx is not None) else True
    sector_short_ok = sector_ctx.get("short_ok", False) if (REQUIRE_SECTOR_REGIME and sector_ctx is not None) else True
    rs_long_ok = (rs_ctx.get("vs_market_long", False) and rs_ctx.get("vs_sector_long", True)) if REQUIRE_RELATIVE_STRENGTH else True
    rs_short_ok = (rs_ctx.get("vs_market_short", False) and rs_ctx.get("vs_sector_short", True)) if REQUIRE_RELATIVE_STRENGTH else True

    buy_setup_ok, z_dist_buy, cir_buy = _pullback_reclaim_buy(df_daily)
    sell_setup_ok, z_dist_sell, cir_sell = _pullback_reclaim_sell(df_daily)

    macd_accel_buy = _macd_accel_ok(True)
    macd_accel_sell = _macd_accel_ok(False)

    if buy_setup_ok and trend_ctx.get("long_ok", False):
        buy_votes = [
            slope_21 > SLOPE_MIN_BUY,
            Z_MIN_BUY <= z_dist_buy <= Z_MAX_BUY,
            macd_accel_buy,
            adx_rising if USE_ADX_CONFIRM else True,
            liquid_ok,
            market_long_ok,
            sector_long_ok,
            rs_long_ok,
        ]
        if USE_OBV:
            try:
                obv = df_daily["obv"]
                buy_votes.append(float(obv.iloc[-1]) > float(obv.iloc[-5]) if len(obv) > 5 else True)
            except Exception:
                buy_votes.append(True)

        if sum(bool(v) for v in buy_votes) >= CONFIRM_SCORE_BUY:
            c_parts = [
                _slope_score(slope_21, SLOPE_MIN_BUY, "BUY"),
                _z_score(z_dist_buy, Z_MIN_BUY, Z_MAX_BUY),
                _macd_score(True),
                _adx_score(adx) if USE_ADX_CONFIRM else 1.0,
                float(np.clip(cir_buy, 0, 1)),
                market_ctx.get("score_long", 1.0),
                sector_ctx.get("score_long", 1.0) if sector_ctx is not None else 1.0,
                rs_ctx.get("score", 1.0),
            ]
            confidence = round(float(np.nanmean(c_parts)), 2)
            return (
                "BUY", close, adx, confidence,
                {
                    "setup": "PULLBACK_RECLAIM",
                    "trend4h_slope": round(trend_ctx.get("slope_4h", 0.0), 4),
                    "z_dist": round(z_dist_buy, 2),
                    "market_score": market_ctx.get("score_long", 0.0),
                    "sector_score": sector_ctx.get("score_long", 0.0) if sector_ctx is not None else 1.0,
                    "rs_score": rs_ctx.get("score", 0.0),
                    "avg_dollar_vol_m": round(avg_dollar_vol, 1),
                }
            )

    if sell_setup_ok and trend_ctx.get("short_ok", False):
        sell_votes = [
            slope_21 < SLOPE_MIN_SELL,
            Z_MIN_SELL <= z_dist_sell <= Z_MAX_SELL,
            macd_accel_sell,
            adx_rising if USE_ADX_CONFIRM else True,
            liquid_ok,
            market_short_ok,
            sector_short_ok,
            rs_short_ok,
        ]
        if USE_OBV:
            try:
                obv = df_daily["obv"]
                sell_votes.append(float(obv.iloc[-1]) < float(obv.iloc[-5]) if len(obv) > 5 else True)
            except Exception:
                sell_votes.append(True)

        if sum(bool(v) for v in sell_votes) >= CONFIRM_SCORE_SELL:
            c_parts = [
                _slope_score(slope_21, abs(SLOPE_MIN_SELL), "SELL"),
                _z_score(z_dist_sell, Z_MIN_SELL, Z_MAX_SELL),
                _macd_score(False),
                _adx_score(adx) if USE_ADX_CONFIRM else 1.0,
                float(np.clip(1.0 - cir_sell, 0, 1)),
                market_ctx.get("score_short", 1.0),
                sector_ctx.get("score_short", 1.0) if sector_ctx is not None else 1.0,
                rs_ctx.get("score", 1.0),
            ]
            confidence = round(float(np.nanmean(c_parts)), 2)
            return (
                "SELL", close, adx, confidence,
                {
                    "setup": "PULLBACK_RECLAIM",
                    "trend4h_slope": round(trend_ctx.get("slope_4h", 0.0), 4),
                    "z_dist": round(z_dist_sell, 2),
                    "market_score": market_ctx.get("score_short", 0.0),
                    "sector_score": sector_ctx.get("score_short", 0.0) if sector_ctx is not None else 1.0,
                    "rs_score": rs_ctx.get("score", 0.0),
                    "avg_dollar_vol_m": round(avg_dollar_vol, 1),
                }
            )

    return None

# ----------------------------------------------------------------------
# Scanner
# ----------------------------------------------------------------------

def _fetch_proxy_maps_for_batch(batch):
    proxies = {MARKET_PROXY, GROWTH_PROXY}
    for sym in batch:
        sp = infer_sector_proxy(sym)
        if sp:
            proxies.add(sp)

    daily = _download_batch_chunked(sorted(proxies), period="300d", interval=TIMEFRAME_DAILY, label="proxy-daily")
    h1 = _download_batch_chunked(sorted(proxies), period=TREND_PERIOD, interval=TIMEFRAME_4H_BASE, label="proxy-1h-base")
    prepared_daily = {}
    trend_map = {}
    regime_map = {}

    for p in proxies:
        d = _prepare_daily_df(daily.get(p))
        prepared_daily[p] = d
        tr = _extract_ema_trend(h1.get(p))
        trend_map[p] = tr
        regime_map[p] = build_regime_context(d, tr, p)

    return prepared_daily, trend_map, regime_map

def scan_tickers_batched(tickers, *, offset=0, batch_size=SCAN_BATCH_SIZE):
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
        f"After this batch: next_offset={next_offset} (~{(next_offset / n) * 100.0:.1f}% of universe covered)"
    )

    try:
        daily_map = _download_batch_chunked(batch, period="240d", interval=TIMEFRAME_DAILY, label="daily")
        if RATE_LIMIT_DELAY > 0:
            time.sleep(RATE_LIMIT_DELAY)

        h1_map = _download_batch_chunked(batch, period=TREND_PERIOD, interval=TIMEFRAME_4H_BASE, label="1h-base-for-4h")
        if RATE_LIMIT_DELAY > 0:
            time.sleep(RATE_LIMIT_DELAY)

        proxy_daily_map, proxy_trend_map, proxy_regime_map = _fetch_proxy_maps_for_batch(batch)
    except Exception as e:
        LAST_SCAN_SUMMARY["last_error"] = f"Batch download failed: {e}"
        logging.warning(f"Batch download failed: {e}")
        return conf_signals, next_offset

    market_df = proxy_daily_map.get(MARKET_PROXY)
    market_regime = proxy_regime_map.get(MARKET_PROXY, {"long_ok": False, "short_ok": False, "score_long": 0.0, "score_short": 0.0})

    for sym in batch:
        df_daily = daily_map.get(sym)
        df_1h = h1_map.get(sym)

        if (df_daily is None) or df_daily.empty:
            df_daily = _download_single_symbol(sym, period="240d", interval=TIMEFRAME_DAILY, label=f"daily:{sym}")
        if (df_1h is None) or df_1h.empty:
            df_1h = _download_single_symbol(sym, period=TREND_PERIOD, interval=TIMEFRAME_4H_BASE, label=f"1h-base-for-4h:{sym}")

        if df_daily is None or df_daily.empty or df_1h is None or df_1h.empty:
            continue

        df_daily = _prepare_daily_df(df_daily)
        if df_daily is None or len(df_daily) < max(EMA_SLOW, 100):
            continue

        trend_ctx = _extract_ema_trend(df_1h)
        if not trend_ctx:
            continue

        sector_proxy = infer_sector_proxy(sym)
        sector_daily = proxy_daily_map.get(sector_proxy) if sector_proxy else None
        sector_regime = proxy_regime_map.get(sector_proxy) if sector_proxy else None
        rs_ctx = compute_relative_strength_context(df_daily, market_df, sector_daily)

        sig = _compute_signal_for_df(df_daily, trend_ctx, market_regime, sector_regime, rs_ctx)
        if sig is None:
            continue

        side, close, adx, conf, meta = sig

        insider_ctx = get_openinsider_context(sym) if (ENABLE_OPENINSIDER and (side == "BUY" or sym in BIOTECH_SYMBOLS)) else None
        ok_event, event_reason = passes_event_filter(sym, side, df_daily, insider_ctx)

        if not ok_event:
            logging.info(f"Skipping {sym} {side}: {event_reason}")
            continue

        # OpenInsider is used as a BUY booster and biotech gate, not a hard SELL confirmer.
        if side == "BUY" and insider_ctx and insider_ctx.get("available"):
            conf = round(min(1.0, conf + 0.15 * insider_ctx.get("bullish_score", 0.0)), 2)
            meta["openinsider_score"] = insider_ctx.get("bullish_score", 0.0)
            meta["openinsider_bullish_cluster"] = insider_ctx.get("bullish_buy_cluster", False)
            meta["openinsider_recent_buy_value"] = insider_ctx.get("recent_buy_value", 0.0)

        if conf < MIN_CONFIDENCE_ALERT:
            logging.info(f"Skipping low-confidence {sym} signal ({conf:.2f})")
            continue

        if side == "BUY":
            extra = ""
            if insider_ctx and insider_ctx.get("bullish_buy_cluster"):
                extra = f" | OI {insider_ctx.get('bullish_score', 0):.2f}"
            msg = f"BUY {sym} @ {close:.2f} | ADX {adx:.1f}, CONF {conf:.2f}{extra}"
            record_signal("BUY", sym, close, conf, meta)
        elif side == "SELL":
            msg = f"SELL {sym} @ {close:.2f} | ADX {adx:.1f}, CONF {conf:.2f}"
            record_signal("SELL", sym, close, conf, meta)
        else:
            continue

        conf_signals.append(msg)

    LAST_SCAN_SUMMARY["signals_in_batch"] = len(conf_signals)
    return conf_signals, next_offset

# ----------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------

def record_signal(signal_type, sym, price, confidence=None, meta=None):
    date = datetime.date.today().isoformat()
    data = {
        "date": date,
        "signal": signal_type,
        "symbol": sym,
        "price": float(price),
        "confidence": float(confidence) if confidence is not None else np.nan,
        "setup": (meta or {}).get("setup", ""),
        "trend4h_slope": (meta or {}).get("trend4h_slope", np.nan),
        "market_score": (meta or {}).get("market_score", np.nan),
        "sector_score": (meta or {}).get("sector_score", np.nan),
        "rs_score": (meta or {}).get("rs_score", np.nan),
    }
    df = pd.DataFrame([data])
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)

def _download_single_symbol_for_backtest(sym: str):
    return _download_single_symbol(sym, period="15d", interval="1d", label=f"backtest:{sym}")

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
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        if "confidence" in df.columns:
            df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    except Exception:
        pass

    cutoff = datetime.date.today() - datetime.timedelta(days=HOLD_DAYS)
    eligible = df[df["date"].dt.date <= cutoff]
    if eligible.empty:
        return None

    results = []
    used_idx = []

    for idx, row in eligible.iterrows():
        sym = str(row["symbol"])
        side = str(row["signal"])
        entry = to_float(row["price"])

        try:
            df_hist = _download_single_symbol_for_backtest(sym)
            if df_hist is None or len(df_hist) < HOLD_DAYS:
                continue

            exit_price = float(df_hist["Close"].iloc[min(HOLD_DAYS - 1, len(df_hist) - 1)])
            ret = (exit_price - entry) / entry if entry else 0.0
            if side.upper() == "SELL":
                ret = -ret
            profit = ret * CAPITAL_PER_TRADE
            results.append((sym, side, float(entry), float(exit_price), float(profit), row.get("confidence", np.nan), row.get("setup", "")))
            used_idx.append(idx)
        except Exception as e:
            logging.error(f"Backtest {sym}: {e}")

    if not results:
        return None

    total = sum(p[4] for p in results)
    winrate = sum(1 for p in results if p[4] > 0) / len(results) * 100
    avg_trade = total / len(results)
    avg_conf = np.nanmean([p[5] for p in results]) if results else np.nan

    msg = "**Weekly Performance Report**\n"
    for sym, side, entry, exitp, prof, conf, setup in results:
        emoji = "WIN" if prof > 0 else "LOSS"
        msg += f"{emoji} {sym} {side} {entry:.2f} -> {exitp:.2f} | {prof:+.2f} USD | C {conf:.2f} | {setup}\n"
    msg += f"\nTotal: {total:+.2f} USD | Winrate: {winrate:.1f}% | Avg: {avg_trade:+.2f} USD/trade | AvgConf: {avg_conf:.2f}"

    df_remaining = df.drop(index=used_idx)
    try:
        df_remaining.to_csv(LOG_FILE, index=False)
    except Exception as e:
        logging.error(f"Failed to update log file: {e}")

    return msg

# ----------------------------------------------------------------------
# Universe
# ----------------------------------------------------------------------

def _build_universe():
    extra = normalize_tickers(
        [x.strip() for x in EXTRA_TICKERS.split(",") if x.strip()]
    )

    raw = (
        get_sp500_tickers()
        + get_sp400_tickers()
        + get_sp600_tickers()
        + get_biotech_tickers()
        + get_nasdaq100_tickers()
        + get_dow30_tickers()
        + get_sector_tickers()
        + extra
    )
    return sorted(set(normalize_tickers(raw)))
# ----------------------------------------------------------------------
# Flask
# ----------------------------------------------------------------------

app = Flask(__name__)

@app.route("/")
def home():
    return "EMA Scanner running."

@app.route("/healthz")
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
        logging.info("EMA Multi-Factor Precision Scanner Started")
        logging.info(f"Daily interval: {TIMEFRAME_DAILY}")
        logging.info(f"4H base download interval: {TIMEFRAME_4H_BASE}")
        logging.info(f"4H resample rule: {RESAMPLE_4H_RULE}")
        logging.info(f"Trend period for 1H history: {TREND_PERIOD}")
        send_discord_message("Bot online and scanning high-selectivity continuation setups.")
    except Exception as e:
        logging.error(f"Failed to build ticker universe: {e}")
        raise

    random.shuffle(tickers)
    LAST_SCAN_SUMMARY["universe_size"] = len(tickers)

    offset = 0
    prev_offset = 0

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
                    msg = f"**EMA Precision Alerts ({now})**\n" + "\n".join(conf_signals)
                    send_discord_message(msg)

                if i < batches - 1 and BATCH_PAUSE > 0:
                    time.sleep(BATCH_PAUSE)

            if total_signals == 0:
                logging.info(f"{datetime.datetime.now():%H:%M} - No signals in these {batches} batch(es)")

            report = evaluate_old_signals()
            if report:
                send_discord_message(report)

        except Exception as e:
            LAST_SCAN_SUMMARY["last_error"] = f"{type(e).__name__}: {e}"
            logging.error(f"Global loop error: {type(e).__name__} - {e}")
            traceback.print_exc()

        if RUN_ONCE:
            if offset < prev_offset:
                logging.info("Completed one full pass through the universe (RUN_ONCE=1). Exiting.")
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
