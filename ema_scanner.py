# ============================================================
#  EMA Multi-Factor Scanner Bot — Precision Continuation Edition
# ============================================================
# Scans S&P 500/400/600 + NASDAQ-100 + Dow 30 + biotech + sector ETFs
# Ranks pullback continuation setups and sends the strongest Discord alerts
# Uses a market-open-aligned 4H EMA200 trend filter via 1H -> 4H resampling
# Adds market regime, sector regime, event filters, relative strength,
# and optional OpenInsider confirmation for BUY setups.
# ============================================================

import datetime
import hmac
import logging
import os
import random
import re
import time
import traceback
from dataclasses import dataclass
from io import StringIO
from threading import Lock, Thread
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import ta
import yfinance as yf
from flask import Flask, jsonify, request

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
# Yahoo's regular-session hourly bars start at 09:30 New York time. Anchoring
# resampling there avoids the 08:00/12:00 buckets produced by pandas' default.
RESAMPLE_4H_OFFSET = os.getenv("RESAMPLE_4H_OFFSET", "9h30min")
TREND_PERIOD = os.getenv("TREND_PERIOD", "180d")

CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", 120))
HOLD_DAYS = int(os.getenv("HOLD_DAYS", 5))
CAPITAL_PER_TRADE = float(os.getenv("CAPITAL_PER_TRADE", 500))
LOG_FILE = os.getenv("LOG_FILE", "trades_log.csv")
RESULTS_LOG_FILE = os.getenv("RESULTS_LOG_FILE", "signal_results.csv")
CALIBRATION_MIN_SAMPLES = int(os.getenv("CALIBRATION_MIN_SAMPLES", 20))
RANK_SCORE_VERSION = os.getenv("RANK_SCORE_VERSION", "v3").strip() or "v3"

# Eligibility and ranking inputs
ADX_MIN = float(os.getenv("ADX_MIN", 17))
ADX_HARD_MIN = float(os.getenv("ADX_HARD_MIN", 17))
TREND_BUF = float(os.getenv("TREND_BUF", 0.995))
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
PROXY_CACHE_MINUTES = int(os.getenv("PROXY_CACHE_MINUTES", 15))
BACKTEST_PERIOD = os.getenv("BACKTEST_PERIOD", "1y")
MAX_ALERTS_PER_CYCLE = int(os.getenv("MAX_ALERTS_PER_CYCLE", 10))
MIN_RANK_SCORE = float(os.getenv("MIN_RANK_SCORE", 0.40))
DAILY_BAR_CLOSE_BUFFER_MINUTES = int(os.getenv("DAILY_BAR_CLOSE_BUFFER_MINUTES", 15))

# Precision-mode controls
MARKET_PROXY = os.getenv("MARKET_PROXY", "SPY")
MIN_DOLLAR_VOL_M = float(os.getenv("MIN_DOLLAR_VOL_M", 25))
MAX_DOLLAR_VOL_M = float(
    os.getenv("MAX_DOLLAR_VOL_M", 0)
)  # 0 = no upper bound; set 1000 to cut mega-cap churn
EXTRA_TICKERS = os.getenv("EXTRA_TICKERS", "")
PULLBACK_LOOKBACK = int(os.getenv("PULLBACK_LOOKBACK", 3))
PULLBACK_TOUCH_PCT = float(os.getenv("PULLBACK_TOUCH_PCT", 0.006))
CLOSE_IN_RANGE_MIN = float(os.getenv("CLOSE_IN_RANGE_MIN", 0.60))
TREND_ESTABLISHED_BARS = int(os.getenv("TREND_ESTABLISHED_BARS", 3))
FOURH_SLOPE_BARS = int(os.getenv("FOURH_SLOPE_BARS", 5))
FOURH_SLOPE_MIN = float(os.getenv("FOURH_SLOPE_MIN", 0.002))
FOURH_MAX_STRETCH_ATR = float(os.getenv("FOURH_MAX_STRETCH_ATR", 4.5))

USE_MARKET_REGIME_SCORE = (
    os.getenv("USE_MARKET_REGIME_SCORE", os.getenv("REQUIRE_MARKET_REGIME", "1")) == "1"
)
USE_SECTOR_REGIME_SCORE = (
    os.getenv("USE_SECTOR_REGIME_SCORE", os.getenv("REQUIRE_SECTOR_REGIME", "1")) == "1"
)
USE_RELATIVE_STRENGTH_SCORE = (
    os.getenv("USE_RELATIVE_STRENGTH_SCORE", os.getenv("REQUIRE_RELATIVE_STRENGTH", "1")) == "1"
)

# Defaults deliberately remain equal-weight. Environment overrides make controlled
# ablation possible without silently changing the production baseline in code.
RANK_COMPONENT_WEIGHTS = {
    "ema_slope": float(os.getenv("WEIGHT_EMA_SLOPE", 1.0)),
    "pullback_location": float(os.getenv("WEIGHT_PULLBACK_LOCATION", 1.0)),
    "macd_momentum": float(os.getenv("WEIGHT_MACD_MOMENTUM", 1.0)),
    "reclaim_candle": float(os.getenv("WEIGHT_RECLAIM_CANDLE", 1.0)),
    "trend_4h": float(os.getenv("WEIGHT_TREND_4H", 1.0)),
    "adx_strength": float(os.getenv("WEIGHT_ADX_STRENGTH", 1.0)),
    "obv_direction": float(os.getenv("WEIGHT_OBV_DIRECTION", 1.0)),
    "market_regime": float(os.getenv("WEIGHT_MARKET_REGIME", 1.0)),
    "sector_regime": float(os.getenv("WEIGHT_SECTOR_REGIME", 1.0)),
    "relative_strength": float(os.getenv("WEIGHT_RELATIVE_STRENGTH", 1.0)),
}

RS_LOOKBACK = int(os.getenv("RS_LOOKBACK", 10))
RS_EMA = int(os.getenv("RS_EMA", 20))

ENABLE_EVENT_FILTER = os.getenv("ENABLE_EVENT_FILTER", "1") == "1"
ENABLE_EARNINGS_FILTER = os.getenv("ENABLE_EARNINGS_FILTER", "1") == "1"
EARNINGS_SKIP_DAYS_BEFORE = int(os.getenv("EARNINGS_SKIP_DAYS_BEFORE", 3))
EARNINGS_SKIP_DAYS_AFTER = int(os.getenv("EARNINGS_SKIP_DAYS_AFTER", 2))
GAP_ATR_MAX = float(os.getenv("GAP_ATR_MAX", 1.20))
ALLOW_BIOTECH_WITHOUT_INSIDERS = os.getenv("ALLOW_BIOTECH_WITHOUT_INSIDERS", "0") == "1"
BIOTECH_WHITELIST = {
    s.strip().upper() for s in os.getenv("BIOTECH_WHITELIST", "").split(",") if s.strip()
}

ENABLE_OPENINSIDER = os.getenv("ENABLE_OPENINSIDER", "1") == "1"
OPENINSIDER_LOOKBACK_DAYS = int(os.getenv("OPENINSIDER_LOOKBACK_DAYS", 30))
OPENINSIDER_MIN_BUY_VALUE = float(os.getenv("OPENINSIDER_MIN_BUY_VALUE", 100000))
OPENINSIDER_CLUSTER_MIN_COUNT = int(os.getenv("OPENINSIDER_CLUSTER_MIN_COUNT", 2))
OPENINSIDER_BULLISH_SCORE_MIN = float(os.getenv("OPENINSIDER_BULLISH_SCORE_MIN", 0.55))
OPENINSIDER_CACHE_MINUTES = int(os.getenv("OPENINSIDER_CACHE_MINUTES", 60))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M"
)
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
    "scan_running": False,
    "manual_scan_running": False,
    "last_scan_started_at": None,
    "last_scan_completed_at": None,
    "last_error": None,
}

SECTOR_PROXY_BY_SYMBOL = {}
BIOTECH_SYMBOLS = set()
EARNINGS_CACHE = {}
OPENINSIDER_CACHE = {}
PROXY_CACHE = {}

RUN_TOKEN = os.getenv("RUN_TOKEN", "")
MANUAL_SCAN_LOCK = Lock()
SCAN_LOCK = Lock()
SIGNAL_LOG_LOCK = Lock()
_RECORDED_SIGNAL_KEYS = None
BACKOFF_RANDOM = random.SystemRandom()
NEW_YORK_TZ = ZoneInfo("America/New_York")


@dataclass
class SignalCandidate:
    """An eligible setup that can be ranked against the rest of the universe."""

    symbol: str
    side: str
    signal_date: datetime.date
    decision_price: float
    adx: float
    score: float
    metadata: dict
    insider_note: str = ""


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


def _split_discord_message(content: str, limit: int = 2_000) -> list[str]:
    """Split a message on line boundaries to respect Discord's content limit."""
    if len(content) <= limit:
        return [content]

    chunks = []
    current = ""
    for line in content.splitlines(keepends=True):
        while len(line) > limit:
            if current:
                chunks.append(current.rstrip("\n"))
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]

        if current and len(current) + len(line) > limit:
            chunks.append(current.rstrip("\n"))
            current = ""
        current += line

    if current:
        chunks.append(current.rstrip("\n"))
    return chunks


def _post_discord_chunk(chunk: str) -> None:
    try:
        resp = requests.post(DISCORD_WEBHOOK, json={"content": chunk}, timeout=10)
        if resp.status_code >= 300:
            logging.error(f"Discord send failed: {resp.status_code} {resp.text}")
        else:
            logging.info("Discord alert sent.")
    except requests.RequestException as exc:
        logging.error(f"Discord send failed: {exc}")


def send_discord_message(content: str):
    """Send one or more webhook posts, splitting oversized Discord messages."""
    if not DISCORD_WEBHOOK:
        logging.warning("No valid webhook URL configured. Alert not sent.")
        return

    for chunk in _split_discord_message(content):
        _post_discord_chunk(chunk)


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
    headers = {"User-Agent": "ema-bot/1.0 (+https://github.com/eddoonn/ema-bot)"}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    return pd.read_html(StringIO(response.text))


def safe_get_json(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/market-activity/quotes/nasdaq-ndx-index",
    }
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    return response.json()


def _sleep_backoff(attempt: int):
    wait = (BACKOFF_BASE**attempt) + BACKOFF_RANDOM.random() * BACKOFF_JITTER_MAX
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


def _weighted_rank_score(components: dict[str, float]) -> float:
    """Combine finite score components using explicit, configurable weights."""
    weighted_sum = 0.0
    total_weight = 0.0
    for name, value in components.items():
        weight = RANK_COMPONENT_WEIGHTS.get(name, 1.0)
        if weight <= 0 or not np.isfinite(value):
            continue
        weighted_sum += float(value) * weight
        total_weight += weight
    return float(np.clip(weighted_sum / total_weight, 0, 1)) if total_weight else 0.0


def recent_bar_time():
    return datetime.datetime.now(datetime.timezone.utc)


def validate_config() -> None:
    """Fail early with one actionable message for unsafe environment settings."""
    errors = []
    positive_values = {
        "EMA_FAST": EMA_FAST,
        "EMA_SLOW": EMA_SLOW,
        "EMA_TREND": EMA_TREND,
        "MACD_FAST": MACD_FAST,
        "MACD_SLOW": MACD_SLOW,
        "MACD_SIGNAL": MACD_SIGNAL,
        "MACD_ACCEL_BARS": MACD_ACCEL_BARS,
        "SCAN_BATCH_SIZE": SCAN_BATCH_SIZE,
        "BATCHES_PER_LOOP": BATCHES_PER_LOOP,
        "YF_BATCH_CHUNK": YF_BATCH_CHUNK,
        "MAX_ALERTS_PER_CYCLE": MAX_ALERTS_PER_CYCLE,
        "CALIBRATION_MIN_SAMPLES": CALIBRATION_MIN_SAMPLES,
        "SLOPE_W": SLOPE_W,
        "PULLBACK_LOOKBACK": PULLBACK_LOOKBACK,
        "TREND_ESTABLISHED_BARS": TREND_ESTABLISHED_BARS,
        "FOURH_SLOPE_BARS": FOURH_SLOPE_BARS,
        "HOLD_DAYS": HOLD_DAYS,
        "RS_LOOKBACK": RS_LOOKBACK,
        "RS_EMA": RS_EMA,
        "OPENINSIDER_LOOKBACK_DAYS": OPENINSIDER_LOOKBACK_DAYS,
        "OPENINSIDER_CLUSTER_MIN_COUNT": OPENINSIDER_CLUSTER_MIN_COUNT,
    }
    errors.extend(f"{name} must be > 0" for name, value in positive_values.items() if value <= 0)

    if EMA_FAST >= EMA_SLOW:
        errors.append("EMA_FAST must be smaller than EMA_SLOW")
    if MACD_FAST >= MACD_SLOW:
        errors.append("MACD_FAST must be smaller than MACD_SLOW")
    if Z_MIN_BUY > Z_MAX_BUY or Z_MIN_SELL > Z_MAX_SELL:
        errors.append("each Z_MIN value must be <= its Z_MAX value")
    if not 0 <= CLOSE_IN_RANGE_MIN <= 1:
        errors.append("CLOSE_IN_RANGE_MIN must be between 0 and 1")
    if not 0 <= MIN_RANK_SCORE <= 1:
        errors.append("MIN_RANK_SCORE must be between 0 and 1")
    if min(MIN_DOLLAR_VOL_M, FOURH_MAX_STRETCH_ATR) < 0:
        errors.append("volume and stretch settings cannot be negative")
    if MAX_DOLLAR_VOL_M < 0:
        errors.append("MAX_DOLLAR_VOL_M cannot be negative")
    if ADX_HARD_MIN <= 0:
        errors.append("ADX_HARD_MIN must be > 0")
    if FOURH_SLOPE_MIN < 0:
        errors.append("FOURH_SLOPE_MIN cannot be negative")
    if TREND_BUF <= 0:
        errors.append("TREND_BUF must be > 0")
    if MAX_RETRIES < 0 or BACKOFF_JITTER_MAX < 0:
        errors.append("retry and backoff-jitter settings cannot be negative")
    if BACKOFF_BASE <= 0:
        errors.append("BACKOFF_BASE must be > 0")
    if min(CHECK_INTERVAL, BATCH_PAUSE, RATE_LIMIT_DELAY, PROXY_CACHE_MINUTES) < 0:
        errors.append("interval, pause, delay, and cache settings cannot be negative")
    if DAILY_BAR_CLOSE_BUFFER_MINUTES < 0:
        errors.append("DAILY_BAR_CLOSE_BUFFER_MINUTES cannot be negative")
    if any(weight < 0 for weight in RANK_COMPONENT_WEIGHTS.values()):
        errors.append("ranking component weights cannot be negative")
    if not any(weight > 0 for weight in RANK_COMPONENT_WEIGHTS.values()):
        errors.append("at least one ranking component weight must be > 0")

    if errors:
        raise ValueError("Invalid configuration: " + "; ".join(errors))


# ----------------------------------------------------------------------
# Universe fetchers
# ----------------------------------------------------------------------


def _register_known_sector_proxies(symbols: dict[str, str]) -> list[str]:
    """Register sector proxies for a small, explicit fallback universe."""
    for symbol, proxy in symbols.items():
        SECTOR_PROXY_BY_SYMBOL.setdefault(symbol, proxy)
    return list(symbols)


def _register_gics_sector_proxies(df: pd.DataFrame, symbol_col: str) -> None:
    sector_col = "GICS Sector" if "GICS Sector" in df.columns else None
    if not sector_col:
        return

    for _, row in df[[symbol_col, sector_col]].dropna().iterrows():
        symbol = _normalize_ticker(row[symbol_col])
        proxy = GICS_TO_ETF.get(str(row[sector_col]).strip())
        if symbol and proxy:
            SECTOR_PROXY_BY_SYMBOL[symbol] = proxy


def _get_sp_index_tickers(url: str, label: str, fallback: dict[str, str]) -> list[str]:
    """Load an S&P constituent table and populate its symbol-to-sector map."""
    try:
        tables = safe_read_html(url)
        df = tables[0]
        symbol_col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        tickers = normalize_tickers(df[symbol_col].astype(str).tolist())
        _register_gics_sector_proxies(df, symbol_col)
        logging.info(f"Loaded {len(tickers)} {label} tickers from Wikipedia.")
        return tickers
    except Exception as e:
        logging.error(f"{label} fetch failed: {e}")
        return _register_known_sector_proxies(fallback)


def get_sp500_tickers():
    return _get_sp_index_tickers(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "S&P 500",
        {
            "AAPL": "XLK",
            "MSFT": "XLK",
            "TSLA": "XLY",
            "NVDA": "XLK",
            "AMZN": "XLY",
            "GOOG": "XLC",
            "META": "XLC",
        },
    )


def get_sp400_tickers():
    return _get_sp_index_tickers(
        "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        "S&P 400",
        {
            "DKS": "XLY",
            "WSM": "XLY",
            "FND": "XLY",
            "ONTO": "XLK",
            "RPM": "XLB",
            "ALGN": "XLV",
            "SCI": "XLY",
        },
    )


def get_sp600_tickers():
    return _get_sp_index_tickers(
        "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
        "S&P 600",
        {
            "BOOT": "XLY",
            "INSP": "XLV",
            "AXON": "XLI",
            "IBP": "XLY",
            "CROX": "XLY",
            "SHAK": "XLY",
            "SMPL": "XLP",
        },
    )


def get_biotech_tickers():
    try:
        tables = safe_read_html("https://en.wikipedia.org/wiki/List_of_biotechnology_companies")

        # First try the old behavior: look for an explicit ticker/symbol column
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            for wanted in ["ticker", "symbol", "nyse", "nasdaq"]:
                matches = [i for i, c in enumerate(cols) if wanted in c]
                if matches:
                    idx = matches[0]
                    tickers = t.iloc[:, idx].dropna().astype(str).tolist()
                    tickers = normalize_tickers(tickers)
                    if tickers:
                        BIOTECH_SYMBOLS.update(tickers[:100])
                        for sym in tickers[:100]:
                            SECTOR_PROXY_BY_SYMBOL.setdefault(sym, "XBI")
                        logging.info(f"Loaded {len(tickers)} biotech tickers from Wikipedia.")
                        return tickers[:100]

        # Fallback for the current redirected Wikipedia page:
        # ticker is embedded in the "Traded on" column, e.g. "NYSE: JNJ"
        extracted = []
        for t in tables:
            cols = [str(c).strip() for c in t.columns]
            traded_cols = [
                c for c in cols if "Traded on" in c or "Listed on" in c or "Exchange" in c
            ]
            if not traded_cols:
                continue

            col = traded_cols[0]
            for val in t[col].dropna().astype(str):
                m = re.search(r":\s*([A-Z][A-Z0-9\.\-]{0,10})\b", val)
                if m:
                    extracted.append(m.group(1))

        tickers = normalize_tickers(extracted)

        # Keep only US-style tradable symbols for this scanner
        tickers = [t for t in tickers if re.fullmatch(r"[A-Z][A-Z0-9\-]{0,10}", t)]

        if tickers:
            tickers = list(dict.fromkeys(tickers))  # dedupe, preserve order
            BIOTECH_SYMBOLS.update(tickers[:100])
            for sym in tickers[:100]:
                SECTOR_PROXY_BY_SYMBOL.setdefault(sym, "XBI")
            logging.info(f"Loaded {len(tickers)} biotech tickers from redirected Wikipedia page.")
            return tickers[:100]

        raise ValueError("No ticker-like data found in biotech table.")

    except Exception as e:
        logging.error(f"Biotech list error: {e}")
        fallback = ["BIIB", "REGN", "VRTX", "GILD", "ALNY", "ILMN"]
        BIOTECH_SYMBOLS.update(fallback)
        for sym in fallback:
            SECTOR_PROXY_BY_SYMBOL.setdefault(sym, "XBI")
        return fallback


def get_nasdaq100_tickers():
    try:
        payload = safe_get_json("https://api.nasdaq.com/api/quote/list-type/nasdaq100")
        rows = payload["data"]["data"]["rows"]
        tickers = normalize_tickers(row.get("symbol") for row in rows)
        if len(tickers) < 90:
            raise ValueError(f"NASDAQ API returned only {len(tickers)} valid symbols")
        logging.info(f"Loaded {len(tickers)} NASDAQ-100 listings from Nasdaq.")
        return tickers
    except Exception as e:
        logging.error(f"NASDAQ100 fetch failed: {e}")
        return _register_known_sector_proxies(
            {
                "AAPL": "XLK",
                "MSFT": "XLK",
                "NVDA": "XLK",
                "META": "XLC",
                "AMZN": "XLY",
                "GOOG": "XLC",
                "TSLA": "XLY",
            }
        )


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
        return _register_known_sector_proxies(
            {
                "AAPL": "XLK",
                "MSFT": "XLK",
                "CAT": "XLI",
                "BA": "XLI",
                "JNJ": "XLV",
                "PG": "XLP",
                "V": "XLF",
            }
        )


def get_sector_tickers():
    sector_list = [
        "XLE",
        "XLF",
        "XLK",
        "XLV",
        "XLI",
        "XLY",
        "XLU",
        "XLB",
        "XLC",
        "XLP",
        "XLRE",
        "XBI",
        "SMH",
        "SOXX",
        "SPY",
        "QQQ",
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
        close=df["Close"], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    return df


# ----------------------------------------------------------------------
# 1H -> TRUE 4H resampling
# ----------------------------------------------------------------------


def resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly Yahoo bars into market-open-aligned four-hour bars.

    US regular trading hours do not divide evenly into four-hour buckets, so the
    second bucket of each session is shorter. This matches how charting platforms
    represent the closing 4H candle and, crucially, avoids mixing sessions.
    """
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

    out = out.resample(
        RESAMPLE_4H_RULE,
        origin="start_day",
        offset=RESAMPLE_4H_OFFSET,
    ).agg(agg)
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    if out.empty:
        return None
    return out


# ----------------------------------------------------------------------
# Batch downloads
# ----------------------------------------------------------------------


def _download_batch_chunked(tickers, *, period, interval, label):
    tickers = list(dict.fromkeys(normalize_tickers(tickers)))
    if not tickers:
        return {}

    chunk_size = max(YF_BATCH_CHUNK, 1)
    chunks = [tickers[i : i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    merged = {}
    attempts = max(MAX_RETRIES, 1)

    for ci, chunk in enumerate(chunks, 1):
        last_exc = None

        for attempt in range(attempts):
            try:
                df = yf.download(
                    chunk,
                    period=period,
                    interval=interval,
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                    group_by="ticker",
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
                    raise RuntimeError(
                        "Empty DataFrame (possibly rate-limited or unsupported interval)"
                    )

                if RATE_LIMIT_DELAY > 0:
                    time.sleep(RATE_LIMIT_DELAY)
                last_exc = None
                break

            except Exception as e:
                last_exc = e
                msg = str(e)
                msg_l = msg.lower()

                if any(
                    marker in msg_l
                    for marker in ["rate limited", "too many requests", "429", "rate-limit"]
                ):
                    logging.warning(
                        f"{label} chunk {ci}/{len(chunks)}: rate-limited, "
                        f"attempt {attempt + 1}/{attempts}"
                    )
                    if attempt + 1 < attempts:
                        _sleep_backoff(attempt)
                        continue
                    break

                if any(
                    s in msg_l
                    for s in [
                        "timed out",
                        "timeout",
                        "temporary failure",
                        "connection reset",
                        "nonetype",
                        "not subscriptable",
                        "failed download",
                        "jsondecodeerror",
                        "expecting value",
                        "remote end closed connection",
                    ]
                ):
                    logging.warning(
                        f"{label} chunk {ci}/{len(chunks)}: transient error, "
                        f"attempt {attempt + 1}/{attempts}"
                    )
                    if attempt + 1 < attempts:
                        _sleep_backoff(attempt)
                        continue
                    break

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


def _drop_incomplete_daily_bar(
    frame: pd.DataFrame, *, now: datetime.datetime | None = None
) -> pd.DataFrame:
    """Exclude today's Yahoo daily candle until the cash session has settled.

    Yahoo can expose the current session as a mutable daily row. Signals depend on
    the completed candle, so consuming that row before the close would make both
    the live decision and a same-close backtest impossible to reproduce.
    """
    if frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        return frame

    current = now or recent_bar_time()
    if current.tzinfo is None:
        current = current.replace(tzinfo=datetime.timezone.utc)
    current_new_york = current.astimezone(NEW_YORK_TZ)

    latest = pd.Timestamp(frame.index[-1])
    if latest.tzinfo is not None:
        latest_date = latest.tz_convert(NEW_YORK_TZ).date()
    else:
        latest_date = latest.date()

    if latest_date > current_new_york.date():
        return frame.iloc[:-1]
    if latest_date < current_new_york.date():
        return frame

    usable_after = datetime.datetime.combine(
        latest_date,
        datetime.time(16, 0),
        tzinfo=NEW_YORK_TZ,
    ) + datetime.timedelta(minutes=DAILY_BAR_CLOSE_BUFFER_MINUTES)
    return frame if current_new_york >= usable_after else frame.iloc[:-1]


def _prepare_daily_df(df_daily: pd.DataFrame) -> pd.DataFrame:
    if df_daily is None or df_daily.empty:
        return None

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing = set(required_columns).difference(df_daily.columns)
    if missing:
        logging.debug(f"Daily data missing required columns: {sorted(missing)}")
        return None

    df_daily = _to_numeric_cols(df_daily.copy(), required_columns)
    df_daily = df_daily[~df_daily.index.duplicated(keep="last")].sort_index()
    df_daily = df_daily.dropna(subset=required_columns)
    df_daily = _drop_incomplete_daily_bar(df_daily)
    if len(df_daily) < max(30, MACD_SLOW + MACD_SIGNAL):
        return None

    try:
        df_daily = fetch_indicators(df_daily)
        df_daily["ema_fast"] = df_daily["Close"].ewm(span=EMA_FAST, adjust=False).mean()
        df_daily["ema_slow"] = df_daily["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
        df_daily["ema50"] = df_daily["Close"].ewm(span=50, adjust=False).mean()
        df_daily["ema200"] = df_daily["Close"].ewm(span=200, adjust=False).mean()
        df_daily["avg_dollar_vol_20"] = (df_daily["Close"] * df_daily["Volume"]).rolling(
            20
        ).mean() / 1_000_000.0
        df_daily["close_in_range"] = (df_daily["Close"] - df_daily["Low"]) / (
            df_daily["High"] - df_daily["Low"]
        ).replace(0, np.nan)
        df_daily["close_in_range"] = df_daily["close_in_range"].clip(0, 1)
    except Exception as exc:
        logging.debug(f"Daily indicator preparation failed: {exc}", exc_info=True)
        return None

    return df_daily


def _extract_ema_trend(df_1h: pd.DataFrame, *, through_date: datetime.date | None = None):
    if df_1h is None or df_1h.empty:
        return None

    try:
        df_1h = _to_numeric_cols(df_1h.copy(), ["Open", "High", "Low", "Close", "Volume"])
        if through_date is not None and isinstance(df_1h.index, pd.DatetimeIndex):
            df_1h = df_1h[[timestamp.date() <= through_date for timestamp in df_1h.index]]
        df_4h = resample_to_4h(df_1h)
        minimum_bars = max(EMA_TREND + FOURH_SLOPE_BARS, FOURH_SLOPE_BARS + 14)
        if df_4h is None or df_4h.empty or len(df_4h) < minimum_bars:
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
                close_now > ema_now * TREND_BUF
                and slope >= FOURH_SLOPE_MIN
                and stretch_atr <= FOURH_MAX_STRETCH_ATR
            ),
            "short_ok": bool(
                close_now < ema_now / TREND_BUF
                and slope <= -FOURH_SLOPE_MIN
                and stretch_atr >= -FOURH_MAX_STRETCH_ATR
            ),
        }
    except Exception as exc:
        logging.debug(f"4H trend extraction failed: {exc}", exc_info=True)
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
        slope200 = (
            (ema200 / to_float(df_daily["ema200"].iloc[-21]) - 1.0) if len(df_daily) > 21 else 0.0
        )

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
    except Exception as exc:
        logging.debug(f"Regime calculation failed for {label}: {exc}", exc_info=True)
        return {
            "label": label,
            "long_ok": False,
            "short_ok": False,
            "score_long": 0.0,
            "score_short": 0.0,
        }


def compute_relative_strength_context(
    stock_df: pd.DataFrame, market_df: pd.DataFrame, sector_df: pd.DataFrame | None = None
):
    """Score relative strength separately for long and short candidates."""
    out = {
        "vs_market_long": False,
        "vs_market_short": False,
        "vs_sector_long": True,
        "vs_sector_short": True,
        "score_long": 0.0,
        "score_short": 0.0,
    }

    def _ratio_ctx(a_df, b_df):
        if a_df is None or b_df is None or a_df.empty or b_df.empty:
            return None
        joined = pd.concat(
            [a_df["Close"].rename("a"), b_df["Close"].rename("b")], axis=1, join="inner"
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

    long_parts = [out["vs_market_long"]]
    short_parts = [out["vs_market_short"]]
    if sector_df is not None:
        long_parts.append(out["vs_sector_long"])
        short_parts.append(out["vs_sector_short"])
    out["score_long"] = round(sum(bool(x) for x in long_parts) / len(long_parts), 2)
    out["score_short"] = round(sum(bool(x) for x in short_parts) / len(short_parts), 2)
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
            except Exception as exc:
                logging.debug(f"Fallback earnings-date lookup failed for {sym}: {exc}")
    except Exception as exc:
        logging.debug(f"Earnings-date lookup failed for {sym}: {exc}")
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

        if (
            sym in BIOTECH_SYMBOLS
            and sym not in BIOTECH_WHITELIST
            and not ALLOW_BIOTECH_WITHOUT_INSIDERS
        ):
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
    wanted_cols = {
        "Filing Date",
        "Trade Date",
        "Ticker",
        "Insider Name",
        "Title",
        "Trade Type",
        "Value",
    }
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
        cutoff = pd.Timestamp(
            datetime.date.today() - datetime.timedelta(days=OPENINSIDER_LOOKBACK_DAYS)
        )
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

        quality_title = title_str.str.contains(
            "CEO|CFO|COO|PRES|PRESIDENT|DIR|DIRECTOR|COB|CHAIR", na=False
        )
        pure_purchase = trade_type.str.contains(r"\bP\b|\bP - PURCHASE\b", regex=True, na=False)
        sale_code = trade_type.str.contains(r"\bS\b|\bSALE\b", regex=True, na=False)
        tax_or_option = trade_type.str.contains("TAX|OPTION|OE|M -|F -", na=False)

        buy_df = df[pure_purchase & ~sale_code].copy()
        sell_df = df[sale_code & ~pure_purchase & ~tax_or_option].copy()
        quality_buy_df = buy_df[
            quality_title & (buy_df["ValueNum"] >= OPENINSIDER_MIN_BUY_VALUE)
        ].copy()

        buy_value = float(buy_df["ValueNum"].sum()) if not buy_df.empty else 0.0
        sell_value = float(sell_df["ValueNum"].sum()) if not sell_df.empty else 0.0
        quality_buy_count = len(quality_buy_df)
        buy_count = len(buy_df)

        value_score = np.clip(buy_value / max(OPENINSIDER_MIN_BUY_VALUE * 3.0, 1.0), 0, 1)
        cluster_score = np.clip(quality_buy_count / max(OPENINSIDER_CLUSTER_MIN_COUNT, 1), 0, 1)
        sell_penalty = np.clip(sell_value / max(max(buy_value, 1.0), 1.0), 0, 1)

        bullish_score = float(
            np.clip(0.55 * cluster_score + 0.45 * value_score - 0.25 * sell_penalty, 0, 1)
        )
        bullish_cluster = (
            quality_buy_count >= OPENINSIDER_CLUSTER_MIN_COUNT
            or bullish_score >= OPENINSIDER_BULLISH_SCORE_MIN
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
    minimum_rows = max(EMA_SLOW + 5, TREND_ESTABLISHED_BARS, PULLBACK_LOOKBACK, 40)
    if len(df_daily) < minimum_rows:
        return False, 0.0, 0.0

    c = df_daily["Close"]
    o = df_daily["Open"]
    h = df_daily["High"]
    low = df_daily["Low"]
    ef = df_daily["ema_fast"]
    es = df_daily["ema_slow"]
    atr = df_daily["atr"]

    trend_established = all(
        bool(ef.iloc[-i] > es.iloc[-i]) for i in range(1, TREND_ESTABLISHED_BARS + 1)
    )

    recent_lows = low.iloc[-PULLBACK_LOOKBACK:]
    recent_touched = bool(
        (recent_lows.min() <= ef.iloc[-1] * (1 + PULLBACK_TOUCH_PCT))
        or (recent_lows.min() <= es.iloc[-1] * (1 + PULLBACK_TOUCH_PCT))
    )

    prior_pullback = bool(
        (c.iloc[-2] <= ef.iloc[-2] * (1 + PULLBACK_TOUCH_PCT))
        or (low.iloc[-2] <= es.iloc[-2] * (1 + PULLBACK_TOUCH_PCT))
    )

    reclaim = bool(
        c.iloc[-1] > ef.iloc[-1]
        and c.iloc[-1] > es.iloc[-1]
        and c.iloc[-1] > o.iloc[-1]
        and close_in_range(h.iloc[-1], low.iloc[-1], c.iloc[-1]) >= CLOSE_IN_RANGE_MIN
    )

    z_dist = (c.iloc[-1] - es.iloc[-1]) / max(atr.iloc[-1], 1e-8)
    setup_ok = trend_established and recent_touched and prior_pullback and reclaim
    return setup_ok, float(z_dist), float(close_in_range(h.iloc[-1], low.iloc[-1], c.iloc[-1]))


def _pullback_reclaim_sell(df_daily: pd.DataFrame):
    minimum_rows = max(EMA_SLOW + 5, TREND_ESTABLISHED_BARS, PULLBACK_LOOKBACK, 40)
    if len(df_daily) < minimum_rows:
        return False, 0.0, 0.0

    c = df_daily["Close"]
    o = df_daily["Open"]
    h = df_daily["High"]
    low = df_daily["Low"]
    ef = df_daily["ema_fast"]
    es = df_daily["ema_slow"]
    atr = df_daily["atr"]

    trend_established = all(
        bool(ef.iloc[-i] < es.iloc[-i]) for i in range(1, TREND_ESTABLISHED_BARS + 1)
    )

    recent_highs = h.iloc[-PULLBACK_LOOKBACK:]
    recent_touched = bool(
        (recent_highs.max() >= ef.iloc[-1] * (1 - PULLBACK_TOUCH_PCT))
        or (recent_highs.max() >= es.iloc[-1] * (1 - PULLBACK_TOUCH_PCT))
    )

    prior_pullback = bool(
        (c.iloc[-2] >= ef.iloc[-2] * (1 - PULLBACK_TOUCH_PCT))
        or (h.iloc[-2] >= es.iloc[-2] * (1 - PULLBACK_TOUCH_PCT))
    )

    reclaim = bool(
        c.iloc[-1] < ef.iloc[-1]
        and c.iloc[-1] < es.iloc[-1]
        and c.iloc[-1] < o.iloc[-1]
        and close_in_range(h.iloc[-1], low.iloc[-1], c.iloc[-1]) <= (1.0 - CLOSE_IN_RANGE_MIN)
    )

    z_dist = (c.iloc[-1] - es.iloc[-1]) / max(atr.iloc[-1], 1e-8)
    setup_ok = trend_established and recent_touched and prior_pullback and reclaim
    return setup_ok, float(z_dist), float(close_in_range(h.iloc[-1], low.iloc[-1], c.iloc[-1]))


def _compute_signal_for_df(
    df_daily: pd.DataFrame, trend_ctx: dict, market_ctx: dict, sector_ctx: dict | None, rs_ctx: dict
):
    """Evaluate a prepared daily frame against the configured signal filters."""
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
    ema_slow_tmW = float(df_daily["ema_slow"].iloc[-(SLOPE_W + 1)])
    slope_21 = (ema_slow_t / ema_slow_tmW - 1.0) if ema_slow_tmW else 0.0
    adx_prev = (
        float(df_daily["adx"].iloc[-(MACD_ACCEL_BARS + 1)])
        if len(df_daily) > MACD_ACCEL_BARS + 1
        else adx
    )
    adx_rising = adx > adx_prev and adx > ADX_MIN

    avg_dollar_vol = to_float(df_daily["avg_dollar_vol_20"].iloc[-1])
    liquid_ok = avg_dollar_vol >= MIN_DOLLAR_VOL_M
    if MAX_DOLLAR_VOL_M > 0 and avg_dollar_vol > MAX_DOLLAR_VOL_M:
        liquid_ok = False
    # Hard ADX floor: previously ADX was only a ranking vote, allowing weak-trend
    # trades (ADX <15) that dominate the -441 PnL / 19.2% large-loss bucket.
    # Enforce a hard eligibility gate here; tuned to 18 from 22 via 2026 sweep
    # (ADX 22-30: +617, 15-22: +133, <15: -441).
    if not np.isfinite(adx) or adx < ADX_HARD_MIN:
        return None

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
        denom_base = (
            atr_mean if (atr_mean is not None and np.isfinite(atr_mean) and atr_mean > 0) else 1.0
        )
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

    buy_setup_ok, z_dist_buy, cir_buy = _pullback_reclaim_buy(df_daily)
    sell_setup_ok, z_dist_sell, cir_sell = _pullback_reclaim_sell(df_daily)

    macd_accel_buy = _macd_accel_ok(True)
    macd_accel_sell = _macd_accel_ok(False)

    def _ranked_result(side: str, z_distance: float, candle_quality: float, votes: dict):
        long_side = side == "BUY"
        slope_threshold = SLOPE_MIN_BUY if long_side else abs(SLOPE_MIN_SELL)
        trend_slope = to_float(trend_ctx.get("slope_4h", 0.0))
        components = {
            "ema_slope": _slope_score(slope_21, slope_threshold, side),
            "pullback_location": _z_score(
                z_distance,
                Z_MIN_BUY if long_side else Z_MIN_SELL,
                Z_MAX_BUY if long_side else Z_MAX_SELL,
            ),
            "macd_momentum": _macd_score(long_side),
            "reclaim_candle": float(
                np.clip(candle_quality if long_side else 1.0 - candle_quality, 0, 1)
            ),
            "trend_4h": float(
                np.clip(
                    (abs(trend_slope) - FOURH_SLOPE_MIN) / max(2 * FOURH_SLOPE_MIN, 1e-8),
                    0,
                    1,
                )
            ),
        }
        if USE_ADX_CONFIRM:
            components["adx_strength"] = _adx_score(adx)
        if USE_OBV:
            components["obv_direction"] = float(votes.get("obv", False))
        if USE_MARKET_REGIME_SCORE:
            components["market_regime"] = to_float(
                market_ctx.get("score_long" if long_side else "score_short", 0.0)
            )
        if USE_SECTOR_REGIME_SCORE:
            components["sector_regime"] = to_float(
                (sector_ctx or {}).get("score_long" if long_side else "score_short", 0.0)
            )
        if USE_RELATIVE_STRENGTH_SCORE:
            components["relative_strength"] = to_float(
                rs_ctx.get("score_long" if long_side else "score_short", 0.0)
            )

        rank_score = round(_weighted_rank_score(components), 4)
        context_suffix = "long" if long_side else "short"
        return (
            side,
            close,
            adx,
            rank_score,
            {
                "setup": "PULLBACK_RECLAIM",
                "trend4h_slope": round(trend_slope, 4),
                "z_dist": round(z_distance, 2),
                "market_score": market_ctx.get(f"score_{context_suffix}", 0.0),
                "sector_score": (sector_ctx or {}).get(f"score_{context_suffix}", 0.0),
                "rs_score": rs_ctx.get(f"score_{context_suffix}", 0.0),
                "avg_dollar_vol_m": round(avg_dollar_vol, 1),
                "technical_votes": sum(bool(value) for value in votes.values()),
                "technical_vote_count": len(votes),
                "technical_evidence": votes,
                "score_components": {
                    name: round(float(value), 4) for name, value in components.items()
                },
            },
        )

    # The setup, stock trend, and liquidity remain eligibility gates. Technical and
    # broader context indicators contribute continuous ranking evidence instead of
    # forming brittle all-confirmations vetoes.
    if buy_setup_ok and trend_ctx.get("long_ok", False) and liquid_ok:
        buy_votes = {
            "ema_slope": slope_21 > SLOPE_MIN_BUY,
            "pullback_location": Z_MIN_BUY <= z_dist_buy <= Z_MAX_BUY,
            "macd_momentum": macd_accel_buy,
        }
        if USE_ADX_CONFIRM:
            buy_votes["adx_strength"] = adx_rising
        if USE_OBV:
            obv = df_daily.get("obv")
            buy_votes["obv"] = bool(
                obv is not None
                and len(obv) > 5
                and np.isfinite(to_float(obv.iloc[-1]))
                and np.isfinite(to_float(obv.iloc[-5]))
                and to_float(obv.iloc[-1]) > to_float(obv.iloc[-5])
            )
        return _ranked_result("BUY", z_dist_buy, cir_buy, buy_votes)

    if sell_setup_ok and trend_ctx.get("short_ok", False) and liquid_ok:
        sell_votes = {
            "ema_slope": slope_21 < SLOPE_MIN_SELL,
            "pullback_location": Z_MIN_SELL <= z_dist_sell <= Z_MAX_SELL,
            "macd_momentum": macd_accel_sell,
        }
        if USE_ADX_CONFIRM:
            sell_votes["adx_strength"] = adx_rising
        if USE_OBV:
            obv = df_daily.get("obv")
            sell_votes["obv"] = bool(
                obv is not None
                and len(obv) > 5
                and np.isfinite(to_float(obv.iloc[-1]))
                and np.isfinite(to_float(obv.iloc[-5]))
                and to_float(obv.iloc[-1]) < to_float(obv.iloc[-5])
            )
        return _ranked_result("SELL", z_dist_sell, cir_sell, sell_votes)

    return None


# ----------------------------------------------------------------------
# Scanner
# ----------------------------------------------------------------------


def _fetch_proxy_maps_for_batch(batch):
    """Return prepared proxy data, refreshing only stale/missing symbols.

    Sector and market proxies repeat across every scanner batch. Caching their
    prepared contexts removes two redundant Yahoo requests per batch while
    retaining a short, configurable freshness window.
    """
    proxies = {MARKET_PROXY}
    for sym in batch:
        sp = infer_sector_proxy(sym)
        if sp:
            proxies.add(sp)

    now = time.monotonic()
    ttl_seconds = max(PROXY_CACHE_MINUTES, 0) * 60
    stale_proxies = [
        proxy
        for proxy in sorted(proxies)
        if proxy not in PROXY_CACHE
        or ttl_seconds == 0
        or now - PROXY_CACHE[proxy]["cached_at"] >= ttl_seconds
    ]

    if stale_proxies:
        daily = _download_batch_chunked(
            stale_proxies,
            period="300d",
            interval=TIMEFRAME_DAILY,
            label="proxy-daily",
        )
        h1 = _download_batch_chunked(
            stale_proxies,
            period=TREND_PERIOD,
            interval=TIMEFRAME_4H_BASE,
            label="proxy-1h-base",
        )

        for proxy in stale_proxies:
            prepared = _prepare_daily_df(daily.get(proxy))
            through_date = None
            if prepared is not None and not prepared.empty:
                last_index = pd.to_datetime(prepared.index[-1], errors="coerce")
                if not pd.isna(last_index):
                    through_date = last_index.date()
            trend = _extract_ema_trend(h1.get(proxy), through_date=through_date)
            PROXY_CACHE[proxy] = {
                "cached_at": now,
                "daily": prepared,
                "regime": build_regime_context(prepared, trend, proxy),
            }

    prepared_daily = {proxy: PROXY_CACHE[proxy]["daily"] for proxy in proxies}
    regime_map = {proxy: PROXY_CACHE[proxy]["regime"] for proxy in proxies}
    return prepared_daily, regime_map


def _select_batch(tickers, offset, batch_size):
    """Select the next batch without crossing the current universe boundary."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if not tickers:
        return [], offset, True

    start = offset % len(tickers)
    # Do not fill the tail batch with symbols from the next pass. Besides wasting
    # requests, that made manual "one pass" runs scan early-universe symbols twice.
    size = min(batch_size, len(tickers) - start)
    batch = tickers[start : start + size]
    completed_cycle = start + size == len(tickers)
    next_offset = 0 if completed_cycle else start + size
    return batch, next_offset, completed_cycle


def scan_tickers_batched(tickers, *, offset=0, batch_size=SCAN_BATCH_SIZE):
    """Serialize a scanner batch so manual and continuous runs cannot overlap."""
    with SCAN_LOCK:
        LAST_SCAN_SUMMARY["scan_running"] = True
        LAST_SCAN_SUMMARY["last_scan_started_at"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
        try:
            return _scan_tickers_batched_unlocked(
                tickers,
                offset=offset,
                batch_size=batch_size,
            )
        finally:
            LAST_SCAN_SUMMARY["scan_running"] = False
            LAST_SCAN_SUMMARY["last_scan_completed_at"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()


def _scan_symbol(
    sym,
    daily_map,
    h1_map,
    proxy_daily_map,
    proxy_regime_map,
    market_df,
    market_regime,
):
    """Evaluate one symbol and return an eligible, unpersisted candidate."""
    df_daily = daily_map.get(sym)
    df_1h = h1_map.get(sym)

    if df_daily is None or df_daily.empty:
        df_daily = _download_single_symbol(
            sym, period="240d", interval=TIMEFRAME_DAILY, label=f"daily:{sym}"
        )
    if df_1h is None or df_1h.empty:
        df_1h = _download_single_symbol(
            sym,
            period=TREND_PERIOD,
            interval=TIMEFRAME_4H_BASE,
            label=f"1h-base-for-4h:{sym}",
        )

    if df_daily is None or df_daily.empty or df_1h is None or df_1h.empty:
        return None

    df_daily = _prepare_daily_df(df_daily)
    if df_daily is None or len(df_daily) < max(EMA_SLOW, 100):
        return None

    signal_timestamp = pd.to_datetime(df_daily.index[-1], errors="coerce")
    if pd.isna(signal_timestamp):
        return None
    signal_date = signal_timestamp.date()
    trend_ctx = _extract_ema_trend(df_1h, through_date=signal_date)
    if not trend_ctx:
        return None

    sector_proxy = infer_sector_proxy(sym)
    sector_daily = proxy_daily_map.get(sector_proxy) if sector_proxy else None
    sector_regime = proxy_regime_map.get(sector_proxy) if sector_proxy else None
    rs_ctx = compute_relative_strength_context(df_daily, market_df, sector_daily)

    signal = _compute_signal_for_df(
        df_daily,
        trend_ctx,
        market_regime,
        sector_regime,
        rs_ctx,
    )
    if signal is None:
        return None

    side, close, adx, rank_score, meta = signal
    insider_ctx = (
        get_openinsider_context(sym)
        if ENABLE_OPENINSIDER and (side == "BUY" or sym in BIOTECH_SYMBOLS)
        else None
    )
    ok_event, event_reason = passes_event_filter(sym, side, df_daily, insider_ctx)
    if not ok_event:
        logging.info(f"Skipping {sym} {side}: {event_reason}")
        return None

    # OpenInsider can lift a BUY's rank and remains a biotech gate; it is not a
    # hard SELL confirmer.
    if side == "BUY" and insider_ctx and insider_ctx.get("available"):
        insider_score = float(np.clip(insider_ctx.get("bullish_score", 0.0), 0, 1))
        rank_score = round(min(1.0, rank_score + 0.15 * insider_score), 4)
        meta["openinsider_score"] = insider_ctx.get("bullish_score", 0.0)
        meta["openinsider_bullish_cluster"] = insider_ctx.get("bullish_buy_cluster", False)
        meta["openinsider_recent_buy_value"] = insider_ctx.get("recent_buy_value", 0.0)

    if rank_score < MIN_RANK_SCORE:
        logging.info(f"Skipping {sym} below optional rank floor ({rank_score:.3f})")
        return None

    insider_note = ""
    if side == "BUY":
        if insider_ctx and insider_ctx.get("bullish_buy_cluster"):
            insider_note = f" | OI {insider_ctx.get('bullish_score', 0):.2f}"
    elif side != "SELL":
        return None

    meta["signal_date"] = signal_date.isoformat()
    meta["entry_rule"] = "NEXT_SESSION_OPEN"
    return SignalCandidate(
        symbol=sym,
        side=side,
        signal_date=signal_date,
        decision_price=close,
        adx=adx,
        score=rank_score,
        metadata=meta,
        insider_note=insider_note,
    )


def _scan_tickers_batched_unlocked(tickers, *, offset=0, batch_size=SCAN_BATCH_SIZE):
    candidates = []

    n = len(tickers)
    if n == 0:
        return candidates, offset, True

    start = offset % n
    batch, next_offset, completed_cycle = _select_batch(tickers, offset, batch_size)
    coverage_pct = 100.0 if completed_cycle else (next_offset / n) * 100.0

    LAST_SCAN_SUMMARY.update(
        {
            "universe_size": n,
            "last_batch_start": start,
            "last_batch_size": len(batch),
            "signals_in_batch": 0,
            "coverage_pct": coverage_pct,
            "last_error": None,
        }
    )

    logging.info(
        f"After this batch: next_offset={next_offset} (~{coverage_pct:.1f}% of universe covered)"
    )

    try:
        daily_map = _download_batch_chunked(
            batch, period="240d", interval=TIMEFRAME_DAILY, label="daily"
        )
        if RATE_LIMIT_DELAY > 0:
            time.sleep(RATE_LIMIT_DELAY)

        h1_map = _download_batch_chunked(
            batch, period=TREND_PERIOD, interval=TIMEFRAME_4H_BASE, label="1h-base-for-4h"
        )
        if RATE_LIMIT_DELAY > 0:
            time.sleep(RATE_LIMIT_DELAY)

        proxy_daily_map, proxy_regime_map = _fetch_proxy_maps_for_batch(batch)
    except Exception as e:
        LAST_SCAN_SUMMARY["last_error"] = f"Batch download failed: {e}"
        logging.warning(f"Batch download failed: {e}")
        return candidates, next_offset, completed_cycle

    market_df = proxy_daily_map.get(MARKET_PROXY)
    market_regime = proxy_regime_map.get(
        MARKET_PROXY, {"long_ok": False, "short_ok": False, "score_long": 0.0, "score_short": 0.0}
    )

    for sym in batch:
        try:
            candidate = _scan_symbol(
                sym,
                daily_map,
                h1_map,
                proxy_daily_map,
                proxy_regime_map,
                market_df,
                market_regime,
            )
        except Exception as exc:
            LAST_SCAN_SUMMARY["last_error"] = f"{sym}: {type(exc).__name__}: {exc}"
            logging.warning(f"Skipping {sym} after scanner error: {exc}", exc_info=True)
            continue

        if candidate:
            candidates.append(candidate)

    candidates.sort(key=lambda item: (-item.score, item.symbol, item.side))
    LAST_SCAN_SUMMARY["signals_in_batch"] = len(candidates)
    return candidates, next_offset, completed_cycle


def _next_open_entry_is_pending(
    signal_date: datetime.date, *, now: datetime.datetime | None = None
) -> bool:
    """Return whether the first regular open after the signal is still tradable."""
    current = now or recent_bar_time()
    if current.tzinfo is None:
        current = current.replace(tzinfo=datetime.timezone.utc)
    current_new_york = current.astimezone(NEW_YORK_TZ)

    if signal_date > current_new_york.date():
        return False
    if signal_date == current_new_york.date():
        usable_after = datetime.datetime.combine(
            signal_date,
            datetime.time(16, 0),
            tzinfo=NEW_YORK_TZ,
        ) + datetime.timedelta(minutes=DAILY_BAR_CLOSE_BUFFER_MINUTES)
        return current_new_york >= usable_after

    # Before a weekday open, or throughout a weekend, the next available cash
    # session has not started. The daily scheduler normally uses the same-date path.
    if current_new_york.weekday() >= 5:
        return True
    return current_new_york.time() < datetime.time(9, 30)


def _finalize_ranked_candidates(
    candidates: list[SignalCandidate], *, now: datetime.datetime | None = None
) -> list[str]:
    """Persist and format the best candidates from one complete universe pass."""
    ranked = sorted(candidates, key=lambda item: (-item.score, item.symbol, item.side))
    try:
        results_history = _read_results_log()
    except Exception as exc:
        logging.warning(f"Could not read historical results for alert calibration: {exc}")
        results_history = pd.DataFrame()

    alerts = []
    for candidate in ranked:
        if len(alerts) >= MAX_ALERTS_PER_CYCLE:
            break
        if not _next_open_entry_is_pending(candidate.signal_date, now=now):
            logging.info(
                f"Skipping stale {candidate.side} candidate for {candidate.symbol}: "
                f"the next-open entry window after {candidate.signal_date} has passed"
            )
            continue

        rank = len(alerts) + 1
        metadata = dict(candidate.metadata)
        metadata["universe_rank"] = rank
        recorded = record_signal(
            candidate.side,
            candidate.symbol,
            candidate.decision_price,
            candidate.score,
            metadata,
            signal_date=candidate.signal_date,
        )
        if not recorded:
            logging.debug(
                f"Skipping duplicate {candidate.side} signal for {candidate.symbol} "
                f"on {candidate.signal_date}"
            )
            continue

        votes = metadata.get("technical_votes", 0)
        vote_count = metadata.get("technical_vote_count", 0)
        empirical = _empirical_win_context(
            candidate.score,
            candidate.side,
            results_history,
        )
        empirical_note = ""
        if empirical:
            empirical_note = (
                f" | HIST {HOLD_DAYS}S {empirical['win_rate']:.0%} (n={empirical['samples']})"
            )
        alerts.append(
            f"#{rank} {candidate.side} {candidate.symbol} | SCORE {candidate.score:.3f} | "
            f"decision close {candidate.decision_price:.2f} | entry next open | "
            f"ADX {candidate.adx:.1f} | TECH {votes}/{vote_count}"
            f"{empirical_note}{candidate.insider_note}"
        )
    return alerts


# ----------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------


def _signal_key(date_value, signal_type, symbol) -> tuple[str, str, str]:
    date_text = pd.to_datetime(date_value, errors="coerce")
    normalized_date = str(date_value)[:10] if pd.isna(date_text) else date_text.date().isoformat()
    return normalized_date, str(signal_type).upper(), str(symbol).upper()


def _read_signal_log() -> pd.DataFrame:
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        return pd.DataFrame()
    return pd.read_csv(LOG_FILE)


def _read_results_log() -> pd.DataFrame:
    if not os.path.exists(RESULTS_LOG_FILE) or os.path.getsize(RESULTS_LOG_FILE) == 0:
        return pd.DataFrame()
    return pd.read_csv(RESULTS_LOG_FILE)


def _write_csv_atomic(frame: pd.DataFrame, path: str) -> None:
    """Replace one CSV only after its complete successor is on the same filesystem."""
    absolute_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
    temporary_path = f"{absolute_path}.tmp"
    try:
        frame.to_csv(temporary_path, index=False)
        os.replace(temporary_path, absolute_path)
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def _persist_signal_results(records: list[dict]) -> pd.DataFrame:
    """Idempotently add evaluated signals to the permanent outcome ledger."""
    existing = _read_results_log()
    if not records:
        return existing

    new_rows = pd.DataFrame(records)
    combined = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    key_columns = ["signal_date", "signal", "symbol", "hold_sessions"]
    missing = set(key_columns).difference(combined.columns)
    if missing:
        raise ValueError(f"Results ledger is missing key columns: {sorted(missing)}")

    combined = combined.drop_duplicates(key_columns, keep="last")
    combined = combined.sort_values(key_columns).reset_index(drop=True)
    _write_csv_atomic(combined, RESULTS_LOG_FILE)
    return combined


def _empirical_win_context(
    score: float,
    side: str,
    results: pd.DataFrame,
) -> dict | None:
    """Return same-side historical win rate for the candidate's 10-point score band."""
    required = {"score", "score_version", "signal", "hold_sessions", "return_decimal"}
    if results.empty or not required.issubset(results.columns) or not np.isfinite(score):
        return None

    history = results.copy()
    history["score"] = pd.to_numeric(history["score"], errors="coerce")
    history["hold_sessions"] = pd.to_numeric(history["hold_sessions"], errors="coerce")
    history["return_decimal"] = pd.to_numeric(history["return_decimal"], errors="coerce")
    band_index = min(max(int(score * 10), 0), 9)
    lower = band_index / 10.0
    upper = lower + 0.1
    upper_match = history["score"].le(upper) if band_index == 9 else history["score"].lt(upper)
    matching = history[
        history["score"].ge(lower)
        & upper_match
        & history["score_version"].eq(RANK_SCORE_VERSION)
        & history["signal"].astype(str).str.upper().eq(str(side).upper())
        & history["hold_sessions"].eq(HOLD_DAYS)
    ].dropna(subset=["return_decimal"])
    if len(matching) < CALIBRATION_MIN_SAMPLES:
        return None

    return {
        "win_rate": float((matching["return_decimal"] > 0).mean()),
        "samples": len(matching),
        "score_band": f"{lower:.1f}-{upper:.1f}",
    }


def record_signal(
    signal_type,
    sym,
    price,
    score=None,
    meta=None,
    *,
    signal_date: datetime.date | None = None,
):
    """Append a signal once per symbol, side, and trading date.

    The scanner revisits a symbol throughout the day. Persisted de-duplication
    prevents repeated alerts after a process restart as well as within one run.
    """
    global _RECORDED_SIGNAL_KEYS

    date = (signal_date or datetime.date.today()).isoformat()
    key = _signal_key(date, signal_type, sym)
    data = {
        "date": date,
        "signal": signal_type,
        "symbol": sym,
        "decision_price": float(price),
        # Retain `price` for compatibility with existing logs and reports.
        "price": float(price),
        "score": float(score) if score is not None else np.nan,
        "score_version": RANK_SCORE_VERSION,
        # Retain the historical column while deployed logs transition to `score`.
        "confidence": float(score) if score is not None else np.nan,
        "setup": (meta or {}).get("setup", ""),
        "trend4h_slope": (meta or {}).get("trend4h_slope", np.nan),
        "market_score": (meta or {}).get("market_score", np.nan),
        "sector_score": (meta or {}).get("sector_score", np.nan),
        "rs_score": (meta or {}).get("rs_score", np.nan),
        "technical_votes": (meta or {}).get("technical_votes", np.nan),
        "technical_vote_count": (meta or {}).get("technical_vote_count", np.nan),
        "entry_rule": (meta or {}).get("entry_rule", "NEXT_SESSION_OPEN"),
        "universe_rank": (meta or {}).get("universe_rank", np.nan),
    }
    # Persist the decomposition, not just the composite, so later iterations can
    # audit and ablate rank inputs without attempting to reconstruct live context.
    for name, value in (meta or {}).get("score_components", {}).items():
        data[f"component_{name}"] = value
    for name, passed in (meta or {}).get("technical_evidence", {}).items():
        data[f"evidence_{name}"] = bool(passed)

    with SIGNAL_LOG_LOCK:
        if _RECORDED_SIGNAL_KEYS is None:
            try:
                existing = _read_signal_log()
                required = {"date", "signal", "symbol"}
                if len(existing.columns) and not required.issubset(existing.columns):
                    missing = sorted(required - set(existing.columns))
                    logging.error(f"Refusing to append to signal log missing columns: {missing}")
                    return False
                _RECORDED_SIGNAL_KEYS = {
                    _signal_key(row["date"], row["signal"], row["symbol"])
                    for _, row in existing.iterrows()
                }
            except Exception as exc:
                logging.error(f"Could not load existing signal keys: {exc}")
                return False

        if key in _RECORDED_SIGNAL_KEYS:
            return False

        log_directory = os.path.dirname(os.path.abspath(LOG_FILE))
        os.makedirs(log_directory, exist_ok=True)
        existing = _read_signal_log()
        output_columns = list(data)
        if len(existing.columns):
            output_columns = list(dict.fromkeys([*existing.columns, *data]))
        existing = existing.reindex(columns=output_columns)
        row = {column: data.get(column, np.nan) for column in output_columns}
        updated = pd.concat(
            [existing, pd.DataFrame([row], columns=output_columns)],
            ignore_index=True,
        )
        _write_csv_atomic(updated, LOG_FILE)
        _RECORDED_SIGNAL_KEYS.add(key)
        return True


def evaluate_old_signals():
    try:
        with SIGNAL_LOG_LOCK:
            df = _read_signal_log()
    except Exception as e:
        logging.error(f"Failed to read log file '{LOG_FILE}': {e}")
        return None

    if df.empty:
        return None

    required_columns = {"date", "signal", "symbol"}
    if not required_columns.issubset(df.columns):
        logging.error(
            f"Signal log is missing columns: {sorted(required_columns - set(df.columns))}"
        )
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["date", "signal", "symbol"])

    cutoff = datetime.date.today() - datetime.timedelta(days=HOLD_DAYS)
    eligible = df[df["date"].dt.date <= cutoff].copy()
    if eligible.empty:
        return None

    symbols = sorted(set(normalize_tickers(eligible["symbol"].astype(str))))
    history_map = _download_batch_chunked(
        symbols,
        period=BACKTEST_PERIOD,
        interval="1d",
        label="backtest",
    )

    results = []
    result_records = []
    evaluated_keys = set()
    hold_bars = max(HOLD_DAYS, 1)
    evaluated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    for _, row in eligible.iterrows():
        sym = str(row["symbol"])
        side = str(row["signal"])
        signal_date = row["date"].date()
        logged_score = to_float(row.get("score", np.nan))
        if not np.isfinite(logged_score):
            logged_score = to_float(row.get("confidence", np.nan))

        try:
            df_hist = history_map.get(sym)
            if df_hist is None or df_hist.empty or not {"Open", "Close"}.issubset(df_hist.columns):
                continue

            history = df_hist.copy()
            history.index = pd.to_datetime(history.index, errors="coerce")
            history = history[~history.index.isna()].sort_index()
            future_bars = history[[index.date() > signal_date for index in history.index]]
            if len(future_bars) < hold_bars:
                continue

            entry = to_float(future_bars["Open"].iloc[0])
            exit_price = to_float(future_bars["Close"].iloc[hold_bars - 1])
            if not np.isfinite(entry) or entry <= 0 or not np.isfinite(exit_price):
                continue

            ret = (exit_price - entry) / entry if entry else 0.0
            if side.upper() == "SELL":
                ret = -ret
            profit = ret * CAPITAL_PER_TRADE
            entry_date = pd.Timestamp(future_bars.index[0]).date().isoformat()
            exit_date = pd.Timestamp(future_bars.index[hold_bars - 1]).date().isoformat()
            results.append(
                (
                    sym,
                    side,
                    float(entry),
                    float(exit_price),
                    float(profit),
                    logged_score,
                    row.get("setup", ""),
                )
            )
            result_records.append(
                {
                    "evaluated_at": evaluated_at,
                    "signal_date": signal_date.isoformat(),
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "signal": side.upper(),
                    "symbol": sym.upper(),
                    "decision_price": to_float(row.get("decision_price", row.get("price", np.nan))),
                    "entry_price": float(entry),
                    "exit_price": float(exit_price),
                    "hold_sessions": hold_bars,
                    "return_decimal": float(ret),
                    "profit_usd": float(profit),
                    "score": logged_score,
                    "score_version": row.get("score_version", "legacy"),
                    "universe_rank": to_float(row.get("universe_rank", np.nan)),
                    "setup": row.get("setup", ""),
                }
            )
            evaluated_keys.add(_signal_key(signal_date, side, sym))
        except Exception as e:
            logging.error(f"Backtest {sym}: {e}")

    if not results:
        return None

    results_history = pd.DataFrame()
    try:
        with SIGNAL_LOG_LOCK:
            results_history = _persist_signal_results(result_records)
            current = _read_signal_log()
            if not current.empty and required_columns.issubset(current.columns):
                keep = [
                    _signal_key(row["date"], row["signal"], row["symbol"]) not in evaluated_keys
                    for _, row in current.iterrows()
                ]
                _write_csv_atomic(current.loc[keep], LOG_FILE)
                if _RECORDED_SIGNAL_KEYS is not None:
                    _RECORDED_SIGNAL_KEYS.difference_update(evaluated_keys)
    except Exception as e:
        # Pending signals remain available for a safe retry if the permanent
        # outcome ledger or queue update cannot be completed.
        logging.error(f"Failed to persist evaluated signals: {e}")

    total = sum(p[4] for p in results)
    winrate = sum(1 for p in results if p[4] > 0) / len(results) * 100
    avg_trade = total / len(results)
    finite_scores = [p[5] for p in results if np.isfinite(to_float(p[5]))]
    average_score = float(np.mean(finite_scores)) if finite_scores else np.nan

    msg = "**Weekly Performance Report (next-open entries)**\n"
    for sym, side, entry, exitp, prof, score, setup in results:
        emoji = "WIN" if prof > 0 else "LOSS"
        msg += f"{emoji} {sym} {side} {entry:.2f} -> {exitp:.2f} | {prof:+.2f} USD | Score {score:.3f} | {setup}\n"
    msg += f"\nTotal: {total:+.2f} USD | Winrate: {winrate:.1f}% | Avg: {avg_trade:+.2f} USD/trade | AvgScore: {average_score:.3f}"
    if not results_history.empty:
        msg += f" | Historical outcomes: {len(results_history)}"

    return msg


# ----------------------------------------------------------------------
# Universe
# ----------------------------------------------------------------------


def _build_universe():
    extra = normalize_tickers([x.strip() for x in EXTRA_TICKERS.split(",") if x.strip()])

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


def _scan_segment(tickers, offset, *, log_prefix=""):
    """Scan configured batches and return candidates for universe-wide ranking."""
    candidates = []
    completed_cycle = False
    batches = max(1, BATCHES_PER_LOOP)

    for index in range(batches):
        batch_candidates, offset, batch_completed = scan_tickers_batched(
            tickers,
            offset=offset,
            batch_size=SCAN_BATCH_SIZE,
        )
        completed_cycle = completed_cycle or batch_completed
        candidates.extend(batch_candidates)

        if index < batches - 1 and not completed_cycle and BATCH_PAUSE > 0:
            time.sleep(BATCH_PAUSE)
        if completed_cycle:
            break

    if not candidates:
        logging.info(f"{log_prefix}No candidates in this pass segment")
    return candidates, offset, completed_cycle


def _send_ranked_alerts(candidates: list[SignalCandidate]) -> int:
    alerts = _finalize_ranked_candidates(candidates)
    if not alerts:
        logging.info("No new ranked alerts after de-duplication")
        return 0

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    message = f"**EMA Ranked Alerts ({now})**\n" + "\n".join(alerts)
    send_discord_message(message)
    return len(alerts)


def _send_performance_report():
    report = evaluate_old_signals()
    if report:
        send_discord_message(report)


def run_one_full_pass(*, lock_already_acquired=False):
    """Run one complete universe pass, normally from the authenticated endpoint."""
    if not lock_already_acquired and not MANUAL_SCAN_LOCK.acquire(blocking=False):
        logging.info("Manual scan already running. Skipping new request.")
        return False

    LAST_SCAN_SUMMARY["manual_scan_running"] = True
    try:
        validate_config()
        tickers = _build_universe()
        random.shuffle(tickers)
        logging.info(f"Manual run: total tickers to scan: {len(tickers)}")

        offset = 0
        completed_cycle = not tickers
        cycle_candidates = []
        while not completed_cycle:
            candidates, offset, completed_cycle = _scan_segment(
                tickers,
                offset,
                log_prefix="Manual run: ",
            )
            cycle_candidates.extend(candidates)
            if not completed_cycle and BATCH_PAUSE > 0:
                time.sleep(BATCH_PAUSE)

        _send_ranked_alerts(cycle_candidates)
        _send_performance_report()
        logging.info("Manual run: completed one full pass through the universe.")
        return True

    except Exception as e:
        logging.error(f"Manual run failed: {type(e).__name__} - {e}")
        traceback.print_exc()
        return False
    finally:
        LAST_SCAN_SUMMARY["manual_scan_running"] = False
        MANUAL_SCAN_LOCK.release()


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
    # Render needs the service bound beyond localhost for container ingress.
    app.run(host="0.0.0.0", port=port)  # nosec B104


@app.route("/run-once", methods=["POST"])
def run_once_endpoint():
    # Query-string tokens remain supported for existing Render URLs, but the
    # X-Run-Token header is preferred because URLs are commonly logged.
    token = request.headers.get("X-Run-Token") or request.args.get("token") or ""

    if not RUN_TOKEN or not hmac.compare_digest(token, RUN_TOKEN):
        return jsonify({"ok": False, "error": "unauthorized"}), 403

    if not MANUAL_SCAN_LOCK.acquire(blocking=False):
        return jsonify({"ok": True, "status": "already running"}), 202

    try:
        Thread(
            target=run_one_full_pass,
            kwargs={"lock_already_acquired": True},
            daemon=True,
        ).start()
    except Exception:
        MANUAL_SCAN_LOCK.release()
        raise
    return jsonify({"ok": True, "status": "started"}), 202


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    try:
        validate_config()
        tickers = _build_universe()
        logging.info(f"Total tickers to scan: {len(tickers)}")
        logging.info("EMA Multi-Factor Precision Scanner Started")
        logging.info(f"Daily interval: {TIMEFRAME_DAILY}")
        logging.info(f"4H base download interval: {TIMEFRAME_4H_BASE}")
        logging.info(f"4H resample rule: {RESAMPLE_4H_RULE}")
        logging.info(f"4H resample offset: {RESAMPLE_4H_OFFSET}")
        logging.info(f"Trend period for 1H history: {TREND_PERIOD}")
        send_discord_message("Bot online and ranking completed-session continuation setups.")
    except Exception as e:
        logging.error(f"Failed to build ticker universe: {e}")
        raise

    random.shuffle(tickers)
    LAST_SCAN_SUMMARY["universe_size"] = len(tickers)

    offset = 0
    cycle_candidates = []

    while True:
        completed_cycle = False
        try:
            candidates, offset, completed_cycle = _scan_segment(tickers, offset)
            cycle_candidates.extend(candidates)
            if completed_cycle:
                _send_ranked_alerts(cycle_candidates)
                cycle_candidates = []
                _send_performance_report()

        except Exception as e:
            LAST_SCAN_SUMMARY["last_error"] = f"{type(e).__name__}: {e}"
            logging.error(f"Global loop error: {type(e).__name__} - {e}")
            traceback.print_exc()

        if RUN_ONCE and completed_cycle:
            logging.info("Completed one full pass through the universe (RUN_ONCE=1). Exiting.")
            break

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
        raise
