"""
LSE (London Strategic Edge) data provider for the EMA scanner.

Wraps the `lse-data` Python SDK to provide:
- OHLCV candles at 14 resolutions (replaces Yahoo Finance)
- Economic calendar (earnings/events filter)
- Insider trades (replaces OpenInsider)
- Fundamentals, financial reports, company profiles
- Bond yields, macro series (regime context)
- Options chain/flow (future enhancement)
"""

from __future__ import annotations

import datetime
import logging
import os
from functools import lru_cache
from typing import Any

import pandas as pd

try:
    from lse import LSE, LSEError
    LSE_AVAILABLE = True
except Exception:
    LSE_AVAILABLE = False
    LSE = None
    LSEError = Exception

logger = logging.getLogger(__name__)

_LSE_CLIENT = None
_LSE_KEY = os.getenv("LSE_API_KEY", "").strip() or os.getenv("TRADE_DISCORD_WEBHOOK", "").strip()


def _get_client() -> LSE:
    global _LSE_CLIENT
    if not LSE_AVAILABLE:
        raise RuntimeError("lse-data package not installed. pip install lse-data")
    if _LSE_CLIENT is None:
        key = _LSE_KEY or os.getenv("LSE_API_KEY", "").strip()
        if not key:
            raise RuntimeError("LSE_API_KEY not set in environment")
        _LSE_CLIENT = LSE(api_key=key)
    return _LSE_CLIENT


def _to_dataframe(rows: list[dict[str, Any]], timestamp_col: str = "timestamp") -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
        df = df.set_index(timestamp_col).sort_index()
    return df


def _normalize_symbol(sym: str) -> str:
    sym = sym.strip().upper()
    if "/" not in sym and sym.isalpha() and len(sym) <= 5:
        return sym
    return sym


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def is_available() -> bool:
    return LSE_AVAILABLE and bool(_LSE_KEY)


def get_candles(
    symbol: str,
    timeframe: str = "1d",
    start: str | None = None,
    end: str | None = None,
    limit: int = 0,
) -> pd.DataFrame:
    """
    Fetch OHLCV candles from LSE vault.

    timeframe: 1s, 5s, 15s, 30s, 1m, 3m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1mo
    """
    if not is_available():
        return pd.DataFrame()

    client = _get_client()
    sym = _normalize_symbol(symbol)
    try:
        kwargs = {"start": start} if start else {}
        if end:
            kwargs["end"] = end
        if limit:
            kwargs["limit"] = limit
        rows = client.candles(sym, timeframe, **kwargs)
        df = _to_dataframe(rows)
        if df.empty:
            return pd.DataFrame()
        df = df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except LSEError as e:
        logger.warning(f"LSE candles failed for {symbol} {timeframe}: {e.status} {e.message}")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"LSE candles error for {symbol}: {e}")
        return pd.DataFrame()


@lru_cache(maxsize=32)
def get_daily_candles(symbol: str, start: str | None = None, end: str | None = None, limit: int = 0) -> pd.DataFrame:
    """Convenience: daily candles (1d) — cached."""
    return get_candles(symbol, "1d", start, end, limit)


@lru_cache(maxsize=32)
def get_hourly_candles(symbol: str, start: str | None = None, end: str | None = None, limit: int = 0) -> pd.DataFrame:
    """Convenience: hourly candles (1h) — cached."""
    return get_candles(symbol, "1h", start, end, limit)


@lru_cache(maxsize=32)
def get_4h_candles(symbol: str, start: str | None = None, end: str | None = None, limit: int = 0) -> pd.DataFrame:
    """Convenience: 4-hour candles (4h) — cached."""
    return get_candles(symbol, "4h", start, end, limit)


def get_economic_calendar(
    region: str = "US",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Economic calendar events (earnings, macro releases, etc.)."""
    if not is_available():
        return pd.DataFrame()
    client = _get_client()
    try:
        kwargs = {}
        if start:
            kwargs["start"] = start
        if end:
            kwargs["end"] = end
        rows = client.economic_calendar(region=region, **kwargs)
        df = _to_dataframe(rows, timestamp_col="datetime")
        return df
    except LSEError as e:
        logger.warning(f"LSE economic_calendar failed: {e.status} {e.message}")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"LSE economic_calendar error: {e}")
        return pd.DataFrame()


def get_insider_trades(symbol: str, type: str | None = None) -> pd.DataFrame:
    """Insider trades from LSE (replaces OpenInsider)."""
    if not is_available():
        return pd.DataFrame()
    client = _get_client()
    try:
        rows = client.insider_trades(symbol, type=type)
        return _to_dataframe(rows, timestamp_col="datetime")
    except LSEError as e:
        logger.warning(f"LSE insider_trades failed for {symbol}: {e.status} {e.message}")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"LSE insider_trades error for {symbol}: {e}")
        return pd.DataFrame()


def get_fundamentals(symbol: str) -> dict:
    """Company fundamentals snapshot."""
    if not is_available():
        return {}
    client = _get_client()
    try:
        rows = client.fundamentals(symbol)
        return rows[0] if rows else {}
    except LSEError as e:
        logger.warning(f"LSE fundamentals failed for {symbol}: {e.status} {e.message}")
        return {}
    except Exception as e:
        logger.warning(f"LSE fundamentals error for {symbol}: {e}")
        return {}


def get_financial_reports(
    symbol: str, report_type: str = "income", period: str = "FY"
) -> list[dict]:
    """Financial reports (income, balance, cash flow)."""
    if not is_available():
        return []
    client = _get_client()
    try:
        return client.financial_reports(symbol, report_type=report_type, period=period)
    except LSEError as e:
        logger.warning(f"LSE financial_reports failed for {symbol}: {e.status} {e.message}")
        return []
    except Exception as e:
        logger.warning(f"LSE financial_reports error for {symbol}: {e}")
        return []


def get_company_profile(symbol: str) -> dict:
    """Company profile (sector, industry, employees, etc.)."""
    if not is_available():
        return {}
    client = _get_client()
    try:
        rows = client.company_profiles(symbol)
        return rows[0] if rows else {}
    except LSEError as e:
        logger.warning(f"LSE company_profiles failed for {symbol}: {e.status} {e.message}")
        return {}
    except Exception as e:
        logger.warning(f"LSE company_profiles error for {symbol}: {e}")
        return {}


def get_bond_yields(series: str = "US10Y", start: str | None = None) -> pd.DataFrame:
    """Government bond yield series."""
    if not is_available():
        return pd.DataFrame()
    client = _get_client()
    try:
        kwargs = {"start": start} if start else {}
        rows = client.bond_yields(series, **kwargs)
        return _to_dataframe(rows, timestamp_col="date")
    except LSEError as e:
        logger.warning(f"LSE bond_yields failed for {series}: {e.status} {e.message}")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"LSE bond_yields error: {e}")
        return pd.DataFrame()


def get_macro_series(series_id: str, start: str | None = None) -> pd.DataFrame:
    """Generic macro economic series."""
    if not is_available():
        return pd.DataFrame()
    client = _get_client()
    try:
        kwargs = {"start": start} if start else {}
        rows = client.series(series_id, **kwargs)
        return _to_dataframe(rows, timestamp_col="date")
    except LSEError as e:
        logger.warning(f"LSE series failed for {series_id}: {e.status} {e.message}")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"LSE series error: {e}")
        return pd.DataFrame()


def get_economic_series(series_name: str, start: str | None = None) -> pd.DataFrame:
    """Named macro series (e.g., 'cpi_yoy', 'fdtr')."""
    if not is_available():
        return pd.DataFrame()
    client = _get_client()
    try:
        kwargs = {"start": start} if start else {}
        rows = client.economics(series_name, **kwargs)
        return _to_dataframe(rows, timestamp_col="date")
    except LSEError as e:
        logger.warning(f"LSE economics failed for {series_name}: {e.status} {e.message}")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"LSE economics error: {e}")
        return pd.DataFrame()


def get_catalog(category: str | None = None) -> pd.DataFrame:
    """Instrument catalog (list of available symbols)."""
    if not is_available():
        return pd.DataFrame()
    client = _get_client()
    try:
        rows = client.catalog(category) if category else client.catalog()
        return pd.DataFrame(rows)
    except LSEError as e:
        logger.warning(f"LSE catalog failed: {e.status} {e.message}")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"LSE catalog error: {e}")
        return pd.DataFrame()


# ----------------------------------------------------------------------
# Scanner integration helpers
# ----------------------------------------------------------------------


def prepare_lse_daily_df(symbol: str, lookback_days: int = 500) -> pd.DataFrame | None:
    """Fetch and prepare daily OHLCV from LSE for scanner (replaces Yahoo daily)."""
    end = datetime.date.today().isoformat()
    start = (datetime.date.today() - datetime.timedelta(days=lookback_days)).isoformat()
    df = get_daily_candles(symbol, start=start, end=end)
    if df.empty or len(df) < 30:
        return None
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    return df


def prepare_lse_hourly_df(symbol: str, lookback_days: int = 730) -> pd.DataFrame | None:
    """Fetch hourly OHLCV from LSE for 4H resampling."""
    end = datetime.date.today().isoformat()
    start = (datetime.date.today() - datetime.timedelta(days=lookback_days)).isoformat()
    df = get_hourly_candles(symbol, start=start, end=end)
    if df.empty or len(df) < 100:
        return None
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    return df


def get_lse_earnings_date(
    symbol: str, days_ahead: int = 45, days_behind: int = 5
) -> datetime.date | None:
    """Get nearest earnings date from LSE economic calendar."""
    if not is_available():
        return None
    client = _get_client()
    try:
        today = datetime.date.today()
        start = (today - datetime.timedelta(days=days_behind)).isoformat()
        end = (today + datetime.timedelta(days=days_ahead)).isoformat()
        rows = _get_client().economic_calendar(region="US", start=start, end=end)
        symbol_lower = symbol.lower()
        for r in rows:
            event = r.get("event", "").lower()
            if symbol_lower in event and "earnings" in event:
                dt = pd.to_datetime(r.get("datetime"), utc=True, errors="coerce")
                if not pd.isna(dt):
                    return dt.date()
    except Exception as e:
        logger.debug(f"LSE earnings lookup failed for {symbol}: {e}")
    return None


def get_lse_insider_signal(symbol: str) -> dict:
    """Get insider signal for biotech gate / BUY confirmation."""
    df = get_insider_trades(symbol, type="P-Purchase")
    if df.empty:
        return {"available": True, "bullish_buy_cluster": False, "bullish_score": 0.0}
    # Simple signal: count of purchases in last 30 days
    recent = df[df.index >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)]
    count = len(recent)
    return {
        "available": True,
        "bullish_buy_cluster": count >= 2,
        "bullish_score": min(count / 5.0, 1.0),
        "recent_buy_count": count,
    }


def get_lse_earnings_dates(
    symbol: str, days_ahead: int = 45, days_behind: int = 5
) -> list[datetime.date]:
    """Get all earnings dates near current date from LSE calendar."""
    if not is_available():
        return []
    client = _get_client()
    try:
        today = datetime.date.today()
        start = (today - datetime.timedelta(days=days_behind)).isoformat()
        end = (today + datetime.timedelta(days=days_ahead)).isoformat()
        rows = client.economic_calendar(region="US", start=start, end=end)
        dates = []
        sym_lower = symbol.lower()
        for r in rows:
            event = r.get("event", "").lower()
            if sym_lower in event and "earnings" in event:
                dt = pd.to_datetime(r.get("datetime"), utc=True, errors="coerce")
                if not pd.isna(dt):
                    dates.append(dt.date())
        return dates
    except Exception as e:
        logger.debug(f"LSE earnings lookup failed for {symbol}: {e}")
        return []


def get_lse_macro_regime() -> dict:
    """Fetch key macro series for regime scoring."""
    if not is_available():
        return {"us10y": pd.DataFrame(), "cpi": pd.DataFrame(), "ffr": pd.DataFrame()}
    return {
        "us10y": get_bond_yields("US10Y", start=(datetime.date.today() - datetime.timedelta(days=730)).isoformat()),
        "cpi": get_economic_series("cpi_yoy", start=(datetime.date.today() - datetime.timedelta(days=730)).isoformat()),
        "ffr": get_macro_series("fdtr", start=(datetime.date.today() - datetime.timedelta(days=730)).isoformat()),
    }


def build_lse_market_regime(spy_df: pd.DataFrame, macro: dict | None = None) -> dict:
    """Build market regime dict using LSE bond yields + SPY."""
    regime = {
        "label": "LSE_MARKET",
        "long_ok": False,
        "short_ok": False,
        "score_long": 0.0,
        "score_short": 0.0,
    }
    if spy_df is None or spy_df.empty:
        return regime
    try:
        close = float(spy_df["Close"].iloc[-1])
        ema50 = float(spy_df["Close"].ewm(span=50, adjust=False).mean().iloc[-1])
        ema200 = float(spy_df["Close"].ewm(span=200, adjust=False).mean().iloc[-1])
        slope50 = (ema50 / float(spy_df["Close"].ewm(span=50, adjust=False).mean().iloc[-6]) - 1.0) if len(spy_df) > 6 else 0.0
        slope200 = (float(spy_df["Close"].ewm(span=200, adjust=False).mean().iloc[-1]) / float(spy_df["Close"].ewm(span=200, adjust=False).mean().iloc[-21]) - 1.0) if len(spy_df) > 21 else 0.0

        # Add macro confirmation if available
        macro_score = 0.0
        if macro and "us10y" in macro and not macro["us10y"].empty:
            us10y = macro["us10y"]["close"].iloc[-1] if "close" in macro["us10y"].columns else 0
            macro_score = 1.0 if us10y < 4.5 else 0.5 if us10y < 5.0 else 0.0

        long_parts = [close > ema50, ema50 > ema200, slope50 > 0, slope200 >= 0, macro_score > 0.3]
        short_parts = [close < ema50, ema50 < ema200, slope50 < 0, slope200 <= 0, macro_score < 0.7]

        return {
            "label": "LSE_MARKET",
            "close": close,
            "ema50": ema50,
            "ema200": ema200,
            "slope50": slope50,
            "slope200": slope200,
            "long_ok": all(long_parts),
            "short_ok": all(short_parts),
            "score_long": round(sum(bool(x) for x in long_parts) / len(long_parts), 2),
            "score_short": round(sum(bool(x) for x in short_parts) / len(short_parts), 2),
            "macro_score": macro_score,
        }
    except Exception as e:
        logger.debug(f"LSE market regime failed: {e}")
        return {
            "label": "LSE_MARKET",
            "long_ok": False,
            "short_ok": False,
            "score_long": 0.0,
            "score_short": 0.0,
        }


__all__ = [
    "is_available",
    "get_candles",
    "get_daily_candles",
    "get_hourly_candles",
    "get_4h_candles",
    "get_economic_calendar",
    "get_insider_trades",
    "get_fundamentals",
    "get_financial_reports",
    "get_company_profile",
    "get_bond_yields",
    "get_macro_series",
    "get_economic_series",
    "get_catalog",
    "prepare_lse_daily_df",
    "prepare_lse_hourly_df",
    "get_lse_earnings_date",
    "get_lse_insider_signal",
    "get_lse_earnings_dates",
    "get_lse_macro_regime",
    "build_lse_market_regime",
]