"""Build a reproducible August 2026 backtest cache from Yahoo chart JSON.

This bypasses yfinance's crumb endpoint (which can be rate limited) while using
the same unadjusted OHLCV bars returned by Yahoo's public chart endpoint.
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

import backtest_2026 as bt
import ema_scanner as scanner

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "labs" / "cache" / "august_2026"
RAW_DIR = CACHE_DIR / "raw_json"
PROXIES = {scanner.MARKET_PROXY, *scanner.GICS_TO_ETF.values(), "XBI"}
PERIOD1 = int(dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc).timestamp())
PERIOD2 = int(dt.datetime(2026, 8, 28, tzinfo=dt.timezone.utc).timestamp())


def _url(symbol: str, interval: str) -> str:
    query = urllib.parse.urlencode(
        {
            "period1": PERIOD1
            if interval == "1d"
            else int(dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc).timestamp()),
            "period2": PERIOD2,
            "interval": interval,
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    return f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}?{query}"


def _download(symbol: str, interval: str) -> tuple[str, str, dict]:
    # The strategy needs 205 completed 4H bars just for warm-up. Keep the
    # extended-hourly snapshot separate so an older short-history cache cannot
    # be mistaken for a valid response on a resumed run.
    folder = "1h_extended" if interval == "1h" else interval
    target = RAW_DIR / folder / f"{symbol}.json"
    if target.exists():
        return symbol, interval, json.loads(target.read_text(encoding="utf-8"))
    request = urllib.request.Request(_url(symbol, interval), headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(7):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = json.load(response)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            return symbol, interval, payload
        except (  # noqa: PERF203 - retries require one exception boundary per attempt
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
        ) as exc:
            if attempt == 6:
                raise RuntimeError(f"{symbol} {interval}: {exc}") from exc
            time.sleep(min(2**attempt, 20))
    raise AssertionError("unreachable")


def _frame(payload: dict, interval: str) -> pd.DataFrame:
    chart = payload.get("chart", {})
    if chart.get("error") or not chart.get("result"):
        return pd.DataFrame()
    result = chart["result"][0]
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    if not timestamps:
        return pd.DataFrame()
    index = pd.to_datetime(timestamps, unit="s", utc=True)
    if interval == "1d":
        timezone = result.get("meta", {}).get("exchangeTimezoneName", "America/New_York")
        index = index.tz_convert(timezone).tz_localize(None).normalize()
    values = {
        "Open": quote.get("open", []),
        "High": quote.get("high", []),
        "Low": quote.get("low", []),
        "Close": quote.get("close", []),
        "Volume": quote.get("volume", []),
    }
    frame = pd.DataFrame(values, index=index)
    adjusted = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    if adjusted and len(adjusted) == len(frame):
        frame["Adj Close"] = adjusted
    return frame[~frame.index.duplicated(keep="last")].sort_index().dropna(subset=["Close"])


def main() -> int:
    symbols = sorted(set(scanner.get_sp500_tickers()))
    sector_proxies = {
        symbol: scanner.infer_sector_proxy(symbol)
        for symbol in symbols
        if scanner.infer_sector_proxy(symbol)
    }
    downloads = [
        (symbol, interval) for symbol in sorted(set(symbols) | PROXIES) for interval in ("1d", "1h")
    ]
    results: dict[tuple[str, str], dict] = {}
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_download, *item): item for item in downloads}
        for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                got_symbol, got_interval, payload = future.result()
                results[(got_symbol, got_interval)] = payload
            except Exception as exc:  # retain every failure for the coverage audit
                failures.append(str(exc))
            if number % 100 == 0:
                print(f"downloaded {number}/{len(downloads)}; failures={len(failures)}", flush=True)

    daily = {
        symbol: _frame(results[(symbol, "1d")], "1d")
        for symbol, _ in downloads
        if (symbol, "1d") in results
    }
    hourly = {
        symbol: _frame(results[(symbol, "1h")], "1h")
        for symbol, _ in downloads
        if (symbol, "1h") in results
    }
    usable = [
        symbol
        for symbol in symbols
        if not daily.get(symbol, pd.DataFrame()).empty
        and not hourly.get(symbol, pd.DataFrame()).empty
    ]
    dataset = bt.MarketDataset(
        symbols=usable,
        daily=daily,
        hourly=hourly,
        sector_proxies={
            symbol: proxy for symbol, proxy in sector_proxies.items() if symbol in usable
        },
        downloaded_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    cache_path = bt.save_dataset_cache(dataset, CACHE_DIR)
    coverage = {
        "requested_stocks": len(symbols),
        "usable_stocks": len(usable),
        "missing_stocks": sorted(set(symbols) - set(usable)),
        "required_proxies": sorted(PROXIES),
        "missing_daily_proxies": sorted(
            proxy for proxy in PROXIES if daily.get(proxy, pd.DataFrame()).empty
        ),
        "missing_hourly_proxies": sorted(
            proxy for proxy in PROXIES if hourly.get(proxy, pd.DataFrame()).empty
        ),
        "request_failures": failures,
        "daily_latest": {
            key: str(value.index.max()) for key, value in daily.items() if not value.empty
        },
        "hourly_latest": {
            key: str(value.index.max()) for key, value in hourly.items() if not value.empty
        },
    }
    (CACHE_DIR / "coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in coverage.items()
                if key not in {"daily_latest", "hourly_latest"}
            },
            indent=2,
        )
    )
    print(cache_path)
    return 0 if not failures and len(usable) == len(symbols) else 2


if __name__ == "__main__":
    raise SystemExit(main())
