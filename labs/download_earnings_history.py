"""Download reported earnings dates from Nasdaq for the cached stock universe."""

from __future__ import annotations

import concurrent.futures
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

import backtest_2026 as bt

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "labs" / "cache" / "august_2026"
RAW_DIR = CACHE_DIR / "nasdaq_earnings"
OUTPUT = CACHE_DIR / "earnings_dates.csv"


def _fetch(symbol: str) -> tuple[str, dict]:
    target = RAW_DIR / f"{symbol}.json"
    if target.exists():
        return symbol, json.loads(target.read_text(encoding="utf-8"))
    encoded = urllib.parse.quote(symbol.replace("-", "."))
    url = f"https://api.nasdaq.com/api/company/{encoded}/earnings-surprise"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://www.nasdaq.com",
            "Referer": "https://www.nasdaq.com/market-activity/earnings",
        },
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = json.load(response)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            return symbol, payload
        except (  # noqa: PERF203 - retries require one network exception boundary per attempt
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
        ) as exc:
            if attempt == 5:
                raise RuntimeError(f"{symbol}: {exc}") from exc
            time.sleep(min(2**attempt, 20))
    raise AssertionError("unreachable")


def main() -> int:
    dataset = bt.load_dataset_cache(CACHE_DIR)
    rows = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_fetch, symbol) for symbol in dataset.symbols]
        for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                symbol, payload = future.result()
                table = ((payload.get("data") or {}).get("earningsSurpriseTable") or {}).get(
                    "rows"
                ) or []
                for record in table:
                    reported = pd.to_datetime(record.get("dateReported"), errors="coerce")
                    if pd.isna(reported):
                        continue
                    rows.append(
                        {
                            "symbol": symbol,
                            "earnings_date": reported.date().isoformat(),
                            "fiscal_quarter_end": record.get("fiscalQtrEnd"),
                            "eps": record.get("eps"),
                            "consensus_forecast": record.get("consensusForecast"),
                            "percentage_surprise": record.get("percentageSurprise"),
                            "source": "Nasdaq earnings surprise",
                        }
                    )
            except Exception as exc:
                failures.append(str(exc))
            if number % 50 == 0:
                print(f"earnings {number}/{len(futures)}; failures={len(failures)}", flush=True)
    frame = pd.DataFrame(rows).sort_values(["earnings_date", "symbol"])
    frame.to_csv(OUTPUT, index=False)
    coverage = {
        "requested_symbols": len(dataset.symbols),
        "symbols_with_dates": int(frame["symbol"].nunique()),
        "reported_dates": len(frame),
        "failures": failures,
    }
    (CACHE_DIR / "earnings_coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(coverage, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
