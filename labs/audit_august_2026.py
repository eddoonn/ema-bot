"""Independently audit the frozen August 2026 backtest ledger."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_2026 as bt
import ema_scanner as scanner

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "labs" / "cache" / "august_2026"
OUTPUT_DIR = ROOT / "labs" / "results" / "backtest_august_2026"
CONFIG = bt.BacktestConfig(start_date=dt.date(2026, 8, 1), end_date=dt.date(2026, 8, 26))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    dataset = bt.load_dataset_cache(CACHE_DIR)
    prepared = bt.prepare_daily_frames(dataset)
    with bt.scanner_settings(CONFIG):
        by_date = bt.collect_candidates(dataset, CONFIG, prepared)

    selected = []
    for signal_date, candidates in sorted(by_date.items()):
        ranked = sorted(candidates, key=lambda item: (-item.score, item.symbol, item.side))[
            : CONFIG.max_alerts_per_day
        ]
        for rank, item in enumerate(ranked, 1):
            frame = prepared[item.symbol]
            entry_position = item.position + 1
            exit_position = item.position + CONFIG.hold_days
            entry_date = bt._date_from_index(frame.index[entry_position]) if entry_position < len(frame) else None
            exit_date = bt._date_from_index(frame.index[exit_position]) if exit_position < len(frame) else None
            selected.append(
                {
                    "signal_date": signal_date.isoformat(),
                    "symbol": item.symbol,
                    "side": item.side,
                    "universe_rank": rank,
                    "score": item.score,
                    "adx": item.adx,
                    "entry_date": entry_date.isoformat() if entry_date else None,
                    "scheduled_exit_date": exit_date.isoformat() if exit_date else None,
                    "status": "matured" if exit_date and exit_date <= CONFIG.end_date else "pending",
                    "avg_dollar_vol_m": item.metadata.get("avg_dollar_vol_m"),
                    "trend4h_slope": item.metadata.get("trend4h_slope"),
                }
            )
    selected_frame = pd.DataFrame(selected)
    selected_frame.to_csv(OUTPUT_DIR / "selected_signals.csv", index=False)
    pending = selected_frame[selected_frame["status"] == "pending"].copy()
    pending.to_csv(OUTPUT_DIR / "pending_signals.csv", index=False)

    trades_path = OUTPUT_DIR / "trades_2026.csv"
    trades = pd.read_csv(trades_path)
    shutil.copyfile(trades_path, OUTPUT_DIR / "trades.csv")
    errors: list[str] = []
    if trades.duplicated(["signal_date", "symbol", "side"]).any():
        errors.append("duplicate trade key")
    if not (trades["score"] >= CONFIG.min_score - 1e-12).all():
        errors.append("score below threshold")
    if not (trades["adx"] >= CONFIG.adx_hard_min - 1e-12).all():
        errors.append("ADX below hard minimum")
    if not trades["avg_dollar_vol_m"].between(
        CONFIG.min_dollar_vol_m, CONFIG.max_dollar_vol_m
    ).all():
        errors.append("dollar volume outside configured bounds")
    if not (trades["universe_rank"].between(1, CONFIG.max_alerts_per_day)).all():
        errors.append("rank outside daily cap")
    if not (pd.to_datetime(trades["entry_date"]) > pd.to_datetime(trades["signal_date"])).all():
        errors.append("entry is not after signal")
    if not (pd.to_datetime(trades["exit_date"]) <= pd.Timestamp(CONFIG.end_date)).all():
        errors.append("unobservable exit included")

    matured = selected_frame[selected_frame["status"] == "matured"]
    expected_keys = set(map(tuple, matured[["signal_date", "symbol", "side"]].to_numpy()))
    actual_keys = set(map(tuple, trades[["signal_date", "symbol", "side"]].to_numpy()))
    if expected_keys != actual_keys:
        errors.append("matured selected signals do not match trade ledger")

    one_r = CONFIG.capital_per_trade * CONFIG.risk_fraction
    for row in trades.itertuples(index=False):
        frame = prepared[row.symbol]
        dates = [bt._date_from_index(value).isoformat() for value in frame.index]
        signal_position = dates.index(row.signal_date)
        expected_entry_date = dates[signal_position + 1]
        expected_exit_date = dates[signal_position + CONFIG.hold_days]
        expected_entry = float(frame["Open"].iloc[signal_position + 1])
        expected_exit = float(frame["Close"].iloc[signal_position + CONFIG.hold_days])
        expected_return = (expected_exit - expected_entry) / expected_entry
        if row.side == "SELL":
            expected_return *= -1
        checks = [
            row.entry_date == expected_entry_date,
            row.exit_date == expected_exit_date,
            np.isclose(row.entry_price, expected_entry, rtol=0, atol=1e-9),
            np.isclose(row.exit_price, expected_exit, rtol=0, atol=1e-9),
            np.isclose(row.return_decimal, expected_return, rtol=0, atol=1e-12),
            np.isclose(row.profit_usd, expected_return * CONFIG.capital_per_trade, rtol=0, atol=1e-9),
            np.isclose(row.r_multiple, expected_return * CONFIG.capital_per_trade / one_r, rtol=0, atol=1e-9),
        ]
        if not all(checks):
            errors.append(f"fill/P&L mismatch: {row.signal_date} {row.symbol} {row.side}")

    recomputed = bt.summarize_trades(trades)
    written_summary = json.loads((OUTPUT_DIR / "summary.json").read_text(encoding="utf-8"))
    for key in recomputed:
        left, right = recomputed[key], written_summary[key]
        if left is None or right is None:
            if left != right:
                errors.append(f"summary mismatch: {key}")
        elif not np.isclose(float(left), float(right), rtol=0, atol=1e-9):
            errors.append(f"summary mismatch: {key}")

    required = {scanner.MARKET_PROXY, *scanner.GICS_TO_ETF.values(), "XBI"}
    coverage = json.loads((CACHE_DIR / "coverage.json").read_text(encoding="utf-8"))
    if coverage["usable_stocks"] != coverage["requested_stocks"] or coverage["request_failures"]:
        errors.append("stock data coverage incomplete")
    errors.extend(
        f"missing proxy: {proxy}"
        for proxy in required
        if proxy not in dataset.daily or proxy not in dataset.hourly
    )
    for symbol in dataset.symbols:
        if bt._date_from_index(dataset.daily[symbol].index.max()) < CONFIG.end_date:
            errors.append(f"daily series ends early: {symbol}")
        hourly_day = bt._date_from_index(dataset.hourly[symbol].index.max())
        if hourly_day < CONFIG.end_date:
            errors.append(f"hourly series ends early: {symbol}")

    audit = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "universe_requested": coverage["requested_stocks"],
        "universe_usable": coverage["usable_stocks"],
        "required_proxies": sorted(required),
        "selected_signals": len(selected_frame),
        "matured_trades": len(trades),
        "pending_signals": len(pending),
        "cache_sha256": _sha256(CACHE_DIR / bt.CACHE_FILE_NAME),
        "ledger_sha256": _sha256(OUTPUT_DIR / "trades.csv"),
        "checks": [
            "full current S&P 500 data coverage",
            "all required market and sector proxies present",
            "score, ADX, liquidity, and daily-rank limits",
            "next-session-open entries",
            "fifth-subsequent-session close exits",
            "no exits after the data cutoff",
            "BUY/SELL return, USD P&L, and R arithmetic",
            "independent aggregate summary",
        ],
    }
    (OUTPUT_DIR / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
