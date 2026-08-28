"""Create and audit the final walk-forward optimization report."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_2026 as bt
from labs.optimize_strategy_2026 import (
    BASELINE,
    CACHE_DIR,
    END,
    OUTPUT_DIR,
    PERIODS,
    START,
    apply_variant,
    summarize,
)

RECOMMENDED_HOLD = 7
RECOMMENDED = dataclasses.replace(
    BASELINE,
    volume_min_m=100.0,
    volume_max_m=750.0,
    gap_atr_max=1.2,
)


def _with_hold(
    ledger: pd.DataFrame,
    prepared: dict[str, pd.DataFrame],
    hold: int,
) -> pd.DataFrame:
    index_maps = {
        symbol: {
            bt._date_from_index(value).isoformat(): position
            for position, value in enumerate(frame.index)
        }
        for symbol, frame in prepared.items()
    }
    records = []
    for row in ledger.itertuples(index=False):
        frame = prepared[row.symbol]
        signal_position = index_maps[row.symbol][row.signal_date]
        exit_position = signal_position + hold
        if exit_position >= len(frame):
            continue
        exit_date = bt._date_from_index(frame.index[exit_position]).isoformat()
        exit_price = float(frame["Close"].iloc[exit_position])
        direction = 1.0 if row.side == "BUY" else -1.0
        result = (exit_price / float(row.entry_price) - 1.0) * direction
        record = row._asdict()
        record.update(
            {
                "exit_date": exit_date,
                "exit_price": exit_price,
                "return_decimal": result,
                "profit_usd": result * 500.0,
                "r_multiple": result * 100.0,
                "hold_sessions": hold,
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def _summary_row(strategy: str, period: str, trades: pd.DataFrame) -> dict:
    result = summarize(trades)
    return {
        "strategy": strategy,
        "period": period,
        **result,
        "pnl_usd": result["total_r"] * 5.0,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(OUTPUT_DIR / "candidate_ledger.csv")
    dataset = bt.load_dataset_cache(CACHE_DIR)
    prepared = bt.prepare_daily_frames(dataset)
    baseline_ledger = _with_hold(raw, prepared, 5)
    recommended_ledger = _with_hold(raw, prepared, RECOMMENDED_HOLD)

    comparison_rows = []
    for period, bounds in PERIODS.items():
        baseline = apply_variant(baseline_ledger, BASELINE, *bounds)
        recommended = apply_variant(recommended_ledger, RECOMMENDED, *bounds)
        baseline.to_csv(OUTPUT_DIR / f"final_trades_baseline_{period}.csv", index=False)
        recommended.to_csv(OUTPUT_DIR / f"final_trades_recommended_{period}.csv", index=False)
        comparison_rows.extend(
            [
                _summary_row("baseline_5_session", period, baseline),
                _summary_row("recommended_7_session", period, recommended),
            ]
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUT_DIR / "final_walk_forward_comparison.csv", index=False)

    hold_rows = []
    for hold in range(2, 11):
        held = _with_hold(raw, prepared, hold)
        for period, bounds in PERIODS.items():
            trades = apply_variant(held, RECOMMENDED, *bounds)
            hold_rows.append(
                {"hold_sessions": hold, **_summary_row("liquidity_filter", period, trades)}
            )
    pd.DataFrame(hold_rows).to_csv(OUTPUT_DIR / "hold_period_sweep.csv", index=False)

    recommended_all = apply_variant(recommended_ledger, RECOMMENDED, START, END)
    recommended_all.to_csv(OUTPUT_DIR / "final_recommended_trades_all.csv", index=False)
    errors = []
    if recommended_all.duplicated(["signal_date", "symbol", "side"]).any():
        errors.append("duplicate trade key")
    if not (recommended_all["score"] >= 0.40).all():
        errors.append("score below 0.40")
    if not (recommended_all["adx"] >= 18.0).all():
        errors.append("ADX below 18")
    if not recommended_all["avg_dollar_vol_m"].between(100.0, 750.0).all():
        errors.append("liquidity outside $100M-$750M")
    if not (recommended_all["signal_gap_atr"] <= 1.2).all():
        errors.append("signal gap above 1.2 ATR")
    if not (recommended_all["hold_sessions"] == RECOMMENDED_HOLD).all():
        errors.append("wrong hold period")

    for row in recommended_all.itertuples(index=False):
        frame = prepared[row.symbol]
        date_map = {
            bt._date_from_index(value).isoformat(): position
            for position, value in enumerate(frame.index)
        }
        position = date_map[row.signal_date]
        expected_entry_date = bt._date_from_index(frame.index[position + 1]).isoformat()
        expected_exit_date = bt._date_from_index(
            frame.index[position + RECOMMENDED_HOLD]
        ).isoformat()
        expected_entry = float(frame["Open"].iloc[position + 1])
        expected_exit = float(frame["Close"].iloc[position + RECOMMENDED_HOLD])
        direction = 1.0 if row.side == "BUY" else -1.0
        expected_return = (expected_exit / expected_entry - 1.0) * direction
        if not all(
            [
                row.entry_date == expected_entry_date,
                row.exit_date == expected_exit_date,
                np.isclose(row.entry_price, expected_entry, rtol=0, atol=1e-9),
                np.isclose(row.exit_price, expected_exit, rtol=0, atol=1e-9),
                np.isclose(row.return_decimal, expected_return, rtol=0, atol=1e-12),
                np.isclose(row.r_multiple, expected_return * 100.0, rtol=0, atol=1e-9),
            ]
        ):
            errors.append(f"fill mismatch: {row.signal_date} {row.symbol} {row.side}")

    earnings = pd.read_csv(CACHE_DIR / "earnings_dates.csv")
    august_baseline = pd.read_csv(OUTPUT_DIR / "final_trades_baseline_august_posthoc.csv")
    earnings_august = august_baseline[august_baseline["earnings_window_3_before_2_after"]]
    earnings_august.to_csv(OUTPUT_DIR / "august_earnings_sensitivity.csv", index=False)
    audit = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "current_sp500_constituents": len(dataset.symbols),
        "earnings_symbols_with_reported_dates": int(earnings["symbol"].nunique()),
        "candidate_rows": len(raw),
        "recommended_trades_all_periods": len(recommended_all),
        "recommended_ledger_sha256": _sha256(OUTPUT_DIR / "final_recommended_trades_all.csv"),
        "market_cache_sha256": _sha256(CACHE_DIR / bt.CACHE_FILE_NAME),
        "checks": [
            "next-session-open entry",
            "seventh-subsequent-session close exit",
            "BUY/SELL return direction",
            "score and ADX floors",
            "$100M-$750M average-dollar-volume band",
            "1.2 ATR point-in-time signal-gap guard",
            "no duplicate trade keys",
        ],
    }
    (OUTPUT_DIR / "final_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "audit": audit,
        "recommended_defaults": {
            "hold_sessions": 7,
            "min_average_dollar_volume_m": 100,
            "max_average_dollar_volume_m": 750,
            "earnings_days_before": 3,
            "earnings_days_after": 0,
            "score_min": 0.40,
            "adx_hard_min": 18,
            "four_hour_slope_min": 0.003,
        },
        "comparison": comparison.to_dict(orient="records"),
        "earnings_sensitivity_note": (
            "Reported dates are retrospective, not a point-in-time schedule archive; earnings results "
            "are descriptive and excluded from the core walk-forward proof."
        ),
        "session_note": (
            "The source contains regular US cash-session bars only; Asia/London filters were not tested."
        ),
    }
    (OUTPUT_DIR / "final_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(comparison.to_string(index=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
