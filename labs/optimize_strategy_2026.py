"""Walk-forward research for robust, operational EMA-scanner improvements.

The grid is selected on January-April 2026, narrowed on May-June, evaluated
once on July, and then compared with the already-observed August MTD period.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_2026 as bt
import ema_scanner as scanner

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "labs" / "cache" / "august_2026"
OUTPUT_DIR = ROOT / "labs" / "results" / "optimization_2026"
START = dt.date(2026, 1, 2)
END = dt.date(2026, 8, 26)
PERIODS = {
    "training": (dt.date(2026, 1, 2), dt.date(2026, 4, 30)),
    "validation": (dt.date(2026, 5, 1), dt.date(2026, 6, 30)),
    "july_oos": (dt.date(2026, 7, 1), dt.date(2026, 7, 31)),
    "august_posthoc": (dt.date(2026, 8, 1), dt.date(2026, 8, 26)),
}


@dataclasses.dataclass(frozen=True)
class Variant:
    score_min: float = 0.40
    adx_min: float = 18.0
    volume_min_m: float = 25.0
    volume_max_m: float = 1500.0
    slope_gate: float = 0.003
    exclude_friday: bool = False
    earnings_window: bool = False
    gap_atr_max: float | None = None
    no_symbol_overlap: bool = False
    max_alerts: int = 10

    @property
    def name(self) -> str:
        values = dataclasses.asdict(self)
        return "__".join(f"{key}={value}" for key, value in values.items())


BASELINE = Variant()
PRODUCTION_EVENTS = dataclasses.replace(BASELINE, earnings_window=True, gap_atr_max=1.2)


def _nearest_earnings_features(
    symbol: str,
    signal_date: dt.date,
    earnings: dict[str, list[dt.date]],
) -> tuple[float, float, bool]:
    dates = earnings.get(symbol, [])
    before = [(value - signal_date).days for value in dates if value >= signal_date]
    after = [(signal_date - value).days for value in dates if value <= signal_date]
    days_to_next = min(before) if before else math.nan
    days_since_previous = min(after) if after else math.nan
    blocked = any(-2 <= (value - signal_date).days <= 3 for value in dates)
    return days_to_next, days_since_previous, blocked


def build_candidate_ledger() -> pd.DataFrame:
    dataset = bt.load_dataset_cache(CACHE_DIR)
    scanner.SECTOR_PROXY_BY_SYMBOL.update(dataset.sector_proxies)
    prepared = bt.prepare_daily_frames(dataset)
    earnings_frame = pd.read_csv(CACHE_DIR / "earnings_dates.csv")
    earnings_frame["earnings_date"] = pd.to_datetime(earnings_frame["earnings_date"]).dt.date
    earnings = {
        symbol: sorted(set(group["earnings_date"]))
        for symbol, group in earnings_frame.groupby("symbol")
    }
    permissive = bt.BacktestConfig(
        start_date=START,
        end_date=END,
        adx_hard_min=0.0,
        min_score=0.0,
        fourh_slope_min=BASELINE.slope_gate,
        min_dollar_vol_m=0.0,
        max_dollar_vol_m=0.0,
    )
    with bt.scanner_settings(permissive):
        trends = bt.precompute_trend_contexts(dataset, prepared, permissive)
        by_date = bt.collect_candidates(dataset, permissive, prepared, trends)

    records = []
    for signal_date, candidates in sorted(by_date.items()):
        for candidate in candidates:
            frame = prepared[candidate.symbol]
            entry_position = candidate.position + 1
            exit_position = candidate.position + permissive.hold_days
            if exit_position >= len(frame):
                continue
            entry_date = bt._date_from_index(frame.index[entry_position])
            exit_date = bt._date_from_index(frame.index[exit_position])
            if exit_date > END:
                continue
            entry = float(frame["Open"].iloc[entry_position])
            exit_price = float(frame["Close"].iloc[exit_position])
            signal_close = float(frame["Close"].iloc[candidate.position])
            previous_close = float(frame["Close"].iloc[candidate.position - 1])
            signal_open = float(frame["Open"].iloc[candidate.position])
            atr = float(frame["atr"].iloc[candidate.position])
            raw_return = (exit_price - entry) / entry
            direction = 1.0 if candidate.side == "BUY" else -1.0
            result = raw_return * direction
            days_to_next, days_since_previous, earnings_blocked = _nearest_earnings_features(
                candidate.symbol, signal_date, earnings
            )
            metadata = candidate.metadata
            components = metadata.get("score_components", {})
            records.append(
                {
                    "signal_date": signal_date.isoformat(),
                    "entry_date": entry_date.isoformat(),
                    "exit_date": exit_date.isoformat(),
                    "symbol": candidate.symbol,
                    "sector_proxy": dataset.sector_proxies.get(candidate.symbol),
                    "side": candidate.side,
                    "score": candidate.score,
                    "adx": candidate.adx,
                    "avg_dollar_vol_m": metadata.get("avg_dollar_vol_m"),
                    "trend4h_slope": metadata.get("trend4h_slope"),
                    "z_dist": metadata.get("z_dist"),
                    "market_score": metadata.get("market_score"),
                    "sector_score": metadata.get("sector_score"),
                    "rs_score": metadata.get("rs_score"),
                    "technical_votes": metadata.get("technical_votes"),
                    "ema_slope_score": components.get("ema_slope"),
                    "macd_score": components.get("macd_momentum"),
                    "reclaim_score": components.get("reclaim_candle"),
                    "signal_weekday": signal_date.strftime("%A"),
                    "entry_weekday": entry_date.strftime("%A"),
                    "signal_price": signal_close,
                    "entry_price": entry,
                    "exit_price": exit_price,
                    "signal_gap_atr": abs(signal_open - previous_close) / max(atr, 1e-8),
                    "entry_gap_pct": (entry - signal_close) / signal_close * direction,
                    "days_to_next_earnings": days_to_next,
                    "days_since_previous_earnings": days_since_previous,
                    "earnings_window_3_before_2_after": earnings_blocked,
                    "return_decimal": result,
                    "profit_usd": result * permissive.capital_per_trade,
                    "r_multiple": result / permissive.risk_fraction,
                }
            )
    frame = pd.DataFrame(records).sort_values(
        ["signal_date", "score", "symbol", "side"], ascending=[True, False, True, True]
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / "candidate_ledger.csv", index=False)
    return frame


def apply_variant(
    ledger: pd.DataFrame,
    variant: Variant,
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    start_text, end_text = start.isoformat(), end.isoformat()
    selected = ledger[
        ledger["signal_date"].between(start_text, end_text)
        & (ledger["exit_date"] <= end_text)
        & (ledger["score"] >= variant.score_min)
        & (ledger["adx"] >= variant.adx_min)
        & ledger["avg_dollar_vol_m"].between(variant.volume_min_m, variant.volume_max_m)
        & (ledger["trend4h_slope"].abs() >= variant.slope_gate)
    ]
    if variant.exclude_friday:
        selected = selected[selected["signal_weekday"] != "Friday"]
    if variant.earnings_window:
        # Match the optimized live rule: block the report date and three
        # calendar days before it, then permit setups from the next session.
        selected = selected[~selected["days_to_next_earnings"].between(0, 3)]
    if variant.gap_atr_max is not None:
        selected = selected[selected["signal_gap_atr"] <= variant.gap_atr_max]

    output = []
    active_until: dict[str, str] = {}
    for _, day in selected.groupby("signal_date", sort=True):
        ranked = day.sort_values(["score", "symbol", "side"], ascending=[False, True, True])
        accepted = 0
        for row in ranked.itertuples(index=False):
            if variant.no_symbol_overlap and row.signal_date < active_until.get(row.symbol, ""):
                continue
            output.append(row._asdict())
            accepted += 1
            if variant.no_symbol_overlap:
                active_until[row.symbol] = row.exit_date
            if accepted >= variant.max_alerts:
                break
    return pd.DataFrame(output, columns=ledger.columns)


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "total_r": 0.0,
            "average_r": 0.0,
            "profit_factor": None,
            "worst_r": None,
            "positive_month_pct": 0.0,
            "lcb_mean_r": float("-inf"),
        }
    values = trades["r_multiple"].astype(float)
    gross_win = float(values[values > 0].sum())
    gross_loss = abs(float(values[values < 0].sum()))
    monthly = trades.assign(month=trades["signal_date"].str.slice(0, 7)).groupby("month")[
        "r_multiple"
    ].sum()
    standard_error = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0
    return {
        "trades": len(trades),
        "win_rate_pct": float((values > 0).mean() * 100),
        "total_r": float(values.sum()),
        "average_r": float(values.mean()),
        "profit_factor": gross_win / gross_loss if gross_loss else None,
        "worst_r": float(values.min()),
        "positive_month_pct": float((monthly > 0).mean() * 100),
        "lcb_mean_r": float(values.mean() - standard_error),
    }


def _variant_grid():
    for values in itertools.product(
        [0.40, 0.45, 0.50],
        [18.0, 20.0, 22.0],
        [25.0, 100.0, 200.0],
        [500.0, 750.0, 1000.0, 1500.0],
        [0.003, 0.004],
        [False, True],
        [False, True],
        [None, 1.2],
        [False, True],
        [5, 10],
    ):
        variant = Variant(*values)
        if variant.volume_min_m <= variant.volume_max_m:
            yield variant


def _record(variant: Variant, period: str, trades: pd.DataFrame) -> dict:
    return {"variant": variant.name, **dataclasses.asdict(variant), "period": period, **summarize(trades)}


def descriptive_tables(baseline: pd.DataFrame) -> None:
    specifications = {
        "weekday": (baseline["signal_weekday"], ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
        "volume_band": (
            pd.cut(
                baseline["avg_dollar_vol_m"],
                [0, 100, 250, 500, 750, 1000, 1500, np.inf],
                right=False,
            ).astype(str),
            None,
        ),
        "score_band": (pd.cut(baseline["score"], [0.4, 0.45, 0.5, 0.55, 0.6, 1.01], right=False).astype(str), None),
        "adx_band": (pd.cut(baseline["adx"], [18, 20, 22, 25, 30, 40, 101], right=False).astype(str), None),
        "earnings_distance": (
            np.select(
                [
                    baseline["earnings_window_3_before_2_after"],
                    baseline["days_since_previous_earnings"].between(3, 7),
                    baseline["days_to_next_earnings"].between(4, 7),
                ],
                ["-2 to +3 calendar days", "3-7 days after", "4-7 days before"],
                default="outside 7-day vicinity",
            ),
            None,
        ),
    }
    for filename, (groups, order) in specifications.items():
        rows = []
        for group, trades in baseline.assign(_group=groups).groupby("_group", observed=True):
            rows.append({filename: str(group), **summarize(trades)})
        result = pd.DataFrame(rows)
        if order:
            result[filename] = pd.Categorical(result[filename], categories=order, ordered=True)
            result = result.sort_values(filename)
        result.to_csv(OUTPUT_DIR / f"by_{filename}.csv", index=False)


def main() -> int:
    ledger_path = OUTPUT_DIR / "candidate_ledger.csv"
    ledger = pd.read_csv(ledger_path) if ledger_path.exists() else build_candidate_ledger()
    training_start, training_end = PERIODS["training"]
    training_rows = []
    variants: dict[str, Variant] = {}
    for number, variant in enumerate(_variant_grid(), 1):
        variants[variant.name] = variant
        trades = apply_variant(ledger, variant, training_start, training_end)
        summary = summarize(trades)
        if summary["trades"] >= 20:
            training_rows.append(_record(variant, "training", trades))
        if number % 1000 == 0:
            print(f"optimized {number} variants", flush=True)
    training = pd.DataFrame(training_rows).sort_values(
        ["lcb_mean_r", "total_r", "trades"], ascending=[False, False, False]
    )
    training.to_csv(OUTPUT_DIR / "training_grid.csv", index=False)

    finalists = training.head(200)
    validation_start, validation_end = PERIODS["validation"]
    validation_rows = []
    for row in finalists.itertuples(index=False):
        variant = variants[row.variant]
        trades = apply_variant(ledger, variant, validation_start, validation_end)
        validation_rows.append(_record(variant, "validation", trades))
    validation = pd.DataFrame(validation_rows)
    pd.DataFrame(validation_rows).to_csv(OUTPUT_DIR / "validation_top200_all.csv", index=False)
    eligible = validation[
        (validation["trades"] >= 10)
        & (validation["total_r"] > 0)
        & (validation["lcb_mean_r"] > -0.25)
    ].copy()
    eligible = eligible.merge(
        training[["variant", "lcb_mean_r", "positive_month_pct"]].rename(
            columns={"lcb_mean_r": "training_lcb", "positive_month_pct": "training_positive_month_pct"}
        ),
        on="variant",
    )
    eligible["selection_score"] = eligible["lcb_mean_r"] + 0.25 * eligible["training_lcb"]
    eligible = eligible.sort_values(
        ["selection_score", "trades", "variant"], ascending=[False, False, True]
    )
    eligible.to_csv(OUTPUT_DIR / "validation_finalists.csv", index=False)
    if eligible.empty:
        # Refuse to promote a narrow high-dimensional grid winner. This simple
        # liquidity/event variant is evaluated separately with the hold sweep in
        # finalize_optimization_2026.py.
        chosen = dataclasses.replace(
            BASELINE,
            volume_min_m=100.0,
            volume_max_m=750.0,
            earnings_window=True,
            gap_atr_max=1.2,
        )
        selected_validation_score = None
        selection_note = "No high-dimensional grid finalist passed validation; used simple fallback"
    else:
        chosen = variants[eligible.iloc[0]["variant"]]
        selected_validation_score = float(eligible.iloc[0]["selection_score"])
        selection_note = "Selected from training finalists on validation"

    comparison_rows = []
    trade_outputs = {}
    named = {"baseline": BASELINE, "production_events": PRODUCTION_EVENTS, "selected": chosen}
    for name, variant in named.items():
        for period, (start, end) in PERIODS.items():
            trades = apply_variant(ledger, variant, start, end)
            comparison_rows.append({"strategy": name, **_record(variant, period, trades)})
            trade_outputs[(name, period)] = trades
            trades.to_csv(OUTPUT_DIR / f"trades_{name}_{period}.csv", index=False)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUT_DIR / "walk_forward_comparison.csv", index=False)

    all_start, all_end = START, END
    baseline_all = apply_variant(ledger, BASELINE, all_start, all_end)
    selected_all = apply_variant(ledger, chosen, all_start, all_end)
    descriptive_tables(baseline_all)
    month_rows = []
    for name, trades in [("baseline", baseline_all), ("selected", selected_all)]:
        for month, group in trades.groupby(trades["signal_date"].str.slice(0, 7)):
            month_rows.append({"strategy": name, "month": month, **summarize(group)})
    pd.DataFrame(month_rows).to_csv(OUTPUT_DIR / "monthly_comparison.csv", index=False)

    report = {
        "candidate_rows": len(ledger),
        "grid_variants": len(variants),
        "selection_protocol": {
            "training": [value.isoformat() for value in PERIODS["training"]],
            "validation": [value.isoformat() for value in PERIODS["validation"]],
            "formal_out_of_sample": [value.isoformat() for value in PERIODS["july_oos"]],
            "august_status": "post-hoc because the user and analyst already saw August outcomes",
        },
        "baseline": dataclasses.asdict(BASELINE),
        "production_events": dataclasses.asdict(PRODUCTION_EVENTS),
        "selected": dataclasses.asdict(chosen),
        "selected_variant": chosen.name,
        "selected_validation_score": selected_validation_score,
        "selection_note": selection_note,
        "comparison": comparison.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "optimization.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "comparison"}, indent=2))
    print(comparison[["strategy", "period", "trades", "total_r", "profit_factor", "lcb_mean_r"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
