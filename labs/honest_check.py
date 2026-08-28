"""Choose settings on a training window, then evaluate them once out of sample."""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ema_scanner as scanner  # noqa: E402
from backtest_2026 import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    BacktestConfig,
    config_to_dict,
    download_dataset,
    load_dataset_cache,
    prepare_daily_frames,
    run_backtest,
    save_dataset_cache,
    summarize_trades,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "labs" / "results" / "honest_check"


def _date(value: str) -> datetime.date:
    try:
        return datetime.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from exc


def _float_grid(value: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not values:
        raise argparse.ArgumentTypeError("grid cannot be empty")
    return values


def _objective_value(summary: dict, objective: str) -> float:
    value = summary.get(objective)
    if value is None or not math.isfinite(float(value)):
        return float("-inf")
    return float(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=_date, default=datetime.date(2026, 1, 1))
    parser.add_argument("--split", type=_date, default=datetime.date(2026, 6, 30))
    parser.add_argument("--end", type=_date, default=datetime.date.today())
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--hold", type=int, default=5)
    parser.add_argument("--adx-min", type=float, default=17.0)
    parser.add_argument("--adx-hard-mins", type=_float_grid, default=[17.0, 18.0])
    parser.add_argument("--min-scores", type=_float_grid, default=[0.30, 0.32, 0.40])
    parser.add_argument("--slopes", type=_float_grid, default=[0.002, 0.003])
    parser.add_argument("--min-dollar-vol", type=float, default=25.0)
    parser.add_argument("--max-dollar-vols", type=_float_grid, default=[0.0, 1500.0])
    parser.add_argument("--offset", default="9h30min")
    parser.add_argument("--max-alerts", type=int, default=10)
    parser.add_argument(
        "--objective",
        choices=["total_pnl_usd", "total_r", "profit_factor", "sharpe"],
        default="total_pnl_usd",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--save-cache", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.start <= args.split < args.end:
        raise ValueError("Require start <= split < end")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative")

    if args.use_cache:
        dataset = load_dataset_cache(args.cache_dir).subset(limit=args.limit)
    else:
        dataset = download_dataset(limit=args.limit)
        if args.save_cache:
            save_dataset_cache(dataset, args.cache_dir)
    scanner.SECTOR_PROXY_BY_SYMBOL.update(dataset.sector_proxies)
    prepared = prepare_daily_frames(dataset)

    training_rows = []
    best_config = None
    best_summary = None
    best_objective = float("-inf")
    combinations = itertools.product(
        args.adx_hard_mins,
        args.min_scores,
        args.slopes,
        args.max_dollar_vols,
    )
    for adx_hard_min, min_score, slope, max_dollar_vol in combinations:
        config = BacktestConfig(
            start_date=args.start,
            end_date=args.split,
            hold_days=args.hold,
            max_alerts_per_day=args.max_alerts,
            adx_min=args.adx_min,
            adx_hard_min=adx_hard_min,
            min_score=min_score,
            fourh_slope_min=slope,
            min_dollar_vol_m=args.min_dollar_vol,
            max_dollar_vol_m=max_dollar_vol,
            resample_offset=args.offset,
        )
        trades = run_backtest(dataset, config, prepared_daily=prepared)
        summary = summarize_trades(trades)
        training_rows.append({**config_to_dict(config), **summary})
        objective_value = _objective_value(summary, args.objective)
        if best_config is None or objective_value > best_objective:
            best_config = config
            best_summary = summary
            best_objective = objective_value

    if best_config is None:
        raise RuntimeError("The training grid produced no selectable configuration")
    test_config = BacktestConfig(
        **{
            **config_to_dict(best_config),
            "start_date": args.split + datetime.timedelta(days=1),
            "end_date": args.end,
        }
    )
    test_trades = run_backtest(dataset, test_config, prepared_daily=prepared)
    test_summary = summarize_trades(test_trades)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(training_rows).sort_values(
        args.objective,
        ascending=False,
        na_position="last",
    ).to_csv(args.output_dir / "training_grid.csv", index=False)
    test_trades.to_csv(args.output_dir / "test_trades.csv", index=False)
    report = {
        "objective": args.objective,
        "selected_training_config": config_to_dict(best_config),
        "training_summary": best_summary,
        "out_of_sample_config": config_to_dict(test_config),
        "out_of_sample_summary": test_summary,
    }
    (args.output_dir / "honest_check.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
