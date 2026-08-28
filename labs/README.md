# Backtest lab

The lab uses the production signal engine without using future bars in the decision.
For each historical session it slices daily and hourly data through that date, builds
market and sector context from the same slice, ranks the selected universe, keeps the
top configured candidates, enters at the next trading session's open, and exits at the
close of the configured hold session.

At the default `$500` position size, `1R` is one percent (`$5`). Signals whose full
holding window is not inside the requested test period remain unevaluated. This matters
at train/test boundaries: the training optimizer never sees a return that finishes in
the test window.

## Data cache

Create a raw-data cache once, then reuse identical bytes across sweeps:

```bash
python backtest_2026.py --limit 150 --save-cache
python backtest_2026.py --use-cache --limit 150 --hold 5
```

The cache is `labs/cache/market_data.pkl.gz` and is intentionally ignored by Git because
hourly history can be large. Pickle can execute code while loading, so use only a cache
created locally by this script. `--use-cache` fails clearly if the cache does not exist.

The one-run output directory contains:

- `trades_2026.csv`: one row per mature next-open trade.
- `monthly_2026.csv`: monthly P/L, R, win rate, profit factor, and risk statistics.
- `summary.json`: aggregate metrics.
- `config.json`: exact settings used by the run.

## Sweeps

Run the grid using one prepared cache:

```bash
python labs/profit_sweep.py --use-cache --limit 150
```

The default grid compares hard ADX `17/18`, score floors `0.30/0.32/0.40`, 4-hour
slopes `0.002/0.003`, and dollar-volume caps `0/1500`. Results are written to
`labs/results/profit_sweep.csv`. Every dimension is configurable from `--help`.

## Honest train/test check

Select a configuration only on January-June data, then evaluate it once on the later
window:

```bash
python labs/honest_check.py \
  --use-cache \
  --limit 150 \
  --start 2026-01-01 \
  --split 2026-06-30 \
  --end 2026-08-27
```

`training_grid.csv` contains the selection evidence, `test_trades.csv` contains only
out-of-sample trades, and `honest_check.json` records both summaries and configurations.

## Honest limitations

- Yahoo's currently downloaded constituent list is not survivorship-bias-free.
- Today's earnings calendar and OpenInsider data are not historical point-in-time
  archives, so the lab excludes those filters instead of leaking current knowledge.
- Hourly availability depends on Yahoo's retention window; preserve the generated cache
  if a historical comparison must remain exactly reproducible.
- A small alphabetical subset is useful for iteration but is not proof that a setting
  scales to the full universe.
