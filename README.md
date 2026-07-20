# EMA scanner

`ema_scanner.py` scans US equity universes for daily pullback/reclaim setups, checks
them against a market-open-aligned 4-hour EMA trend, ranks eligible candidates, and
posts the strongest candidates to Discord. A small Flask server exposes health and
authenticated one-pass endpoints for Render or another always-on host.

## Runtime flow

1. Build and de-duplicate the S&P 500/400/600, NASDAQ-100, Dow, biotech, sector ETF,
   and optional custom universes.
2. Download daily and hourly bars in bounded Yahoo Finance chunks with retry/backoff.
3. Prepare daily indicators and aggregate hourly bars into 4-hour candles anchored at
   the 09:30 US market open. The final candle of a regular session is intentionally
   shorter because the 6.5-hour cash session does not divide evenly into four hours.
4. Exclude a mutable current-session daily candle and align 4-hour context to the same
   completed session as the daily setup.
5. Apply the pullback setup plus the 4-hour trend, market/sector regime, relative
   strength, event, and liquidity eligibility gates.
6. Score every eligible candidate from continuous technical and context components,
   rank the entire universe pass, and alert no more than `MAX_ALERTS_PER_CYCLE`.
7. Record the decision close for auditing, label the intended entry as the next session
   open, and alert each symbol/side at most once per signal date.
8. Evaluate mature signals from the next session's open through the configured holding
   period instead of assuming an unattainable fill at the signal close. Outcomes move
   from the pending log into an idempotent permanent results ledger.

Market and sector proxy data is cached for `PROXY_CACHE_MINUTES`; it should not be
downloaded again for every scanner batch. The pullback setup, symbol-level 4-hour trend,
`MIN_DOLLAR_VOL_M`, and event-risk rules are eligibility gates. EMA slope, pullback
location, MACD, ADX, reclaim-candle quality, 4-hour trend strength, market/sector
regimes, and relative strength are ranking evidence rather than all-confirmations
vetoes. Technical vote counts remain in alerts and logs as diagnostics, but they do not
decide eligibility.

Base score components are equal-weight by default. Each can be overridden with its
matching `WEIGHT_*` environment variable for controlled ablation. Changing score logic
or weights should also change `RANK_SCORE_VERSION`; historical win-rate context is only
computed from the same score version, side, holding period, and 10-point score band.

## Required environment

| Variable | Purpose |
| --- | --- |
| `DISCORD_WEBHOOK` | Discord webhook that receives alerts and reports. |
| `RUN_TOKEN` | Secret required by `POST /run-once`. |
| `PORT` | Flask port; defaults to `10000`. |

The endpoint prefers `X-Run-Token: <RUN_TOKEN>`. A `token` query parameter remains
accepted for older deployments, but headers are safer because URLs are often logged.

## Important tuning

| Variable | Default | Notes |
| --- | ---: | --- |
| `RUN_ONCE` | `0` | Exit after exactly one full universe pass. |
| `SCAN_BATCH_SIZE` | `125` | Symbols per batch; values larger than the universe are safely capped. |
| `BATCHES_PER_LOOP` | `1` | Batches processed before the continuous-loop delay. |
| `CHECK_INTERVAL` | `120` | Seconds between continuous scan segments. |
| `YF_BATCH_CHUNK` | `40` | Symbols per Yahoo Finance request. |
| `PROXY_CACHE_MINUTES` | `15` | Market/sector context cache lifetime. Set `0` to disable. |
| `MAX_ALERTS_PER_CYCLE` | `10` | Highest-ranked candidates retained from one full universe pass. |
| `MIN_RANK_SCORE` | `0.0` | Optional score floor; the default leaves selection to ranking and the cycle cap. |
| `DAILY_BAR_CLOSE_BUFFER_MINUTES` | `15` | Delay after 16:00 New York time before today's daily candle is considered complete. |
| `USE_MARKET_REGIME_SCORE` | `1` | Include broad-market regime quality in ranking. |
| `USE_SECTOR_REGIME_SCORE` | `1` | Include sector-regime quality in ranking. |
| `USE_RELATIVE_STRENGTH_SCORE` | `1` | Include stock-versus-market/sector strength in ranking. |
| `RESAMPLE_4H_RULE` | `4h` | Pandas resampling rule for hourly data. |
| `RESAMPLE_4H_OFFSET` | `9h30min` | Session anchor; change only with matching market/data semantics. |
| `HOLD_DAYS` | `5` | Trading bars after the signal used for performance evaluation. |
| `BACKTEST_PERIOD` | `1y` | History window used to resolve mature logged signals. |
| `RESULTS_LOG_FILE` | `signal_results.csv` | Permanent, de-duplicated evaluated-signal ledger. |
| `CALIBRATION_MIN_SAMPLES` | `20` | Same-band historical outcomes required before an alert displays a historical win rate. |
| `RANK_SCORE_VERSION` | `v1` | Calibration namespace; increment after changing score semantics or weights. |
| `EXTRA_TICKERS` | empty | Comma-separated additional Yahoo symbols. |

Signal thresholds and feature flags are grouped near the top of `ema_scanner.py`. Keep
their environment names stable so deployment tuning survives code releases.
The former `REQUIRE_MARKET_REGIME`, `REQUIRE_SECTOR_REGIME`, and
`REQUIRE_RELATIVE_STRENGTH` variables remain accepted as fallbacks for their matching
`USE_*_SCORE` variables, but now enable ranking components rather than hard gates.
Component weight variables are `WEIGHT_EMA_SLOPE`, `WEIGHT_PULLBACK_LOCATION`,
`WEIGHT_MACD_MOMENTUM`, `WEIGHT_RECLAIM_CANDLE`, `WEIGHT_TREND_4H`,
`WEIGHT_ADX_STRENGTH`, `WEIGHT_OBV_DIRECTION`, `WEIGHT_MARKET_REGIME`,
`WEIGHT_SECTOR_REGIME`, and `WEIGHT_RELATIVE_STRENGTH`; every default is `1.0`.
The results ledger is permanent relative to the configured filesystem. On Render or
another ephemeral container host, point `RESULTS_LOG_FILE` (and preferably `LOG_FILE`)
at a mounted persistent disk; otherwise deploys or instance replacement can remove the
history needed for calibration.

## Local development

Python 3.10 or newer is required.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt -r requirements-dev.txt
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m ruff format --check .
.venv/Scripts/python -m pytest
```

On macOS/Linux, use `.venv/bin/python` instead. Tests use synthetic market data and do
not call Yahoo Finance, Wikipedia, OpenInsider, Discord, or Render.

## Deployment and scheduling

Run the service with:

```bash
python ema_scanner.py
```

Health is available at `GET /healthz`. The response includes batch coverage, current
scan state, timestamps, signal count, and the most recent top-level error.

The daily GitHub Actions workflow requires repository secrets:

- `RENDER_RUN_URL`: the full HTTPS URL ending in `/run-once`, without a token.
- `RUN_TOKEN`: the same value configured on the running scanner.

The scheduled trigger runs at 22:00 UTC on weekdays, safely after the US cash close in
both standard and daylight-saving time. Signals generated from that completed session
therefore target the next available market open. Manually triggering a pass during the
cash session is supported for diagnostics, but today's mutable daily bar is excluded
and a candidate is not alerted after its modeled next-open entry window has passed.

CI runs lint, formatting, and regression tests for every push and pull request.

## Maintenance notes

- `_prepare_daily_df` is the boundary between raw Yahoo frames and the signal engine;
  new indicators should be prepared there rather than recomputed per signal branch.
- `_select_batch` owns wrap/completion semantics. Callers should use its completion
  flag instead of inferring a pass from offset comparisons.
- Relative-strength context has separate long and short scores. Do not reuse the long
  score when adding short-side ranking features.
- Keep eligibility gates separate from rank components. New evidence should normally
  improve ordering among valid setups instead of silently making the setup impossible.
- Signal-log updates are lock-protected because continuous and manual scan threads can
  both produce or evaluate records.
- The signal log retains the composite score, universe rank, component scores, and
  technical evidence flags. Preserve those fields when adding evaluation tooling; they
  support point-in-time score audits and one-feature-at-a-time ablation.
- `LOG_FILE` is the pending-evaluation queue; `RESULTS_LOG_FILE` is permanent history.
  Do not derive empirical win rates from the pending file or delete evaluated history.
- External event-data failures currently fail open for ordinary equities. Biotech BUY
  signals still require the configured insider confirmation unless explicitly allowed.
