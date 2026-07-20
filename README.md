# EMA scanner

`ema_scanner.py` scans US equity universes for daily pullback/reclaim setups, confirms
them against a market-open-aligned 4-hour EMA trend, and posts qualifying alerts to
Discord. A small Flask server exposes health and authenticated one-pass endpoints for
Render or another always-on host.

## Runtime flow

1. Build and de-duplicate the S&P 500/400/600, NASDAQ-100, Dow, biotech, sector ETF,
   and optional custom universes.
2. Download daily and hourly bars in bounded Yahoo Finance chunks with retry/backoff.
3. Prepare daily indicators and aggregate hourly bars into 4-hour candles anchored at
   the 09:30 US market open. The final candle of a regular session is intentionally
   shorter because the 6.5-hour cash session does not divide evenly into four hours.
4. Apply the pullback setup, 4-hour trend, market/sector regimes, relative strength,
   event, liquidity, and confirmation-score checks.
5. Record and alert each symbol/side at most once per trading date.
6. After a complete universe pass, evaluate mature signals using the configured number
   of trading sessions after their signal date.

Market and sector proxy data is cached for `PROXY_CACHE_MINUTES`; it should not be
downloaded again for every scanner batch. `REQUIRE_MARKET_REGIME`,
`REQUIRE_SECTOR_REGIME`, and `REQUIRE_RELATIVE_STRENGTH` are hard gates when enabled.
`MIN_DOLLAR_VOL_M` is also an eligibility gate. `CONFIRM_SCORE_BUY` and
`CONFIRM_SCORE_SELL` count technical confirmations only; disabled optional indicators
are removed from the available vote count.

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
| `RESAMPLE_4H_RULE` | `4h` | Pandas resampling rule for hourly data. |
| `RESAMPLE_4H_OFFSET` | `9h30min` | Session anchor; change only with matching market/data semantics. |
| `HOLD_DAYS` | `5` | Trading bars after the signal used for performance evaluation. |
| `BACKTEST_PERIOD` | `1y` | History window used to resolve mature logged signals. |
| `EXTRA_TICKERS` | empty | Comma-separated additional Yahoo symbols. |

Signal thresholds and feature flags are grouped near the top of `ema_scanner.py`. Keep
their environment names stable so deployment tuning survives code releases.

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

CI runs lint, formatting, and regression tests for every push and pull request.

## Maintenance notes

- `_prepare_daily_df` is the boundary between raw Yahoo frames and the signal engine;
  new indicators should be prepared there rather than recomputed per signal branch.
- `_select_batch` owns wrap/completion semantics. Callers should use its completion
  flag instead of inferring a pass from offset comparisons.
- Relative-strength context has separate long and short scores. Do not reuse the long
  score when adding short-side confidence features.
- Signal-log updates are lock-protected because continuous and manual scan threads can
  both produce or evaluate records.
- External event-data failures currently fail open for ordinary equities. Biotech BUY
  signals still require the configured insider confirmation unless explicitly allowed.
