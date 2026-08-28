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
| `MIN_RANK_SCORE` | `0.40` | Score floor; only trades >= 0.40 are alerted. `0.32` = balanced 68% WR, `0.30` = max total. |
| `ADX_MIN` | `17` | ADX rising threshold (was 22). Lowered after sweep: 22-30 +617, <15 -441. |
| `ADX_HARD_MIN` | `17` | **New** hard ADX gate; `0` disables. Blocks weak-trend 19% large-loss bucket. |
| `FOURH_SLOPE_MIN` | `0.002` | 4H EMA200 slope floor; `0.003` = balanced, `0.002` = max total. |
| `MAX_DOLLAR_VOL_M` | `0` | **New** upper vol cap ($M, 20d avg); `0` = no cap, `1500` cuts mega-cap churn (+557 2026). |
| `DAILY_BAR_CLOSE_BUFFER_MINUTES` | `15` | Delay after 16:00 New York time before today's daily candle is considered complete. |
| `USE_MARKET_REGIME_SCORE` | `1` | Include broad-market regime quality in ranking. |
| `USE_SECTOR_REGIME_SCORE` | `1` | Include sector-regime quality in ranking. |
| `USE_RELATIVE_STRENGTH_SCORE` | `1` | Include stock-versus-market/sector strength in ranking. |
| `RESAMPLE_4H_RULE` | `4h` | Pandas resampling rule for hourly data. |
| `RESAMPLE_4H_OFFSET` | `9h30min` | Session anchor: `9h30min` NY, `3h` London, `20h` Asia, `0h` Midnight. |
| `HOLD_DAYS` | `5` | Bars after signal for evaluation; `3` = quick, `5` = optimal full 2026, `7` = best Jul-Aug. |
| `BACKTEST_PERIOD` | `1y` | History window used to resolve mature logged signals. |
| `RESULTS_LOG_FILE` | `signal_results.csv` | Permanent, de-duplicated evaluated-signal ledger. |
| `CALIBRATION_MIN_SAMPLES` | `20` | Same-band historical outcomes required before an alert displays a historical win rate. |
| `RANK_SCORE_VERSION` | `v3` | Calibration namespace; increment after changing score semantics or weights. |
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

## Backtest Lab — 2023-2026 Honest OOS

Walk-forward `next-open → close[HOLD]`, no look-ahead (`date<=signal`), 150 S&P500 subset (4 missing 1h 0.8% survivorship), `$500/trade`, `1R = 1%`.

**Overall (150, 5d):**

| version | n | WR | Total | avg | Total R | PF | Large <-5% |
|---|---|---|---|---|---|---|---|
| Baseline 2026 v1 `22/0/0.002` | 511 | 52.6% | +397 | +0.78 | +79.5R | 1.07 | 13.9% |
| **V2 Balanced 2026** `18/0.32/0.003` | 72 | 68.1% | +839 | +11.65 | +167.7R | 3.80 | 4.2% |
| **V3 Profit 2026** `17/0.30/0.002` | 227 | 57.7% | **+1214** | +5.35 | +242.9R | 1.75 | 7.9% |
| Baseline 2025 | 415 | 44.3% | -475 | -1.15 | -95R | 0.88 | 11.8% |
| V3 2025 OOS (never tuned) | 202 | 48.0% | **+52** | +0.26 | +10.3R | 1.03 | 9.9% |

**2026 monthly (Baseline vs V2 vs V3):**

| month | Baseline n WR PnL R | V2 n WR PnL R | V3 n WR PnL R |
|---|---|---|---|
| 01 | 54 55.6% +83 +16.6R | 6 50% +84 +16.8R | 29 55% +102 +20.4R |
| 02 | 77 63.6% +293 +58.5R | 16 81% +211 +42R | 45 66% +423 +84.6R |
| 03 | 60 51% -289 -57R | 5 80% +52 +10R | 28 53% +22 +4.3R |
| 04 | 45 53% +80 +16R | 3 100% +101 +20R | 22 54% +2 +0.5R |
| 05 | 54 48% +109 +21R | 6 33% +3 +0.7R | 11 45% -7 -1.3R |
| 06 | 77 45% -47 -9R | 3 100% +57 +11R | 18 61% +335 +67R |
| 07 | 87 52% +127 +25R | 18 61% +158 +31R | 44 61% +219 +43R |
| 08 | 57 49% +42 +8R | 15 66% +173 +34R | 30 50% +118 +23R |

**Honest OOS (train 2026-01→06 → test 2026-07→08):** Baseline H2 144 +169 +33R → V3-filtered H2 **67 +357 +71R PF2.02** (+$189 honest, Jul 42 +191, Aug 25 +167). Full 2026 best train `17/0.30/0.002` = 210 +1327.

**Sweeps (`labs/`):**

- **Volume** `MIN/MAX $M` (`volume_hold_sweep.py`): 2026 best `25/1500` **469 +954 +190R PF1.2** vs no-cap 511 +397; 2025 best no-cap (volume filter regime-dependent).
- **HOLD** `3/5/7` (`2026-07→08` 150, NY): HOLD3 76 +196 +39R, **HOLD5 74 +332 +66R PF1.82**, HOLD7 **68 +433 +86R PF2.03** — HOLD5 optimal full 2026 (+782 HOLD7 full = -35% total).
- **Sessions** `RESAMPLE_4H_OFFSET` (`2026-07→08` 150, HOLD5): NY `9h30min` 74 +332 PF1.82, London `3h` 56 +91 PF1.22, **Asia `20h` 70 +375 PF1.92**, Midnight `0h` 70 +375 — Asia slightly best, NY is documented.

Full sweep cache `labs/cache/` + `backtest_2026.py --use-cache --save-cache` (3.3M daily, 38M 1h), `labs/` for repro. OOS 2024-08→2025-08 V3 also +52 (see `labs/oos_2024_2025/`).

Run lab: `python ema-bot/backtest_2026.py --use-cache --limit 150 --hold 5` or `python -m pytest` (28 tests, `labs/` for sweeps).

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
