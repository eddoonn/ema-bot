"""Point-in-time backtest harness for the EMA scanner.

The lab deliberately evaluates only information available on each signal date,
ranks the complete selected universe for that date, and fills at the next open.
Current earnings-calendar and OpenInsider data are excluded because those sources
are not point-in-time archives.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import gzip
import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import ema_scanner as scanner

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "labs" / "cache"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "labs" / "results" / "backtest_2026"
CACHE_FILE_NAME = "market_data.pkl.gz"
CACHE_FORMAT_VERSION = 1


@dataclass
class MarketDataset:
    """Raw market frames plus the universe metadata needed by the scanner."""

    symbols: list[str]
    daily: dict[str, pd.DataFrame]
    hourly: dict[str, pd.DataFrame]
    sector_proxies: dict[str, str]
    downloaded_at: str

    def subset(self, requested: list[str] | None = None, limit: int = 0) -> MarketDataset:
        selected = requested or self.symbols
        selected = [symbol for symbol in selected if symbol in self.symbols]
        if requested:
            missing = sorted(set(requested).difference(selected))
            if missing:
                raise ValueError(f"Symbols are not present in the cache: {missing}")
        if limit > 0:
            selected = selected[:limit]

        proxies = {
            self.sector_proxies[symbol] for symbol in selected if symbol in self.sector_proxies
        }
        retained = set(selected) | proxies | {scanner.MARKET_PROXY}
        return MarketDataset(
            symbols=selected,
            daily={key: value for key, value in self.daily.items() if key in retained},
            hourly={key: value for key, value in self.hourly.items() if key in retained},
            sector_proxies={
                key: value for key, value in self.sector_proxies.items() if key in selected
            },
            downloaded_at=self.downloaded_at,
        )


@dataclass(frozen=True)
class BacktestConfig:
    start_date: datetime.date
    end_date: datetime.date
    hold_days: int = scanner.HOLD_DAYS
    capital_per_trade: float = 500.0
    risk_fraction: float = 0.01
    max_alerts_per_day: int = 10
    adx_min: float = 17.0
    adx_hard_min: float = 18.0
    min_score: float = 0.40
    fourh_slope_min: float = 0.003
    min_dollar_vol_m: float = scanner.MIN_DOLLAR_VOL_M
    max_dollar_vol_m: float = scanner.MAX_DOLLAR_VOL_M
    resample_offset: str = "9h30min"


@dataclass
class BacktestCandidate:
    symbol: str
    side: str
    signal_date: datetime.date
    position: int
    decision_price: float
    adx: float
    score: float
    metadata: dict


def _date_from_index(value) -> datetime.date:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(scanner.NEW_YORK_TZ)
    return timestamp.date()


def _through_date(frame: pd.DataFrame | None, through: datetime.date) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return None
    mask = [_date_from_index(value) <= through for value in frame.index]
    result = frame.loc[mask]
    return result if not result.empty else None


def _parse_symbols(value: str | None) -> list[str] | None:
    if not value:
        return None
    return sorted(set(scanner.normalize_tickers(value.split(","))))


def download_dataset(
    *,
    requested_symbols: list[str] | None = None,
    limit: int = 0,
    daily_period: str = "5y",
    hourly_period: str = "730d",
) -> MarketDataset:
    """Download one reproducible raw dataset using the scanner's retry logic."""
    symbols = requested_symbols or sorted(scanner._build_universe())
    if limit > 0:
        symbols = symbols[:limit]
    if not symbols:
        raise ValueError("The selected universe is empty")

    sector_proxies = {}
    for symbol in symbols:
        proxy = scanner.infer_sector_proxy(symbol)
        if proxy:
            sector_proxies[symbol] = proxy

    download_symbols = sorted(set(symbols) | set(sector_proxies.values()) | {scanner.MARKET_PROXY})
    scanner.logging.info(
        "Backtest download: %d stocks plus required market/sector proxies",
        len(symbols),
    )
    daily = scanner._download_batch_chunked(
        download_symbols,
        period=daily_period,
        interval=scanner.TIMEFRAME_DAILY,
        label="lab-daily",
    )
    hourly = scanner._download_batch_chunked(
        download_symbols,
        period=hourly_period,
        interval=scanner.TIMEFRAME_4H_BASE,
        label="lab-hourly",
    )
    usable_symbols = [
        symbol
        for symbol in symbols
        if daily.get(symbol) is not None
        and not daily[symbol].empty
        and hourly.get(symbol) is not None
        and not hourly[symbol].empty
    ]
    if not usable_symbols:
        raise RuntimeError("No symbols had both daily and hourly data")
    if len(usable_symbols) != len(symbols):
        scanner.logging.warning(
            "Backtest data is incomplete for %d of %d selected symbols",
            len(symbols) - len(usable_symbols),
            len(symbols),
        )

    return MarketDataset(
        symbols=usable_symbols,
        daily=daily,
        hourly=hourly,
        sector_proxies={
            symbol: proxy for symbol, proxy in sector_proxies.items() if symbol in usable_symbols
        },
        downloaded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


def save_dataset_cache(dataset: MarketDataset, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / CACHE_FILE_NAME
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    payload = {
        "format_version": CACHE_FORMAT_VERSION,
        "symbols": dataset.symbols,
        "daily": dataset.daily,
        "hourly": dataset.hourly,
        "sector_proxies": dataset.sector_proxies,
        "downloaded_at": dataset.downloaded_at,
    }
    with gzip.open(temporary_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(cache_path)
    return cache_path


def load_dataset_cache(cache_dir: Path) -> MarketDataset:
    """Load a cache created by this script; pickle caches must remain trusted."""
    cache_path = cache_dir / CACHE_FILE_NAME
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache not found at {cache_path}. Run once with --save-cache first."
        )
    with gzip.open(cache_path, "rb") as handle:
        # Cache files are locally generated and explicitly documented as trusted-only.
        payload = pickle.load(handle)
    if payload.get("format_version") != CACHE_FORMAT_VERSION:
        raise ValueError("Unsupported cache format; regenerate it with --save-cache")
    return MarketDataset(
        symbols=list(payload["symbols"]),
        daily=dict(payload["daily"]),
        hourly=dict(payload["hourly"]),
        sector_proxies=dict(payload["sector_proxies"]),
        downloaded_at=str(payload["downloaded_at"]),
    )


def prepare_daily_frames(dataset: MarketDataset) -> dict[str, pd.DataFrame]:
    prepared = {}
    for symbol, frame in dataset.daily.items():
        value = scanner._prepare_daily_df(frame)
        if value is not None and not value.empty:
            prepared[symbol] = value
    return prepared


def _precompute_symbol_trends(
    hourly: pd.DataFrame | None,
    signal_dates: list[datetime.date],
) -> dict[datetime.date, dict]:
    """Calculate causal 4H trend contexts once per symbol and signal date.

    The production helper resamples the same hourly prefix for every date. This
    equivalent implementation resamples once, then reads only indicator values
    whose timestamps are on or before each decision date. EWM and ATR are causal,
    so later rows cannot affect an earlier context.
    """
    if hourly is None or hourly.empty or not signal_dates:
        return {}
    source = scanner._to_numeric_cols(
        hourly.copy(), ["Open", "High", "Low", "Close", "Volume"]
    )
    fourh = scanner.resample_to_4h(source)
    minimum_bars = max(
        scanner.EMA_TREND + scanner.FOURH_SLOPE_BARS,
        scanner.FOURH_SLOPE_BARS + 14,
    )
    if fourh is None or fourh.empty or len(fourh) < minimum_bars:
        return {}
    fourh = fourh.copy()
    fourh["atr4h"] = scanner.ta.volatility.AverageTrueRange(
        fourh["High"], fourh["Low"], fourh["Close"], window=14
    ).average_true_range()
    fourh["ema_trend"] = fourh["Close"].ewm(span=scanner.EMA_TREND, adjust=False).mean()
    bar_dates = np.array([_date_from_index(value) for value in fourh.index], dtype=object)
    contexts = {}
    for signal_date in signal_dates:
        end = int(np.searchsorted(bar_dates, signal_date, side="right"))
        if end < minimum_bars:
            continue
        current = end - 1
        previous = current - scanner.FOURH_SLOPE_BARS
        ema_now = scanner.to_float(fourh["ema_trend"].iloc[current])
        ema_prev = scanner.to_float(fourh["ema_trend"].iloc[previous])
        close_now = scanner.to_float(fourh["Close"].iloc[current])
        atr4h = scanner.to_float(fourh["atr4h"].iloc[current])
        if any(pd.isna(value) for value in [ema_now, ema_prev, close_now, atr4h]):
            continue
        slope = (ema_now / ema_prev - 1.0) if ema_prev else np.nan
        stretch_atr = (close_now - ema_now) / max(atr4h, 1e-8)
        contexts[signal_date] = {
            "ema_trend": ema_now,
            "close_4h": close_now,
            "atr4h": atr4h,
            "slope_4h": slope,
            "stretch_atr": stretch_atr,
            "long_ok": bool(
                close_now > ema_now * scanner.TREND_BUF
                and slope >= scanner.FOURH_SLOPE_MIN
                and stretch_atr <= scanner.FOURH_MAX_STRETCH_ATR
            ),
            "short_ok": bool(
                close_now < ema_now / scanner.TREND_BUF
                and slope <= -scanner.FOURH_SLOPE_MIN
                and stretch_atr >= -scanner.FOURH_MAX_STRETCH_ATR
            ),
        }
    return contexts


def precompute_trend_contexts(
    dataset: MarketDataset,
    prepared_daily: dict[str, pd.DataFrame],
    config: BacktestConfig,
) -> dict[tuple[str, datetime.date], dict]:
    """Return point-in-time stock and proxy trend contexts for one test window."""
    dates = sorted(
        {
            _date_from_index(index_value)
            for frame in prepared_daily.values()
            for index_value in frame.index
            if config.start_date <= _date_from_index(index_value) <= config.end_date
        }
    )
    contexts = {}
    for symbol, hourly in dataset.hourly.items():
        contexts.update(
            {
                (symbol, signal_date): context
                for signal_date, context in _precompute_symbol_trends(hourly, dates).items()
            }
        )
    return contexts


@contextlib.contextmanager
def scanner_settings(config: BacktestConfig):
    """Apply one configuration to the production signal engine, then restore it."""
    updates = {
        "ADX_MIN": config.adx_min,
        "ADX_HARD_MIN": config.adx_hard_min,
        "MIN_RANK_SCORE": config.min_score,
        "FOURH_SLOPE_MIN": config.fourh_slope_min,
        "MIN_DOLLAR_VOL_M": config.min_dollar_vol_m,
        "MAX_DOLLAR_VOL_M": config.max_dollar_vol_m,
        "RESAMPLE_4H_OFFSET": config.resample_offset,
        "HOLD_DAYS": config.hold_days,
        "CAPITAL_PER_TRADE": config.capital_per_trade,
        "MAX_ALERTS_PER_CYCLE": config.max_alerts_per_day,
    }
    originals = {name: getattr(scanner, name) for name in updates}
    try:
        for name, value in updates.items():
            setattr(scanner, name, value)
        scanner.validate_config()
        yield
    finally:
        for name, value in originals.items():
            setattr(scanner, name, value)


def _empty_regime(label: str) -> dict:
    return {
        "label": label,
        "long_ok": False,
        "short_ok": False,
        "score_long": 0.0,
        "score_short": 0.0,
    }


def _regime_context(
    proxy: str,
    signal_date: datetime.date,
    prepared_daily: dict[str, pd.DataFrame],
    hourly: dict[str, pd.DataFrame],
    cache: dict[tuple[str, datetime.date], tuple[pd.DataFrame | None, dict]],
    trend_contexts: dict[tuple[str, datetime.date], dict] | None = None,
) -> tuple[pd.DataFrame | None, dict]:
    key = (proxy, signal_date)
    if key in cache:
        return cache[key]
    daily = _through_date(prepared_daily.get(proxy), signal_date)
    trend = (
        trend_contexts.get((proxy, signal_date))
        if trend_contexts is not None
        else scanner._extract_ema_trend(hourly.get(proxy), through_date=signal_date)
    )
    regime = scanner.build_regime_context(daily, trend, proxy) if daily is not None else None
    value = (daily, regime or _empty_regime(proxy))
    cache[key] = value
    return value


def collect_candidates(
    dataset: MarketDataset,
    config: BacktestConfig,
    prepared_daily: dict[str, pd.DataFrame],
    trend_contexts: dict[tuple[str, datetime.date], dict] | None = None,
) -> dict[datetime.date, list[BacktestCandidate]]:
    """Create point-in-time candidates before cross-sectional daily ranking."""
    by_date: dict[datetime.date, list[BacktestCandidate]] = {}
    regime_cache: dict[tuple[str, datetime.date], tuple[pd.DataFrame | None, dict]] = {}

    for symbol in dataset.symbols:
        full_daily = prepared_daily.get(symbol)
        hourly = dataset.hourly.get(symbol)
        if full_daily is None or hourly is None or hourly.empty:
            continue

        for position, index_value in enumerate(full_daily.index):
            signal_date = _date_from_index(index_value)
            if signal_date < config.start_date or signal_date > config.end_date:
                continue
            history = full_daily.iloc[: position + 1]
            if len(history) < max(scanner.EMA_SLOW, 100):
                continue

            trend = (
                trend_contexts.get((symbol, signal_date))
                if trend_contexts is not None
                else scanner._extract_ema_trend(hourly, through_date=signal_date)
            )
            if not trend:
                continue
            market_daily, market_regime = _regime_context(
                scanner.MARKET_PROXY,
                signal_date,
                prepared_daily,
                dataset.hourly,
                regime_cache,
                trend_contexts,
            )
            sector_proxy = dataset.sector_proxies.get(symbol)
            sector_daily = None
            sector_regime = None
            if sector_proxy:
                sector_daily, sector_regime = _regime_context(
                    sector_proxy,
                    signal_date,
                    prepared_daily,
                    dataset.hourly,
                    regime_cache,
                    trend_contexts,
                )
            rs_context = scanner.compute_relative_strength_context(
                history,
                market_daily,
                sector_daily,
            )
            signal = scanner._compute_signal_for_df(
                history,
                trend,
                market_regime,
                sector_regime,
                rs_context,
            )
            if signal is None:
                continue
            side, close, adx, score, metadata = signal
            if score < config.min_score:
                continue
            by_date.setdefault(signal_date, []).append(
                BacktestCandidate(
                    symbol=symbol,
                    side=side,
                    signal_date=signal_date,
                    position=position,
                    decision_price=close,
                    adx=adx,
                    score=score,
                    metadata=metadata,
                )
            )
    return by_date


def _score_band(score: float) -> str:
    band = min(max(int(score * 10), 0), 9)
    return f"{band / 10:.1f}-{(band + 1) / 10:.1f}"


def evaluate_candidates(
    by_date: dict[datetime.date, list[BacktestCandidate]],
    prepared_daily: dict[str, pd.DataFrame],
    config: BacktestConfig,
) -> pd.DataFrame:
    records = []
    one_r_usd = config.capital_per_trade * config.risk_fraction
    for signal_date in sorted(by_date):
        ranked = sorted(
            by_date[signal_date],
            key=lambda candidate: (-candidate.score, candidate.symbol, candidate.side),
        )[: config.max_alerts_per_day]
        for rank, candidate in enumerate(ranked, start=1):
            full_daily = prepared_daily[candidate.symbol]
            exit_position = candidate.position + config.hold_days
            if exit_position >= len(full_daily):
                continue
            entry_position = candidate.position + 1
            entry_date = _date_from_index(full_daily.index[entry_position])
            exit_date = _date_from_index(full_daily.index[exit_position])
            if exit_date > config.end_date:
                continue

            entry_price = scanner.to_float(full_daily["Open"].iloc[entry_position])
            exit_price = scanner.to_float(full_daily["Close"].iloc[exit_position])
            if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(exit_price):
                continue
            return_decimal = (exit_price - entry_price) / entry_price
            if candidate.side == "SELL":
                return_decimal = -return_decimal
            profit_usd = return_decimal * config.capital_per_trade
            records.append(
                {
                    "signal_date": signal_date.isoformat(),
                    "entry_date": entry_date.isoformat(),
                    "exit_date": exit_date.isoformat(),
                    "symbol": candidate.symbol,
                    "side": candidate.side,
                    "universe_rank": rank,
                    "score": candidate.score,
                    "score_band": _score_band(candidate.score),
                    "adx": candidate.adx,
                    "decision_price": candidate.decision_price,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return_decimal": return_decimal,
                    "profit_usd": profit_usd,
                    "r_multiple": profit_usd / one_r_usd,
                    "avg_dollar_vol_m": candidate.metadata.get("avg_dollar_vol_m", np.nan),
                    "trend4h_slope": candidate.metadata.get("trend4h_slope", np.nan),
                }
            )
    return pd.DataFrame(records)


def run_backtest(
    dataset: MarketDataset,
    config: BacktestConfig,
    *,
    prepared_daily: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if config.start_date > config.end_date:
        raise ValueError("start_date must not be after end_date")
    prepared = prepared_daily or prepare_daily_frames(dataset)
    with scanner_settings(config):
        trends = precompute_trend_contexts(dataset, prepared, config)
        candidates = collect_candidates(dataset, config, prepared, trends)
        return evaluate_candidates(candidates, prepared, config)


def summarize_trades(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "total_pnl_usd": 0.0,
            "total_r": 0.0,
            "average_pnl_usd": 0.0,
            "profit_factor": None,
            "large_loss_pct": 0.0,
            "sharpe": None,
        }
    profits = pd.to_numeric(trades["profit_usd"], errors="coerce").dropna()
    r_values = pd.to_numeric(trades["r_multiple"], errors="coerce").dropna()
    gross_profit = float(profits[profits > 0].sum())
    gross_loss = abs(float(profits[profits < 0].sum()))
    daily = (
        trades.assign(_profit=pd.to_numeric(trades["profit_usd"], errors="coerce"))
        .groupby("signal_date")["_profit"]
        .sum()
    )
    sharpe = None
    if len(daily) > 1 and daily.std(ddof=1) > 0:
        sharpe = float(math.sqrt(252) * daily.mean() / daily.std(ddof=1))
    return {
        "trades": len(trades),
        "win_rate_pct": float((profits > 0).mean() * 100),
        "total_pnl_usd": float(profits.sum()),
        "total_r": float(r_values.sum()),
        "average_pnl_usd": float(profits.mean()),
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "large_loss_pct": float((r_values < -5).mean() * 100),
        "sharpe": sharpe,
    }


def monthly_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "trades",
                "win_rate_pct",
                "total_pnl_usd",
                "total_r",
                "profit_factor",
            ]
        )
    records = []
    for month, group in trades.groupby(trades["signal_date"].str.slice(0, 7), sort=True):
        summary = summarize_trades(group)
        records.append({"month": month, **summary})
    return pd.DataFrame(records)


def config_to_dict(config: BacktestConfig) -> dict:
    result = asdict(config)
    result["start_date"] = config.start_date.isoformat()
    result["end_date"] = config.end_date.isoformat()
    return result


def write_results(
    trades: pd.DataFrame,
    config: BacktestConfig,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output_dir / "trades_2026.csv", index=False)
    monthly_summary(trades).to_csv(output_dir / "monthly_2026.csv", index=False)
    summary = summarize_trades(trades)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(config_to_dict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _date_argument(value: str) -> datetime.date:
    try:
        return datetime.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=_date_argument, default=datetime.date(2026, 1, 1))
    parser.add_argument("--end", type=_date_argument, default=datetime.date.today())
    parser.add_argument("--symbols", help="Comma-separated symbols instead of the index universe")
    parser.add_argument("--limit", type=int, default=0, help="Alphabetical universe limit")
    parser.add_argument("--hold", type=int, default=scanner.HOLD_DAYS)
    parser.add_argument("--capital", type=float, default=scanner.CAPITAL_PER_TRADE)
    parser.add_argument("--max-alerts", type=int, default=scanner.MAX_ALERTS_PER_CYCLE)
    parser.add_argument("--adx-min", type=float, default=scanner.ADX_MIN)
    parser.add_argument("--adx-hard-min", type=float, default=scanner.ADX_HARD_MIN)
    parser.add_argument("--min-score", type=float, default=scanner.MIN_RANK_SCORE)
    parser.add_argument("--slope", type=float, default=scanner.FOURH_SLOPE_MIN)
    parser.add_argument("--min-dollar-vol", type=float, default=scanner.MIN_DOLLAR_VOL_M)
    parser.add_argument("--max-dollar-vol", type=float, default=scanner.MAX_DOLLAR_VOL_M)
    parser.add_argument("--offset", default=scanner.RESAMPLE_4H_OFFSET)
    parser.add_argument("--daily-period", default="5y")
    parser.add_argument("--hourly-period", default="730d")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--save-cache", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit < 0:
        raise ValueError("--limit cannot be negative")
    requested = _parse_symbols(args.symbols)
    if args.use_cache:
        dataset = load_dataset_cache(args.cache_dir).subset(requested, args.limit)
    else:
        dataset = download_dataset(
            requested_symbols=requested,
            limit=args.limit,
            daily_period=args.daily_period,
            hourly_period=args.hourly_period,
        )
        if args.save_cache:
            cache_path = save_dataset_cache(dataset, args.cache_dir)
            scanner.logging.info("Saved raw backtest cache to %s", cache_path)

    scanner.SECTOR_PROXY_BY_SYMBOL.update(dataset.sector_proxies)
    config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        hold_days=args.hold,
        capital_per_trade=args.capital,
        max_alerts_per_day=args.max_alerts,
        adx_min=args.adx_min,
        adx_hard_min=args.adx_hard_min,
        min_score=args.min_score,
        fourh_slope_min=args.slope,
        min_dollar_vol_m=args.min_dollar_vol,
        max_dollar_vol_m=args.max_dollar_vol,
        resample_offset=args.offset,
    )
    trades = run_backtest(dataset, config)
    summary = write_results(trades, config, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Results: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
