import datetime

import numpy as np
import pandas as pd
import pytest

import ema_scanner as scanner


def test_select_batch_caps_at_one_universe_and_reports_completion():
    tickers = ["A", "B", "C"]

    batch, next_offset, completed = scanner._select_batch(tickers, 0, 125)
    assert batch == tickers
    assert next_offset == 0
    assert completed is True

    batch, next_offset, completed = scanner._select_batch(tickers, 2, 2)
    assert batch == ["C"]
    assert next_offset == 0
    assert completed is True

    with pytest.raises(ValueError, match="greater than zero"):
        scanner._select_batch(tickers, 0, 0)


def test_validate_config_reports_multiple_invalid_settings(monkeypatch):
    monkeypatch.setattr(scanner, "EMA_FAST", 30)
    monkeypatch.setattr(scanner, "EMA_SLOW", 20)
    monkeypatch.setattr(scanner, "SCAN_BATCH_SIZE", 0)

    with pytest.raises(ValueError) as error:
        scanner.validate_config()

    assert "EMA_FAST must be smaller than EMA_SLOW" in str(error.value)
    assert "SCAN_BATCH_SIZE must be > 0" in str(error.value)


def test_validate_config_rejects_new_gates(monkeypatch):
    monkeypatch.setattr(scanner, "ADX_HARD_MIN", -1.0)
    monkeypatch.setattr(scanner, "MAX_DOLLAR_VOL_M", scanner.MIN_DOLLAR_VOL_M - 1.0)
    monkeypatch.setattr(scanner, "FOURH_SLOPE_MIN", -0.001)

    with pytest.raises(ValueError) as error:
        scanner.validate_config()

    message = str(error.value)
    assert "ADX_MIN and ADX_HARD_MIN must be between 0 and 100" in message
    assert "MAX_DOLLAR_VOL_M must be 0 (disabled) or >= MIN_DOLLAR_VOL_M" in message
    assert "4H slope and stretch settings cannot be negative" in message


def test_rank_components_use_configurable_weights(monkeypatch):
    monkeypatch.setattr(
        scanner,
        "RANK_COMPONENT_WEIGHTS",
        {"strong": 3.0, "weak": 1.0, "disabled": 0.0},
    )

    score = scanner._weighted_rank_score({"strong": 1.0, "weak": 0.0, "disabled": 1.0})

    assert score == pytest.approx(0.75)


def test_resample_to_4h_anchors_candles_at_market_open(monkeypatch):
    monkeypatch.setattr(scanner, "RESAMPLE_4H_RULE", "4h")
    monkeypatch.setattr(scanner, "RESAMPLE_4H_OFFSET", "9h30min")
    index = pd.date_range(
        "2026-07-20 09:30",
        periods=7,
        freq="h",
        tz="America/New_York",
    )
    frame = pd.DataFrame(
        {
            "Open": np.arange(7, dtype=float),
            "High": np.arange(7, dtype=float) + 2,
            "Low": np.arange(7, dtype=float) - 1,
            "Close": np.arange(7, dtype=float) + 1,
            "Volume": np.ones(7),
        },
        index=index,
    )

    result = scanner.resample_to_4h(frame)

    assert [timestamp.strftime("%H:%M") for timestamp in result.index] == ["09:30", "13:30"]
    assert result.iloc[0].to_dict() == {
        "Open": 0.0,
        "High": 5.0,
        "Low": -1.0,
        "Close": 4.0,
        "Volume": 4.0,
    }
    assert result.iloc[1]["Volume"] == 3.0


def test_session_offset_london_and_asia_anchor(monkeypatch):
    monkeypatch.setattr(scanner, "RESAMPLE_4H_RULE", "4h")

    def anchored_times(start: str, offset: str):
        monkeypatch.setattr(scanner, "RESAMPLE_4H_OFFSET", offset)
        index = pd.date_range(start, periods=8, freq="h", tz="UTC")
        values = np.arange(8, dtype=float)
        frame = pd.DataFrame(
            {
                "Open": values,
                "High": values + 2,
                "Low": values - 1,
                "Close": values + 1,
                "Volume": np.ones(8),
            },
            index=index,
        )
        result = scanner.resample_to_4h(frame)
        return [timestamp.strftime("%H:%M") for timestamp in result.index]

    assert anchored_times("2026-07-20 03:00", "3h") == ["03:00", "07:00"]
    assert anchored_times("2026-07-20 20:00", "20h") == ["20:00", "00:00"]


def test_download_makes_one_attempt_when_retries_are_disabled(monkeypatch):
    calls = []
    frame = pd.DataFrame({"Close": [100.0]})

    def fake_download(*args, **kwargs):
        calls.append((args, kwargs))
        return frame

    monkeypatch.setattr(scanner, "MAX_RETRIES", 0)
    monkeypatch.setattr(scanner.yf, "download", fake_download)

    result = scanner._download_batch_chunked(
        ["AAPL"],
        period="5d",
        interval="1d",
        label="test",
    )

    assert result["AAPL"] is frame
    assert len(calls) == 1


def test_relative_strength_scores_short_and_long_directions_separately():
    index = pd.bdate_range("2026-01-01", periods=80)
    stock = pd.DataFrame({"Close": np.linspace(100, 70, len(index))}, index=index)
    market = pd.DataFrame({"Close": np.linspace(100, 120, len(index))}, index=index)

    context = scanner.compute_relative_strength_context(stock, market)

    assert context["vs_market_long"] is False
    assert context["vs_market_short"] is True
    assert context["score_long"] == 0.0
    assert context["score_short"] == 1.0


def test_nasdaq100_parser_uses_official_nested_rows(monkeypatch):
    rows = [{"symbol": f"T{index:03d}"} for index in range(103)]
    monkeypatch.setattr(
        scanner,
        "safe_get_json",
        lambda _url: {"data": {"data": {"rows": rows}}},
    )

    tickers = scanner.get_nasdaq100_tickers()

    assert len(tickers) == 103
    assert tickers[:2] == ["T000", "T001"]


def test_market_regime_ranks_candidates_but_liquidity_remains_a_gate(monkeypatch):
    rows = 120
    macd_hist = np.full(rows, 0.1)
    macd_hist[-4:] = [0.1, 0.2, 0.3, 0.4]
    ema_slow = np.full(rows, 95.0)
    ema_slow[-6:] = [95, 96, 97, 98, 99, 100]
    adx = np.full(rows, 30.0)
    adx[-1] = 35.0
    frame = pd.DataFrame(
        {
            "Close": np.linspace(100, 110, rows),
            "adx": adx,
            "atr": np.full(rows, 2.0),
            "macd_hist": macd_hist,
            "ema_slow": ema_slow,
            "avg_dollar_vol_20": np.full(rows, 100.0),
        }
    )

    monkeypatch.setattr(scanner, "_pullback_reclaim_buy", lambda _frame: (True, 0.0, 0.9))
    monkeypatch.setattr(scanner, "_pullback_reclaim_sell", lambda _frame: (False, 0.0, 0.0))
    monkeypatch.setattr(scanner, "USE_MARKET_REGIME_SCORE", True)
    monkeypatch.setattr(scanner, "USE_SECTOR_REGIME_SCORE", False)
    monkeypatch.setattr(scanner, "USE_RELATIVE_STRENGTH_SCORE", False)
    monkeypatch.setattr(scanner, "USE_OBV", False)

    trend = {"long_ok": True, "short_ok": False, "slope_4h": 0.01}
    blocked_market = {"long_ok": False, "short_ok": False, "score_long": 0.0}
    relative_strength = {"score_long": 1.0, "score_short": 0.0}

    weak_signal = scanner._compute_signal_for_df(
        frame.copy(),
        trend,
        blocked_market,
        None,
        relative_strength,
    )
    assert weak_signal is not None

    allowed_market = {"long_ok": True, "short_ok": False, "score_long": 1.0}
    signal = scanner._compute_signal_for_df(
        frame.copy(),
        trend,
        allowed_market,
        None,
        relative_strength,
    )
    assert signal is not None
    assert signal[0] == "BUY"
    assert signal[3] > weak_signal[3]

    illiquid = frame.copy()
    illiquid["avg_dollar_vol_20"] = 0.0
    assert (
        scanner._compute_signal_for_df(
            illiquid,
            trend,
            allowed_market,
            None,
            relative_strength,
        )
        is None
    )


def test_adx_hard_gate_blocks_weak_trend(monkeypatch):
    rows = 120
    frame = pd.DataFrame(
        {
            "Close": np.linspace(100, 110, rows),
            "adx": np.full(rows, scanner.ADX_HARD_MIN - 0.01),
            "atr": np.full(rows, 2.0),
            "macd_hist": np.linspace(0.01, 0.20, rows),
            "ema_slow": np.linspace(95, 100, rows),
            "avg_dollar_vol_20": np.full(rows, 100.0),
        }
    )
    monkeypatch.setattr(scanner, "_pullback_reclaim_buy", lambda _frame: (True, 0.0, 0.9))
    monkeypatch.setattr(scanner, "_pullback_reclaim_sell", lambda _frame: (False, 0.0, 0.0))

    signal = scanner._compute_signal_for_df(
        frame,
        {"long_ok": True, "short_ok": False, "slope_4h": 0.01},
        {},
        None,
        {},
    )

    assert signal is None


def test_volume_max_gate_blocks_mega_cap(monkeypatch):
    rows = 120
    macd_hist = np.full(rows, 0.1)
    macd_hist[-4:] = [0.1, 0.2, 0.3, 0.4]
    frame = pd.DataFrame(
        {
            "Close": np.linspace(100, 110, rows),
            "adx": np.full(rows, max(scanner.ADX_HARD_MIN, scanner.ADX_MIN) + 1.0),
            "atr": np.full(rows, 2.0),
            "macd_hist": macd_hist,
            "ema_slow": np.linspace(95, 100, rows),
            "avg_dollar_vol_20": np.full(rows, scanner.MAX_DOLLAR_VOL_M + 1.0),
        }
    )
    monkeypatch.setattr(scanner, "_pullback_reclaim_buy", lambda _frame: (True, 0.0, 0.9))
    monkeypatch.setattr(scanner, "_pullback_reclaim_sell", lambda _frame: (False, 0.0, 0.0))
    monkeypatch.setattr(scanner, "USE_MARKET_REGIME_SCORE", False)
    monkeypatch.setattr(scanner, "USE_SECTOR_REGIME_SCORE", False)
    monkeypatch.setattr(scanner, "USE_RELATIVE_STRENGTH_SCORE", False)

    trend = {"long_ok": True, "short_ok": False, "slope_4h": 0.01}
    assert scanner._compute_signal_for_df(frame, trend, {}, None, {}) is None

    monkeypatch.setattr(scanner, "MAX_DOLLAR_VOL_M", 0.0)
    assert scanner._compute_signal_for_df(frame, trend, {}, None, {}) is not None


def test_earnings_filter_blocks_lead_in_but_allows_next_session(monkeypatch):
    today = datetime.date.today()
    frame = pd.DataFrame(
        {
            "Open": [99.0, 100.0],
            "Close": [100.0, 101.0],
            "atr": [2.0, 2.0],
        }
    )
    monkeypatch.setattr(scanner, "ENABLE_EVENT_FILTER", True)
    monkeypatch.setattr(scanner, "EARNINGS_SKIP_DAYS_BEFORE", 3)
    monkeypatch.setattr(scanner, "EARNINGS_SKIP_DAYS_AFTER", 0)

    monkeypatch.setattr(
        scanner, "get_nearby_earnings_date", lambda _symbol: today + datetime.timedelta(days=3)
    )
    assert scanner.passes_event_filter("AAPL", "BUY", frame, None)[0] is False

    monkeypatch.setattr(scanner, "get_nearby_earnings_date", lambda _symbol: today)
    assert scanner.passes_event_filter("AAPL", "BUY", frame, None)[0] is False

    monkeypatch.setattr(
        scanner, "get_nearby_earnings_date", lambda _symbol: today - datetime.timedelta(days=1)
    )
    assert scanner.passes_event_filter("AAPL", "BUY", frame, None)[0] is True


def test_soft_technical_confirmations_rank_but_do_not_veto_setup(monkeypatch):
    rows = 120
    frame = pd.DataFrame(
        {
            "Close": np.linspace(100, 110, rows),
            "adx": np.full(rows, max(scanner.ADX_HARD_MIN, scanner.ADX_MIN) + 1.0),
            "atr": np.full(rows, 2.0),
            "macd_hist": np.full(rows, -0.1),
            "ema_slow": np.full(rows, 100.0),
            "avg_dollar_vol_20": np.full(rows, 100.0),
        }
    )
    monkeypatch.setattr(scanner, "_pullback_reclaim_buy", lambda _frame: (True, 2.0, 0.9))
    monkeypatch.setattr(scanner, "_pullback_reclaim_sell", lambda _frame: (False, 0.0, 0.0))
    monkeypatch.setattr(scanner, "USE_MARKET_REGIME_SCORE", False)
    monkeypatch.setattr(scanner, "USE_SECTOR_REGIME_SCORE", False)
    monkeypatch.setattr(scanner, "USE_RELATIVE_STRENGTH_SCORE", False)
    monkeypatch.setattr(scanner, "USE_ADX_CONFIRM", True)
    monkeypatch.setattr(scanner, "USE_OBV", False)

    signal = scanner._compute_signal_for_df(
        frame,
        {"long_ok": True, "short_ok": False, "slope_4h": scanner.FOURH_SLOPE_MIN},
        {},
        None,
        {},
    )

    assert signal is not None
    assert signal[0] == "BUY"
    assert 0 <= signal[3] <= 1
    assert signal[4]["technical_votes"] == 0
    assert signal[4]["technical_vote_count"] == 4


def test_incomplete_daily_bar_is_not_eligible_until_after_close(monkeypatch):
    monkeypatch.setattr(scanner, "DAILY_BAR_CLOSE_BUFFER_MINUTES", 15)
    frame = pd.DataFrame(
        {"Close": [100.0, 101.0]},
        index=pd.to_datetime(["2026-07-17", "2026-07-20"]),
    )

    before_close = datetime.datetime(2026, 7, 20, 15, 59, tzinfo=scanner.NEW_YORK_TZ)
    after_buffer = datetime.datetime(2026, 7, 20, 16, 15, tzinfo=scanner.NEW_YORK_TZ)

    assert list(scanner._drop_incomplete_daily_bar(frame, now=before_close)["Close"]) == [100.0]
    assert list(scanner._drop_incomplete_daily_bar(frame, now=after_buffer)["Close"]) == [
        100.0,
        101.0,
    ]


def test_proxy_context_is_reused_until_cache_expiry(monkeypatch):
    scanner.PROXY_CACHE.clear()
    monkeypatch.setattr(scanner, "MARKET_PROXY", "SPY")
    monkeypatch.setattr(scanner, "PROXY_CACHE_MINUTES", 15)
    monkeypatch.setitem(scanner.SECTOR_PROXY_BY_SYMBOL, "AAPL", "XLK")
    calls = []

    def fake_download(tickers, *, period, interval, label):
        calls.append((tuple(tickers), period, interval, label))
        return {ticker: pd.DataFrame({"Close": [1.0]}) for ticker in tickers}

    monkeypatch.setattr(scanner, "_download_batch_chunked", fake_download)
    monkeypatch.setattr(scanner, "_prepare_daily_df", lambda frame: frame)
    monkeypatch.setattr(
        scanner,
        "_extract_ema_trend",
        lambda _frame, **_kwargs: {"long_ok": True},
    )
    monkeypatch.setattr(
        scanner,
        "build_regime_context",
        lambda _daily, _trend, label: {"label": label, "long_ok": True},
    )

    first = scanner._fetch_proxy_maps_for_batch(["AAPL"])
    second = scanner._fetch_proxy_maps_for_batch(["AAPL"])

    assert first == second
    assert len(calls) == 2  # one daily and one hourly request, only on the first call


def test_scan_symbol_aligns_trend_and_returns_unpersisted_candidate(monkeypatch):
    signal_date = datetime.date(2026, 7, 17)
    prepared = pd.DataFrame(
        {"Close": np.linspace(90.0, 100.0, 120)},
        index=pd.bdate_range(end=signal_date, periods=120),
    )
    observed = {}

    monkeypatch.setattr(scanner, "_prepare_daily_df", lambda _frame: prepared)

    def fake_trend(_frame, *, through_date=None):
        observed["through_date"] = through_date
        return {"long_ok": True}

    monkeypatch.setattr(scanner, "_extract_ema_trend", fake_trend)
    monkeypatch.setattr(scanner, "infer_sector_proxy", lambda _symbol: None)
    monkeypatch.setattr(scanner, "compute_relative_strength_context", lambda *_args: {})
    monkeypatch.setattr(
        scanner,
        "_compute_signal_for_df",
        lambda *_args: ("BUY", 100.0, 25.0, 0.6, {"setup": "TEST"}),
    )
    monkeypatch.setattr(scanner, "passes_event_filter", lambda *_args: (True, "ok"))
    monkeypatch.setattr(scanner, "ENABLE_OPENINSIDER", False)
    monkeypatch.setattr(scanner, "MIN_RANK_SCORE", 0.0)

    candidate = scanner._scan_symbol(
        "AAPL",
        {"AAPL": pd.DataFrame({"Close": [1.0]})},
        {"AAPL": pd.DataFrame({"Close": [1.0]})},
        {},
        {},
        pd.DataFrame({"Close": [1.0]}),
        {},
    )

    assert isinstance(candidate, scanner.SignalCandidate)
    assert candidate.signal_date == signal_date
    assert candidate.score == 0.6
    assert candidate.metadata["entry_rule"] == "NEXT_SESSION_OPEN"
    assert observed["through_date"] == signal_date


def test_batch_continues_after_one_symbol_raises(monkeypatch):
    monkeypatch.setattr(
        scanner,
        "_download_batch_chunked",
        lambda tickers, **_kwargs: {ticker: pd.DataFrame({"Close": [1.0]}) for ticker in tickers},
    )
    monkeypatch.setattr(
        scanner,
        "_fetch_proxy_maps_for_batch",
        lambda _batch: ({}, {scanner.MARKET_PROXY: {"long_ok": False}}),
    )

    def fake_scan_symbol(symbol, *_args):
        if symbol == "BAD":
            raise ValueError("bad data")
        return scanner.SignalCandidate(
            symbol=symbol,
            side="BUY",
            signal_date=datetime.date(2026, 7, 17),
            decision_price=100.0,
            adx=25.0,
            score=0.5,
            metadata={},
        )

    monkeypatch.setattr(scanner, "_scan_symbol", fake_scan_symbol)

    signals, next_offset, completed = scanner._scan_tickers_batched_unlocked(
        ["BAD", "GOOD"],
        batch_size=2,
    )

    assert [candidate.symbol for candidate in signals] == ["GOOD"]
    assert next_offset == 0
    assert completed is True
    assert "BAD: ValueError: bad data" in scanner.LAST_SCAN_SUMMARY["last_error"]


def test_record_signal_deduplicates_across_repeated_scans(tmp_path, monkeypatch):
    log_path = tmp_path / "signals.csv"
    monkeypatch.setattr(scanner, "LOG_FILE", str(log_path))
    monkeypatch.setattr(scanner, "_RECORDED_SIGNAL_KEYS", None)

    assert scanner.record_signal("BUY", "AAPL", 200.0, 0.8, {"setup": "TEST"}) is True
    assert scanner.record_signal("BUY", "AAPL", 201.0, 0.9, {"setup": "TEST"}) is False

    saved = pd.read_csv(log_path)
    assert len(saved) == 1
    assert saved.iloc[0]["price"] == 200.0


def test_ranked_candidates_are_sorted_capped_and_logged_for_next_open(tmp_path, monkeypatch):
    log_path = tmp_path / "ranked-signals.csv"
    results_path = tmp_path / "ranked-results.csv"
    monkeypatch.setattr(scanner, "LOG_FILE", str(log_path))
    monkeypatch.setattr(scanner, "RESULTS_LOG_FILE", str(results_path))
    monkeypatch.setattr(scanner, "MAX_ALERTS_PER_CYCLE", 2)
    monkeypatch.setattr(scanner, "CALIBRATION_MIN_SAMPLES", 2)
    monkeypatch.setattr(scanner, "HOLD_DAYS", 5)
    monkeypatch.setattr(scanner, "RANK_SCORE_VERSION", "v1")
    monkeypatch.setattr(scanner, "_RECORDED_SIGNAL_KEYS", None)
    signal_date = datetime.date(2026, 7, 17)
    pd.DataFrame(
        {
            "score": [0.91, 0.92],
            "score_version": ["v1", "v1"],
            "signal": ["BUY", "BUY"],
            "hold_sessions": [5, 5],
            "return_decimal": [0.01, -0.01],
        }
    ).to_csv(results_path, index=False)

    candidates = [
        scanner.SignalCandidate(
            symbol=symbol,
            side="BUY",
            signal_date=signal_date,
            decision_price=price,
            adx=25.0,
            score=score,
            metadata={
                "technical_votes": 1,
                "technical_vote_count": 4,
                "score_components": {"ema_slope": score},
                "technical_evidence": {"ema_slope": True},
            },
        )
        for symbol, price, score in (
            ("LOW", 10.0, 0.2),
            ("TOP", 20.0, 0.9),
            ("MID", 30.0, 0.6),
        )
    ]

    alerts = scanner._finalize_ranked_candidates(
        candidates,
        now=datetime.datetime(2026, 7, 17, 17, 0, tzinfo=scanner.NEW_YORK_TZ),
    )

    assert [alert.split()[2] for alert in alerts] == ["TOP", "MID"]
    assert all("entry next open" in alert for alert in alerts)
    assert "HIST 5S 50% (n=2)" in alerts[0]
    assert "HIST" not in alerts[1]
    saved = pd.read_csv(log_path)
    assert list(saved["symbol"]) == ["TOP", "MID"]
    assert list(saved["entry_rule"]) == ["NEXT_SESSION_OPEN", "NEXT_SESSION_OPEN"]
    assert list(saved["universe_rank"]) == [1, 2]
    assert list(saved["decision_price"]) == [20.0, 30.0]
    assert list(saved["component_ema_slope"]) == [0.9, 0.6]
    assert saved["evidence_ema_slope"].all()
    assert set(saved["date"]) == {signal_date.isoformat()}


def test_min_rank_score_filters_low_quality_at_delivery(tmp_path, monkeypatch):
    log_path = tmp_path / "score-filtered-signals.csv"
    monkeypatch.setattr(scanner, "LOG_FILE", str(log_path))
    monkeypatch.setattr(scanner, "RESULTS_LOG_FILE", str(tmp_path / "score-results.csv"))
    monkeypatch.setattr(scanner, "MIN_RANK_SCORE", 0.40)
    monkeypatch.setattr(scanner, "MAX_ALERTS_PER_CYCLE", 10)
    monkeypatch.setattr(scanner, "_RECORDED_SIGNAL_KEYS", None)
    signal_date = datetime.date(2026, 7, 17)
    candidates = [
        scanner.SignalCandidate(
            symbol=symbol,
            side="BUY",
            signal_date=signal_date,
            decision_price=100.0,
            adx=25.0,
            score=score,
            metadata={},
        )
        for symbol, score in (("PASS", 0.40), ("BLOCK", 0.3999))
    ]

    alerts = scanner._finalize_ranked_candidates(
        candidates,
        now=datetime.datetime(2026, 7, 17, 17, 0, tzinfo=scanner.NEW_YORK_TZ),
    )

    assert len(alerts) == 1
    assert "PASS" in alerts[0]
    saved = pd.read_csv(log_path)
    assert list(saved["symbol"]) == ["PASS"]


def test_ranked_candidate_is_not_alerted_after_its_next_open(tmp_path, monkeypatch):
    log_path = tmp_path / "stale-signals.csv"
    monkeypatch.setattr(scanner, "LOG_FILE", str(log_path))
    monkeypatch.setattr(scanner, "RESULTS_LOG_FILE", str(tmp_path / "stale-results.csv"))
    monkeypatch.setattr(scanner, "_RECORDED_SIGNAL_KEYS", None)
    candidate = scanner.SignalCandidate(
        symbol="LATE",
        side="BUY",
        signal_date=datetime.date(2026, 7, 17),
        decision_price=20.0,
        adx=25.0,
        score=0.8,
        metadata={},
    )

    alerts = scanner._finalize_ranked_candidates(
        [candidate],
        now=datetime.datetime(2026, 7, 20, 10, 0, tzinfo=scanner.NEW_YORK_TZ),
    )

    assert alerts == []
    assert not log_path.exists()


def test_record_signal_refuses_to_append_to_malformed_log(tmp_path, monkeypatch):
    log_path = tmp_path / "signals.csv"
    original = "wrong,columns\n1,2\n"
    log_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(scanner, "LOG_FILE", str(log_path))
    monkeypatch.setattr(scanner, "_RECORDED_SIGNAL_KEYS", None)

    assert scanner.record_signal("BUY", "AAPL", 200.0) is False
    assert log_path.read_text(encoding="utf-8") == original


def test_record_signal_migrates_legacy_log_columns(tmp_path, monkeypatch):
    log_path = tmp_path / "legacy-signals.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-07-16",
                "signal": "BUY",
                "symbol": "OLD",
                "price": 10.0,
                "confidence": 0.5,
                "setup": "TEST",
            }
        ]
    ).to_csv(log_path, index=False)
    monkeypatch.setattr(scanner, "LOG_FILE", str(log_path))
    monkeypatch.setattr(scanner, "_RECORDED_SIGNAL_KEYS", None)

    assert scanner.record_signal(
        "BUY",
        "NEW",
        20.0,
        0.7,
        {"entry_rule": "NEXT_SESSION_OPEN"},
        signal_date=datetime.date(2026, 7, 17),
    )

    migrated = pd.read_csv(log_path)
    assert list(migrated["symbol"]) == ["OLD", "NEW"]
    assert "score" in migrated.columns
    assert migrated.loc[migrated["symbol"] == "NEW", "score"].iloc[0] == 0.7
    assert migrated.loc[migrated["symbol"] == "OLD", "confidence"].iloc[0] == 0.5


def test_evaluate_old_signals_uses_bars_after_the_signal_date(tmp_path, monkeypatch):
    signal_date = datetime.date.today() - datetime.timedelta(days=20)
    log_path = tmp_path / "signals.csv"
    pd.DataFrame(
        [
            {
                "date": signal_date.isoformat(),
                "signal": "BUY",
                "symbol": "AAPL",
                "price": 100.0,
                "confidence": 0.8,
                "setup": "TEST",
            }
        ]
    ).to_csv(log_path, index=False)

    index = pd.to_datetime(
        [
            signal_date - datetime.timedelta(days=2),
            signal_date - datetime.timedelta(days=1),
            signal_date,
            *[signal_date + datetime.timedelta(days=day) for day in range(1, 7)],
        ]
    )
    history = pd.DataFrame(
        {
            "Open": [90, 95, 100, 102, 103, 104, 105, 106, 107],
            "Close": [90, 95, 100, 103, 104, 105, 106, 107, 108],
        },
        index=index,
    )

    def fake_download(tickers, *, period, interval, label):
        assert tickers == ["AAPL"]
        assert interval == "1d"
        assert label == "backtest"
        return {"AAPL": history}

    monkeypatch.setattr(scanner, "LOG_FILE", str(log_path))
    results_path = tmp_path / "signal-results.csv"
    monkeypatch.setattr(scanner, "RESULTS_LOG_FILE", str(results_path))
    monkeypatch.setattr(scanner, "HOLD_DAYS", 5)
    monkeypatch.setattr(scanner, "CAPITAL_PER_TRADE", 500.0)
    monkeypatch.setattr(scanner, "_RECORDED_SIGNAL_KEYS", None)
    monkeypatch.setattr(scanner, "_download_batch_chunked", fake_download)

    report = scanner.evaluate_old_signals()

    assert "next-open entries" in report
    assert "102.00 -> 107.00" in report
    assert "+24.51 USD" in report
    assert pd.read_csv(log_path).empty
    saved_results = pd.read_csv(results_path)
    assert len(saved_results) == 1
    assert saved_results.iloc[0]["entry_price"] == 102.0
    assert saved_results.iloc[0]["exit_price"] == 107.0
    assert saved_results.iloc[0]["return_decimal"] == pytest.approx(5 / 102)


def test_result_ledger_is_idempotent_and_keeps_latest_evaluation(tmp_path, monkeypatch):
    results_path = tmp_path / "results.csv"
    monkeypatch.setattr(scanner, "RESULTS_LOG_FILE", str(results_path))
    base = {
        "signal_date": "2026-07-17",
        "signal": "BUY",
        "symbol": "AAPL",
        "hold_sessions": 5,
        "return_decimal": 0.01,
    }

    scanner._persist_signal_results([base])
    updated = dict(base, return_decimal=0.02)
    ledger = scanner._persist_signal_results([updated])

    assert len(ledger) == 1
    assert ledger.iloc[0]["return_decimal"] == 0.02
    assert pd.read_csv(results_path).iloc[0]["return_decimal"] == 0.02


def test_empirical_win_context_requires_same_side_version_and_sample_size(monkeypatch):
    monkeypatch.setattr(scanner, "CALIBRATION_MIN_SAMPLES", 20)
    monkeypatch.setattr(scanner, "HOLD_DAYS", 5)
    monkeypatch.setattr(scanner, "RANK_SCORE_VERSION", "v1")
    history = pd.DataFrame(
        {
            "score": [0.45] * 20,
            "score_version": ["v1"] * 20,
            "signal": ["BUY"] * 20,
            "hold_sessions": [5] * 20,
            "return_decimal": [0.01] * 12 + [-0.01] * 8,
        }
    )

    context = scanner._empirical_win_context(0.49, "BUY", history)

    assert context == {"win_rate": 0.6, "samples": 20, "score_band": "0.4-0.5"}
    assert scanner._empirical_win_context(0.49, "SELL", history) is None
    assert scanner._empirical_win_context(0.59, "BUY", history) is None


def test_discord_messages_are_split_without_exceeding_limit():
    message = "header\n" + "\n".join(["x" * 40] * 10)

    chunks = scanner._split_discord_message(message, limit=100)

    assert "".join(chunks).replace("\n", "") == message.replace("\n", "")
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_trade_messages_use_pinned_webhook(monkeypatch):
    calls = []

    class Response:
        status_code = 204
        text = ""

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr(scanner.requests, "post", fake_post)
    monkeypatch.setattr(scanner, "DISCORD_WEBHOOK", "https://example.invalid/old-status-hook")

    scanner.send_trade_discord_message("trade")

    assert [url for url, _kwargs in calls] == [scanner.TRADE_DISCORD_WEBHOOK]
    assert calls[0][1]["json"] == {"content": "trade"}


def test_alerts_and_performance_reports_use_trade_sender(monkeypatch):
    messages = []
    monkeypatch.setattr(scanner, "_finalize_ranked_candidates", lambda _candidates: ["trade"])
    monkeypatch.setattr(scanner, "evaluate_old_signals", lambda: "performance")
    monkeypatch.setattr(scanner, "send_trade_discord_message", messages.append)

    assert scanner._send_ranked_alerts([]) == 1
    scanner._send_performance_report()

    assert "performance" in messages
    assert len(messages) == 2
    assert messages[0].endswith("\ntrade")


def test_run_once_endpoint_requires_post_and_constant_secret(monkeypatch):
    monkeypatch.setattr(scanner, "RUN_TOKEN", "expected-secret")
    client = scanner.app.test_client()

    assert client.get("/run-once").status_code == 405
    assert client.post("/run-once").status_code == 403
    assert client.post("/run-once", headers={"X-Run-Token": "wrong"}).status_code == 403

    class NoopThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            scanner.MANUAL_SCAN_LOCK.release()

    monkeypatch.setattr(scanner, "Thread", NoopThread)
    response = client.post("/run-once", headers={"X-Run-Token": "expected-secret"})

    assert response.status_code == 202
    assert response.get_json() == {"ok": True, "status": "started"}
