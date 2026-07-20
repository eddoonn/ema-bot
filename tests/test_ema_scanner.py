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


def test_required_market_regime_is_a_hard_signal_gate(monkeypatch):
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
    monkeypatch.setattr(scanner, "REQUIRE_MARKET_REGIME", True)
    monkeypatch.setattr(scanner, "REQUIRE_SECTOR_REGIME", False)
    monkeypatch.setattr(scanner, "REQUIRE_RELATIVE_STRENGTH", False)
    monkeypatch.setattr(scanner, "USE_OBV", False)

    trend = {"long_ok": True, "short_ok": False, "slope_4h": 0.01}
    blocked_market = {"long_ok": False, "short_ok": False, "score_long": 0.0}
    relative_strength = {"score_long": 1.0, "score_short": 0.0}

    assert (
        scanner._compute_signal_for_df(
            frame.copy(),
            trend,
            blocked_market,
            None,
            relative_strength,
        )
        is None
    )

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
    monkeypatch.setattr(scanner, "_extract_ema_trend", lambda _frame: {"long_ok": True})
    monkeypatch.setattr(
        scanner,
        "build_regime_context",
        lambda _daily, _trend, label: {"label": label, "long_ok": True},
    )

    first = scanner._fetch_proxy_maps_for_batch(["AAPL"])
    second = scanner._fetch_proxy_maps_for_batch(["AAPL"])

    assert first == second
    assert len(calls) == 2  # one daily and one hourly request, only on the first call


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
        return f"alert:{symbol}"

    monkeypatch.setattr(scanner, "_scan_symbol", fake_scan_symbol)

    signals, next_offset, completed = scanner._scan_tickers_batched_unlocked(
        ["BAD", "GOOD"],
        batch_size=2,
    )

    assert signals == ["alert:GOOD"]
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


def test_record_signal_refuses_to_append_to_malformed_log(tmp_path, monkeypatch):
    log_path = tmp_path / "signals.csv"
    original = "wrong,columns\n1,2\n"
    log_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(scanner, "LOG_FILE", str(log_path))
    monkeypatch.setattr(scanner, "_RECORDED_SIGNAL_KEYS", None)

    assert scanner.record_signal("BUY", "AAPL", 200.0) is False
    assert log_path.read_text(encoding="utf-8") == original


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
        {"Close": [90, 95, 100, 101, 102, 103, 104, 105, 106]},
        index=index,
    )

    def fake_download(tickers, *, period, interval, label):
        assert tickers == ["AAPL"]
        assert interval == "1d"
        assert label == "backtest"
        return {"AAPL": history}

    monkeypatch.setattr(scanner, "LOG_FILE", str(log_path))
    monkeypatch.setattr(scanner, "HOLD_DAYS", 5)
    monkeypatch.setattr(scanner, "CAPITAL_PER_TRADE", 500.0)
    monkeypatch.setattr(scanner, "_RECORDED_SIGNAL_KEYS", None)
    monkeypatch.setattr(scanner, "_download_batch_chunked", fake_download)

    report = scanner.evaluate_old_signals()

    assert "100.00 -> 105.00" in report
    assert "+25.00 USD" in report
    assert pd.read_csv(log_path).empty


def test_discord_messages_are_split_without_exceeding_limit():
    message = "header\n" + "\n".join(["x" * 40] * 10)

    chunks = scanner._split_discord_message(message, limit=100)

    assert "".join(chunks).replace("\n", "") == message.replace("\n", "")
    assert all(len(chunk) <= 100 for chunk in chunks)


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
