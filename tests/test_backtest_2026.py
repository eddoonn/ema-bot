import pandas as pd

from backtest_2026 import (
    BacktestCandidate,
    BacktestConfig,
    MarketDataset,
    evaluate_candidates,
    load_dataset_cache,
    save_dataset_cache,
)


def test_backtest_ranks_cross_section_and_uses_next_open_to_hold_close():
    index = pd.bdate_range("2026-01-02", periods=8)
    low_frame = pd.DataFrame(
        {"Open": [100, 101, 102, 103, 104, 105, 106, 107], "Close": [100] * 8},
        index=index,
    )
    top_frame = pd.DataFrame(
        {"Open": [50, 51, 52, 53, 54, 55, 56, 57], "Close": [50, 51, 52, 53, 54, 60, 61, 62]},
        index=index,
    )
    signal_date = index[0].date()
    candidates = {
        signal_date: [
            BacktestCandidate("LOW", "BUY", signal_date, 0, 100.0, 25.0, 0.41, {}),
            BacktestCandidate("TOP", "BUY", signal_date, 0, 50.0, 30.0, 0.80, {}),
        ]
    }
    config = BacktestConfig(
        start_date=signal_date,
        end_date=index[-1].date(),
        hold_days=5,
        max_alerts_per_day=1,
    )

    trades = evaluate_candidates(candidates, {"LOW": low_frame, "TOP": top_frame}, config)

    assert list(trades["symbol"]) == ["TOP"]
    assert trades.iloc[0]["entry_price"] == 51
    assert trades.iloc[0]["exit_price"] == 60
    assert trades.iloc[0]["return_decimal"] == (60 - 51) / 51
    assert trades.iloc[0]["universe_rank"] == 1


def test_backtest_does_not_leak_exit_across_period_boundary():
    index = pd.bdate_range("2026-06-22", periods=8)
    frame = pd.DataFrame(
        {"Open": range(100, 108), "Close": range(100, 108)},
        index=index,
    )
    signal_date = index[1].date()
    candidate = BacktestCandidate(
        "TEST",
        "BUY",
        signal_date,
        1,
        101.0,
        25.0,
        0.50,
        {},
    )
    config = BacktestConfig(
        start_date=signal_date,
        end_date=index[5].date(),
        hold_days=5,
    )

    trades = evaluate_candidates({signal_date: [candidate]}, {"TEST": frame}, config)

    assert trades.empty


def test_market_cache_round_trip(tmp_path):
    frame = pd.DataFrame(
        {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [10]},
        index=pd.to_datetime(["2026-01-02"]),
    )
    dataset = MarketDataset(
        symbols=["TEST"],
        daily={"TEST": frame},
        hourly={"TEST": frame},
        sector_proxies={"TEST": "XLK"},
        downloaded_at="2026-01-03T00:00:00+00:00",
    )

    save_dataset_cache(dataset, tmp_path)
    restored = load_dataset_cache(tmp_path)

    assert restored.symbols == ["TEST"]
    assert restored.sector_proxies == {"TEST": "XLK"}
    pd.testing.assert_frame_equal(restored.daily["TEST"], frame)
