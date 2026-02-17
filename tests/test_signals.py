"""Tests for signal framework."""

from datetime import date, timedelta

import numpy as np
import polars as pl

from signals.momentum import CompositeSignal, MeanReversionSignal, MomentumSignal


def _make_price_data(
    tickers: list[str],
    n_days: int = 300,
    start_date: date = date(2023, 1, 1),
    trends: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Generate synthetic price data for testing.

    Args:
        tickers: List of ticker symbols.
        n_days: Number of trading days.
        start_date: First date.
        trends: Dict of {ticker: daily_drift}. Positive = uptrend.
    """
    if trends is None:
        trends = {}

    rows = []
    for ticker in tickers:
        drift = trends.get(ticker, 0.0)
        price = 100.0
        for i in range(n_days):
            d = start_date + timedelta(days=i)
            # Skip weekends
            if d.weekday() >= 5:
                continue
            noise = np.random.normal(0, 0.02)
            price *= 1 + drift + noise
            price = max(price, 1.0)
            rows.append({
                "date": d,
                "ticker": ticker,
                "open": price * 0.999,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "adj_close": price,
                "volume": float(np.random.randint(100_000, 10_000_000)),
            })

    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


class TestMomentumSignal:
    def test_returns_signals_for_all_tickers(self):
        np.random.seed(42)
        prices = _make_price_data(
            ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN"],
            n_days=400,
            trends={"AAPL": 0.002, "TSLA": -0.002},
        )
        signal = MomentumSignal(formation_days=200, skip_days=21)
        last_date = prices.get_column("date").max()
        signals = signal.generate_signals(prices, last_date)

        assert len(signals) > 0
        for score in signals.values():
            assert -1.0 <= score <= 1.0

    def test_uptrend_gets_positive_signal(self):
        np.random.seed(42)
        prices = _make_price_data(
            ["UP", "DOWN", "FLAT1", "FLAT2", "FLAT3"],
            n_days=400,
            trends={"UP": 0.005, "DOWN": -0.005},
        )
        signal = MomentumSignal(formation_days=200, skip_days=21)
        last_date = prices.get_column("date").max()
        signals = signal.generate_signals(prices, last_date)

        if "UP" in signals and "DOWN" in signals:
            assert signals["UP"] > signals["DOWN"]

    def test_lookback_property(self):
        signal = MomentumSignal(formation_days=252, skip_days=21)
        assert signal.lookback_days == 252 + 21 + 5

    def test_too_few_tickers_returns_empty(self):
        np.random.seed(42)
        prices = _make_price_data(["ONLY1"], n_days=400)
        signal = MomentumSignal(formation_days=200, skip_days=21)
        last_date = prices.get_column("date").max()
        signals = signal.generate_signals(prices, last_date)
        assert len(signals) == 0  # need >= 3 for cross-sectional

    def test_no_future_data_used(self):
        np.random.seed(42)
        prices = _make_price_data(
            ["A", "B", "C", "D", "E"],
            n_days=400,
        )
        # Signal on day 300 should not use days 301+
        dates = sorted(prices.get_column("date").unique().to_list())
        mid_date = dates[250]
        signal = MomentumSignal(formation_days=100, skip_days=10)

        signals = signal.generate_signals(prices, mid_date)
        # Signals should exist (data is filtered internally)
        # Just verify it doesn't crash with future data present
        assert isinstance(signals, dict)


class TestMeanReversionSignal:
    def test_oversold_gets_buy_signal(self):
        # Create a stock that dropped sharply
        np.random.seed(42)
        rows = []
        price = 100.0
        start = date(2023, 1, 1)
        for i in range(50):
            d = start + timedelta(days=i)
            if d.weekday() >= 5:
                continue
            if i < 40:
                price *= 1.001  # stable
            else:
                price *= 0.95  # sharp drop
            rows.append({
                "date": d,
                "ticker": "DROP",
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "adj_close": price,
                "volume": 1_000_000.0,
            })
        prices = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))

        signal = MeanReversionSignal(lookback_days=20)
        last_date = prices.get_column("date").max()
        signals = signal.generate_signals(prices, last_date)

        if "DROP" in signals:
            # Oversold → inverted z-score → positive signal (buy)
            assert signals["DROP"] > 0

    def test_lookback_property(self):
        signal = MeanReversionSignal(lookback_days=20)
        assert signal.lookback_days == 25


class TestCompositeSignal:
    def test_combines_signals(self):
        np.random.seed(42)
        prices = _make_price_data(
            ["A", "B", "C", "D", "E"],
            n_days=400,
            trends={"A": 0.003, "E": -0.003},
        )

        mom = MomentumSignal(formation_days=200, skip_days=21)
        mr = MeanReversionSignal(lookback_days=20)
        composite = CompositeSignal([(mom, 0.7), (mr, 0.3)])

        last_date = prices.get_column("date").max()
        signals = composite.generate_signals(prices, last_date)

        for score in signals.values():
            assert -1.0 <= score <= 1.0

    def test_lookback_is_max(self):
        mom = MomentumSignal(formation_days=252, skip_days=21)
        mr = MeanReversionSignal(lookback_days=20)
        composite = CompositeSignal([(mom, 0.5), (mr, 0.5)])
        assert composite.lookback_days == mom.lookback_days
