"""Tests for the HTML tear sheet generator."""

import os
import tempfile
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from engine.backtest import BacktestEngine
from reports.tearsheet import (
    _compute_drawdowns,
    _compute_monthly_returns,
    _heatmap_color,
    _rolling_sharpe,
    generate_tearsheet,
)
from signals.base import Signal


class AlwaysLongSignal(Signal):
    @property
    def lookback_days(self) -> int:
        return 5

    def generate_signals(self, prices, current_date):
        return {"AAPL": 0.5, "MSFT": -0.3}


def _make_simple_prices(n_days: int = 60) -> pl.DataFrame:
    rows = []
    start = date(2024, 1, 2)
    day_count = 0
    for i in range(n_days * 2):
        d = start + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        day_count += 1
        if day_count > n_days:
            break
        for ticker, base_price in [("AAPL", 150.0), ("MSFT", 350.0)]:
            price = base_price * (1 + 0.001 * day_count)
            rows.append({
                "date": d,
                "ticker": ticker,
                "open": price * 0.999,
                "high": price * 1.005,
                "low": price * 0.995,
                "close": price,
                "adj_close": price,
                "volume": 50_000_000.0,
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


# ---------------------------------------------------------------------------
# Unit tests for computation helpers
# ---------------------------------------------------------------------------

class TestComputeDrawdowns:
    def test_no_drawdown_on_uptrend(self):
        equity = [100, 101, 102, 103]
        dd = _compute_drawdowns(equity)
        assert all(d == 0 for d in dd)

    def test_drawdown_after_peak(self):
        equity = [100, 110, 100, 90]
        dd = _compute_drawdowns(equity)
        assert dd[0] == 0
        assert dd[1] == 0
        assert dd[2] == pytest.approx((100 - 110) / 110 * 100, abs=0.01)
        assert dd[3] == pytest.approx((90 - 110) / 110 * 100, abs=0.01)

    def test_recovery(self):
        equity = [100, 110, 100, 115]
        dd = _compute_drawdowns(equity)
        assert dd[3] == pytest.approx((115 - 115) / 115 * 100, abs=0.01)


class TestComputeMonthlyReturns:
    def test_groups_by_month(self):
        from engine.backtest import DailySnapshot

        snapshots = []
        for i in range(60):
            d = date(2024, 1, 2) + timedelta(days=i)
            if d.weekday() >= 5:
                continue
            snapshots.append(DailySnapshot(
                date=d, equity=1_000_000 + i * 100, cash=900_000,
                long_value=100_000, short_value=0, gross_value=100_000,
                net_value=100_000, num_positions=1, num_long=1, num_short=0,
                daily_return=0.001, cumulative_return=0.0,
                total_commission=0, total_spread_cost=0, total_impact_cost=0,
                total_slippage_cost=0, total_borrow_cost=0, trades_today=0,
            ))

        monthly = _compute_monthly_returns(snapshots)
        assert isinstance(monthly, dict)
        assert len(monthly) > 0
        for key in monthly:
            assert isinstance(key, tuple)
            assert len(key) == 2


class TestRollingSharpe:
    def test_none_before_window(self):
        returns = [0.001] * 100
        rs = _rolling_sharpe(returns, window=63)
        assert rs[0] is None
        assert rs[62] is None
        assert rs[63] is not None

    def test_positive_sharpe_for_positive_returns(self):
        returns = [0.001 + 0.0001 * np.random.randn() for _ in range(100)]
        np.random.seed(42)
        rs = _rolling_sharpe(returns, window=63)
        valid = [s for s in rs if s is not None]
        assert len(valid) > 0
        assert any(s > 0 for s in valid)

    def test_zero_std_returns_zero(self):
        returns = [0.0] * 100
        rs = _rolling_sharpe(returns, window=63)
        valid = [s for s in rs if s is not None]
        assert all(s == 0 for s in valid)


class TestHeatmapColor:
    def test_strong_positive(self):
        color = _heatmap_color(6.0)
        assert color == "#238636"

    def test_moderate_positive(self):
        color = _heatmap_color(3.0)
        assert color == "#2ea043"

    def test_slight_positive(self):
        color = _heatmap_color(0.5)
        assert color == "#1a7f37"

    def test_slight_negative(self):
        color = _heatmap_color(-1.0)
        assert color == "#da3633"

    def test_moderate_negative(self):
        color = _heatmap_color(-3.0)
        assert color == "#f85149"

    def test_strong_negative(self):
        color = _heatmap_color(-8.0)
        assert color == "#b62324"


# ---------------------------------------------------------------------------
# Integration test: generate full tearsheet
# ---------------------------------------------------------------------------

class TestGenerateTearsheet:
    def test_generates_html_file(self):
        prices = _make_simple_prices(30)
        engine = BacktestEngine(signal=AlwaysLongSignal())
        engine.run(prices)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name

        try:
            result = generate_tearsheet(engine, output_path=path)
            assert os.path.exists(result)

            with open(result) as f:
                html = f.read()
            assert "<!DOCTYPE html>" in html
            assert "Equity Curve" in html
            assert "Drawdown" in html
            assert "backtest-lab" in html
        finally:
            os.unlink(path)

    def test_contains_svg_charts(self):
        prices = _make_simple_prices(30)
        engine = BacktestEngine(signal=AlwaysLongSignal())
        engine.run(prices)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name

        try:
            generate_tearsheet(engine, output_path=path)
            with open(path) as f:
                html = f.read()
            assert "<svg" in html
            assert "polyline" in html
        finally:
            os.unlink(path)

    def test_contains_summary_stats(self):
        prices = _make_simple_prices(30)
        engine = BacktestEngine(signal=AlwaysLongSignal())
        engine.run(prices)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name

        try:
            generate_tearsheet(engine, output_path=path)
            with open(path) as f:
                html = f.read()
            assert "Total Return" in html
            assert "Sharpe" in html
            assert "Max DD" in html
        finally:
            os.unlink(path)

    def test_contains_trade_log(self):
        prices = _make_simple_prices(30)
        engine = BacktestEngine(signal=AlwaysLongSignal())
        engine.run(prices)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name

        try:
            generate_tearsheet(engine, output_path=path)
            with open(path) as f:
                html = f.read()
            assert "Trade Log" in html
            assert "AAPL" in html or "MSFT" in html
        finally:
            os.unlink(path)

    def test_custom_title(self):
        prices = _make_simple_prices(30)
        engine = BacktestEngine(signal=AlwaysLongSignal())
        engine.run(prices)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name

        try:
            generate_tearsheet(engine, output_path=path, title="My Custom Title")
            with open(path) as f:
                html = f.read()
            assert "My Custom Title" in html
        finally:
            os.unlink(path)

    def test_raises_on_empty_engine(self):
        engine = BacktestEngine(signal=AlwaysLongSignal())
        with pytest.raises(ValueError, match="no snapshots"):
            generate_tearsheet(engine)

    def test_no_trades_no_crash(self):
        """Tearsheet should work even with no trades."""

        class NoSignal(Signal):
            @property
            def lookback_days(self) -> int:
                return 5
            def generate_signals(self, prices, current_date):
                return {}

        prices = _make_simple_prices(30)
        engine = BacktestEngine(signal=NoSignal())
        engine.run(prices)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name

        try:
            generate_tearsheet(engine, output_path=path)
            with open(path) as f:
                html = f.read()
            assert "No trades executed" in html
        finally:
            os.unlink(path)
