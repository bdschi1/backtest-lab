"""Tests for the core backtest engine."""

from datetime import date, timedelta

import polars as pl

from engine.backtest import BacktestConfig, BacktestEngine, Portfolio, Position
from execution.borrow import FixedBorrow
from execution.commission import PerShareCommission
from execution.fill_model import MidPriceFill, SpreadAwareFill
from signals.base import Signal
from signals.momentum import MomentumSignal


# ---------------------------------------------------------------------------
# Test signal that always goes long on one ticker (deterministic)
# ---------------------------------------------------------------------------

class AlwaysLongSignal(Signal):
    """Deterministic signal for testing — always long AAPL."""

    @property
    def lookback_days(self) -> int:
        return 5

    def generate_signals(self, prices, current_date):
        return {"AAPL": 0.5, "MSFT": -0.3}


class NoSignal(Signal):
    """Signal that never fires — for testing zero-trade scenarios."""

    @property
    def lookback_days(self) -> int:
        return 5

    def generate_signals(self, prices, current_date):
        return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_prices(n_days: int = 60) -> pl.DataFrame:
    """Generate simple deterministic price data for AAPL and MSFT."""
    rows = []
    start = date(2024, 1, 2)
    day_count = 0

    for i in range(n_days * 2):  # extra to account for weekends
        d = start + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        day_count += 1
        if day_count > n_days:
            break

        for ticker, base_price in [("AAPL", 150.0), ("MSFT", 350.0)]:
            # Gentle uptrend
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
# Portfolio tests
# ---------------------------------------------------------------------------

class TestPortfolio:
    def test_initial_state(self):
        p = Portfolio(cash=1_000_000)
        assert p.total_equity == 1_000_000
        assert p.long_value == 0
        assert p.short_value == 0

    def test_long_position(self):
        p = Portfolio(cash=900_000)
        p.positions["AAPL"] = Position("AAPL", 100, 150.0, 160.0)
        assert p.long_value == 16_000
        assert p.total_equity == 916_000

    def test_short_position(self):
        p = Portfolio(cash=1_100_000)
        p.positions["TSLA"] = Position("TSLA", -100, 200.0, 190.0)
        assert p.short_value == 19_000
        assert p.total_equity == 1_100_000 - 19_000

    def test_unrealized_pnl(self):
        pos = Position("AAPL", 100, 150.0, 160.0)
        assert pos.unrealized_pnl == 1000.0  # 100 * (160 - 150)

    def test_short_pnl(self):
        pos = Position("TSLA", -100, 200.0, 190.0)
        assert pos.unrealized_pnl == 1000.0  # -100 * (190 - 200)

    def test_update_prices(self):
        p = Portfolio(cash=900_000)
        p.positions["AAPL"] = Position("AAPL", 100, 150.0, 150.0)
        p.update_prices({"AAPL": 160.0})
        assert p.positions["AAPL"].current_price == 160.0


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------

class TestBacktestEngine:
    def test_runs_without_crash(self):
        prices = _make_simple_prices(30)
        engine = BacktestEngine(signal=AlwaysLongSignal())
        snapshots = engine.run(prices)
        assert len(snapshots) > 0

    def test_starts_with_correct_capital(self):
        prices = _make_simple_prices(30)
        config = BacktestConfig(initial_capital=500_000)
        engine = BacktestEngine(signal=NoSignal(), config=config)
        snapshots = engine.run(prices)
        assert snapshots[0].equity == 500_000

    def test_no_signal_no_trades(self):
        prices = _make_simple_prices(30)
        engine = BacktestEngine(signal=NoSignal())
        engine.run(prices)
        assert len(engine.trades) == 0
        assert engine.portfolio.total_equity == 1_000_000

    def test_trades_executed(self):
        prices = _make_simple_prices(30)
        engine = BacktestEngine(signal=AlwaysLongSignal())
        engine.run(prices)
        assert len(engine.trades) > 0

    def test_long_and_short_positions(self):
        prices = _make_simple_prices(30)
        engine = BacktestEngine(signal=AlwaysLongSignal())
        engine.run(prices)
        # AlwaysLongSignal: AAPL +0.5 (long), MSFT -0.3 (short)
        if "AAPL" in engine.portfolio.positions:
            assert engine.portfolio.positions["AAPL"].shares > 0
        if "MSFT" in engine.portfolio.positions:
            assert engine.portfolio.positions["MSFT"].shares < 0

    def test_spread_aware_costs_more(self):
        prices = _make_simple_prices(30)
        config = BacktestConfig(initial_capital=1_000_000)

        engine_mid = BacktestEngine(
            signal=AlwaysLongSignal(),
            fill_model=MidPriceFill(),
            config=config,
        )
        engine_mid.run(prices)

        engine_spread = BacktestEngine(
            signal=AlwaysLongSignal(),
            fill_model=SpreadAwareFill(),
            config=config,
        )
        engine_spread.run(prices)

        # Spread-aware should have higher execution costs
        assert engine_spread._total_spread_cost > engine_mid._total_spread_cost

    def test_commission_deducted(self):
        prices = _make_simple_prices(30)
        engine = BacktestEngine(
            signal=AlwaysLongSignal(),
            commission_model=PerShareCommission(rate=0.01),
        )
        engine.run(prices)
        assert engine._total_commission > 0

    def test_borrow_cost_accrues_for_shorts(self):
        prices = _make_simple_prices(30)
        engine = BacktestEngine(
            signal=AlwaysLongSignal(),
            borrow_model=FixedBorrow(annualized_rate=0.05),
        )
        engine.run(prices)
        # AlwaysLongSignal shorts MSFT → borrow cost should accrue
        assert engine._total_borrow_cost > 0

    def test_summary_stats(self):
        prices = _make_simple_prices(30)
        engine = BacktestEngine(signal=AlwaysLongSignal())
        engine.run(prices)
        summary = engine.summary()

        assert "sharpe_ratio" in summary
        assert "max_drawdown_pct" in summary
        assert "total_trades" in summary
        assert summary["trading_days"] > 0

    def test_results_to_dataframe(self):
        prices = _make_simple_prices(30)
        engine = BacktestEngine(signal=AlwaysLongSignal())
        engine.run(prices)
        df = engine.results_to_dataframe()
        assert "equity" in df.columns
        assert "daily_return" in df.columns
        assert df.height > 0

    def test_trades_to_dataframe(self):
        prices = _make_simple_prices(30)
        engine = BacktestEngine(signal=AlwaysLongSignal())
        engine.run(prices)
        df = engine.trades_to_dataframe()
        assert "ticker" in df.columns
        assert "fill_price" in df.columns

    def test_too_few_days_returns_empty(self):
        prices = _make_simple_prices(3)
        engine = BacktestEngine(
            signal=MomentumSignal(formation_days=252),
        )
        snapshots = engine.run(prices)
        assert len(snapshots) == 0

    def test_weekly_rebalance(self):
        prices = _make_simple_prices(60)
        config = BacktestConfig(rebalance_frequency="weekly")
        engine = BacktestEngine(signal=AlwaysLongSignal(), config=config)
        engine.run(prices)

        # Weekly rebalance should only trade on Mondays
        trade_dates = [t.date for t in engine.trades]
        # First trade is allowed on any day (initial entry)
        if len(trade_dates) > 2:
            rebalance_dates = trade_dates[2:]  # skip initial entry pair
            for d in rebalance_dates:
                assert d.weekday() == 0, f"Non-Monday trade on {d} ({d.strftime('%A')})"
