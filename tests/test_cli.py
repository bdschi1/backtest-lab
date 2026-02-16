"""Tests for the CLI module."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest


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


class TestSyntheticDataGenerator:
    """Test the CLI's synthetic data generation."""

    def test_generates_data(self):
        from cli import _generate_synthetic

        start = date(2024, 1, 2)
        end = date(2024, 6, 30)
        tickers = ["AAPL", "MSFT"]

        df = _generate_synthetic(tickers, start, end)
        assert isinstance(df, pl.DataFrame)
        assert df.height > 0
        assert "date" in df.columns
        assert "ticker" in df.columns
        assert "close" in df.columns

    def test_correct_tickers(self):
        from cli import _generate_synthetic

        tickers = ["AAPL", "GOOG", "TSLA"]
        df = _generate_synthetic(tickers, date(2024, 1, 2), date(2024, 3, 30))
        found_tickers = df.get_column("ticker").unique().to_list()
        assert set(found_tickers) == set(tickers)

    def test_no_weekend_dates(self):
        from cli import _generate_synthetic

        df = _generate_synthetic(["AAPL"], date(2024, 1, 2), date(2024, 6, 30))
        dates = df.get_column("date").to_list()
        for d in dates:
            assert d.weekday() < 5, f"Weekend date found: {d}"

    def test_positive_prices(self):
        from cli import _generate_synthetic

        df = _generate_synthetic(["AAPL", "MSFT"], date(2024, 1, 2), date(2024, 6, 30))
        closes = df.get_column("close").to_list()
        assert all(c > 0 for c in closes)

    def test_has_volume(self):
        from cli import _generate_synthetic

        df = _generate_synthetic(["AAPL"], date(2024, 1, 2), date(2024, 3, 30))
        volumes = df.get_column("volume").to_list()
        assert all(v > 0 for v in volumes)


class TestCLISignalInstantiation:
    """Test that CLI correctly creates signal instances."""

    def test_momentum_signal_creation(self):
        from signals.momentum import MomentumSignal

        signal = MomentumSignal(formation_days=126, skip_days=21)
        assert "Momentum" in signal.name
        # lookback includes formation + skip + buffer
        assert signal.lookback_days >= 126 + 21

    def test_mean_reversion_signal_creation(self):
        from signals.momentum import MeanReversionSignal

        signal = MeanReversionSignal(lookback_days=21, zscore_clip=1.5)
        assert signal.lookback_days >= 21

    def test_composite_signal_creation(self):
        from signals.momentum import CompositeSignal, MomentumSignal, MeanReversionSignal

        mom = MomentumSignal(formation_days=126, skip_days=21)
        mr = MeanReversionSignal(lookback_days=21, zscore_clip=1.5)
        comp = CompositeSignal([(mom, 0.7), (mr, 0.3)])
        assert comp.lookback_days == max(mom.lookback_days, mr.lookback_days)


class TestCLIFillModelSelection:
    """Test fill model selection logic from CLI."""

    def test_mid_fill(self):
        from execution.fill_model import MidPriceFill
        fill = MidPriceFill()
        assert "Mid" in fill.name

    def test_spread_fill(self):
        from execution.fill_model import SpreadAwareFill
        fill = SpreadAwareFill()
        assert "Spread" in fill.name

    def test_impact_fill(self):
        from execution.fill_model import MarketImpactFill
        fill = MarketImpactFill()
        assert "Impact" in fill.name or "Almgren" in fill.name


class TestCLIRiskManagerWiring:
    """Test that risk manager components can be assembled as CLI does."""

    def test_full_risk_manager_assembly(self):
        from risk.stop_loss import ATRTrailingStop
        from risk.position_sizer import PositionSizer
        from risk.drawdown_control import DrawdownController
        from risk.exposure_limits import ExposureLimits
        from risk.risk_manager import RiskManager

        stop = ATRTrailingStop(multiplier=3.0, max_loss_pct=15.0)
        sizer = PositionSizer(max_pct_equity=5.0, max_pct_adv=50.0)
        dd = DrawdownController(warning_dd_pct=-5.0, halt_dd_pct=-15.0)
        exposure = ExposureLimits(max_gross_pct=200.0, max_single_name_pct=10.0)

        rm = RiskManager(
            stop_loss=stop,
            sizer=sizer,
            drawdown=dd,
            exposure=exposure,
        )
        assert rm is not None

    def test_risk_manager_with_engine(self):
        from risk.stop_loss import ATRTrailingStop
        from risk.position_sizer import PositionSizer
        from risk.drawdown_control import DrawdownController
        from risk.exposure_limits import ExposureLimits
        from risk.risk_manager import RiskManager
        from engine.backtest import BacktestEngine, BacktestConfig
        from signals.base import Signal

        class SimpleSignal(Signal):
            @property
            def lookback_days(self) -> int:
                return 5
            def generate_signals(self, prices, current_date):
                return {"AAPL": 0.5}

        rm = RiskManager(
            stop_loss=ATRTrailingStop(multiplier=3.0, max_loss_pct=15.0),
            sizer=PositionSizer(max_pct_equity=5.0, max_pct_adv=50.0),
            drawdown=DrawdownController(warning_dd_pct=-5.0, halt_dd_pct=-15.0),
            exposure=ExposureLimits(max_gross_pct=200.0, max_single_name_pct=10.0),
        )

        prices = _make_simple_prices(30)
        engine = BacktestEngine(
            signal=SimpleSignal(),
            config=BacktestConfig(initial_capital=1_000_000),
            risk_manager=rm,
        )
        snapshots = engine.run(prices)
        assert len(snapshots) > 0
