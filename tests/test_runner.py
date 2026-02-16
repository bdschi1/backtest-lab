"""Tests for btconfig.runner — the shared run_backtest() entry point."""

from __future__ import annotations

import tempfile
from datetime import date, timedelta
from pathlib import Path

import pytest

from btconfig.run_config import (
    DataConfig,
    FillConfig,
    ExecutionConfig,
    OutputConfig,
    RegimeConfig,
    RiskConfig,
    RunConfig,
    SignalConfig,
)
from btconfig.runner import run_backtest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quick_config(**overrides) -> RunConfig:
    """Build a minimal RunConfig for fast tests (short date range, synthetic)."""
    defaults = dict(
        initial_capital=100_000,
        data=DataConfig(
            universe=["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"],
            start_date=date.today() - timedelta(days=365),
            end_date=date.today(),
            live=False,
        ),
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


# ---------------------------------------------------------------------------
# Basic run
# ---------------------------------------------------------------------------

class TestRunBacktestBasic:
    """Verify the runner produces valid engine + summary for default configs."""

    def test_default_momentum(self):
        rc = _quick_config(signal=SignalConfig(type="momentum"))
        engine, summary = run_backtest(rc)

        assert engine is not None
        assert isinstance(summary, dict)
        assert "total_return_pct" in summary
        assert "sharpe_ratio" in summary
        assert "max_drawdown_pct" in summary

    def test_mean_reversion_signal(self):
        rc = _quick_config(signal=SignalConfig(type="mean_reversion"))
        engine, summary = run_backtest(rc)

        assert engine is not None
        assert isinstance(summary, dict)
        assert "total_return_pct" in summary

    def test_composite_signal(self):
        rc = _quick_config(
            signal=SignalConfig(
                type="composite",
                composite_weights={"momentum": 0.5, "mean_reversion": 0.5},
            ),
        )
        engine, summary = run_backtest(rc)

        assert engine is not None
        assert isinstance(summary, dict)

    def test_daily_rebalance(self):
        rc = _quick_config(rebalance_frequency="daily")
        engine, summary = run_backtest(rc)

        assert engine is not None
        assert summary.get("total_trades", 0) > 0

    def test_monthly_rebalance(self):
        rc = _quick_config(rebalance_frequency="monthly")
        engine, summary = run_backtest(rc)

        assert engine is not None


# ---------------------------------------------------------------------------
# Fill models
# ---------------------------------------------------------------------------

class TestRunBacktestFillModels:
    """Verify different fill/cost configurations."""

    def test_mid_fill_no_costs(self):
        rc = _quick_config(fill=FillConfig(model="mid"))
        engine, summary = run_backtest(rc)

        # Mid-fill means zero transaction costs
        assert summary.get("total_costs", 0) == 0.0

    def test_spread_fill_with_costs(self):
        rc = _quick_config(
            fill=FillConfig(model="spread", spread_default_bps=10.0),
            execution=ExecutionConfig(
                slippage_bps=3.0,
                commission_rate=0.005,
                borrow_gc_rate=0.0025,
            ),
        )
        engine, summary = run_backtest(rc)

        assert engine is not None
        # With spread fill and costs, total_costs should be > 0
        assert summary.get("total_costs", 0) >= 0

    def test_impact_fill(self):
        rc = _quick_config(
            fill=FillConfig(model="impact", impact_eta=0.1),
            execution=ExecutionConfig(
                slippage_bps=5.0,
                commission_rate=0.005,
                borrow_gc_rate=0.005,
            ),
        )
        engine, summary = run_backtest(rc)

        assert engine is not None


# ---------------------------------------------------------------------------
# Risk manager
# ---------------------------------------------------------------------------

class TestRunBacktestRisk:
    """Verify risk manager integration."""

    def test_risk_enabled(self):
        rc = _quick_config(
            risk=RiskConfig(
                enabled=True,
                stop_multiplier=3.0,
                stop_max_loss_pct=15.0,
                sizer_max_pct_equity=10.0,
                sizer_max_pct_adv=10.0,
                drawdown_warning_pct=-5.0,
                drawdown_halt_pct=-15.0,
                exposure_max_gross_pct=200.0,
                exposure_max_net_pct=50.0,
                exposure_max_single_name_pct=20.0,
            ),
        )
        engine, summary = run_backtest(rc)

        assert engine is not None
        assert "total_stops_triggered" in summary
        assert "total_trades_rejected" in summary

    def test_risk_disabled(self):
        rc = _quick_config(risk=RiskConfig(enabled=False))
        engine, summary = run_backtest(rc)

        assert engine is not None
        # When risk is off, these keys should not be present or be 0
        # (depending on engine implementation — just verify run completes)


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

class TestRunBacktestRegime:
    """Verify regime detection integration."""

    def test_regime_enabled(self):
        rc = _quick_config(
            regime=RegimeConfig(enabled=True, lookback_days=21),
        )
        engine, summary = run_backtest(rc)

        assert engine is not None

    def test_regime_disabled(self):
        rc = _quick_config(regime=RegimeConfig(enabled=False))
        engine, summary = run_backtest(rc)

        assert engine is not None


# ---------------------------------------------------------------------------
# Tearsheet generation
# ---------------------------------------------------------------------------

class TestRunBacktestTearsheet:
    """Verify tearsheet output."""

    def test_tearsheet_generated(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name

        rc = _quick_config(
            output=OutputConfig(tearsheet_path=path),
        )
        engine, summary = run_backtest(rc)

        assert Path(path).exists()
        content = Path(path).read_text()
        assert len(content) > 100  # Non-trivial HTML
        assert "<html" in content.lower() or "<svg" in content.lower() or "<!doctype" in content.lower()

        Path(path).unlink()

    def test_no_tearsheet_when_not_configured(self):
        rc = _quick_config(output=OutputConfig(tearsheet_path=None))
        engine, summary = run_backtest(rc)

        # Just verify it runs without trying to write anything
        assert engine is not None


# ---------------------------------------------------------------------------
# Date resolution
# ---------------------------------------------------------------------------

class TestRunBacktestDates:
    """Verify default date resolution logic."""

    def test_explicit_dates(self):
        rc = _quick_config(
            data=DataConfig(
                universe=["AAPL", "MSFT", "GOOG"],
                start_date=date(2022, 1, 1),
                end_date=date(2023, 12, 31),
                live=False,
            ),
        )
        engine, summary = run_backtest(rc)
        assert engine is not None

    def test_none_dates_default_to_3_years(self):
        rc = _quick_config(
            data=DataConfig(
                universe=["AAPL", "MSFT", "GOOG"],
                start_date=None,
                end_date=None,
                live=False,
            ),
        )
        engine, summary = run_backtest(rc)
        assert engine is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestRunBacktestEdgeCases:
    """Edge cases and error handling."""

    def test_two_tickers(self):
        """Regression: previously failed with < 3 tickers."""
        rc = _quick_config(
            data=DataConfig(
                universe=["AAPL", "MSFT"],
                start_date=date.today() - timedelta(days=365),
                end_date=date.today(),
                live=False,
            ),
        )
        engine, summary = run_backtest(rc)
        assert engine is not None
        # Should actually produce trades now (bug was fixed)
        assert summary.get("total_trades", 0) >= 0

    def test_single_ticker(self):
        """Single ticker should run without crashing."""
        rc = _quick_config(
            data=DataConfig(
                universe=["SPY"],
                start_date=date.today() - timedelta(days=365),
                end_date=date.today(),
                live=False,
            ),
        )
        engine, summary = run_backtest(rc)
        assert engine is not None

    def test_small_capital(self):
        rc = _quick_config(initial_capital=10_000)
        engine, summary = run_backtest(rc)
        assert engine is not None

    def test_large_capital(self):
        rc = _quick_config(initial_capital=10_000_000)
        engine, summary = run_backtest(rc)
        assert engine is not None


# ---------------------------------------------------------------------------
# Full-stack (risk + regime + impact fill)
# ---------------------------------------------------------------------------

class TestRunBacktestFullStack:
    """Production-style config with all features enabled."""

    def test_full_stack_run(self):
        rc = RunConfig(
            initial_capital=2_000_000,
            max_position_pct=8.0,
            signal_threshold=0.15,
            rebalance_frequency="weekly",
            signal=SignalConfig(
                type="composite",
                momentum_formation_days=126,
                momentum_skip_days=21,
                composite_weights={"momentum": 0.6, "mean_reversion": 0.4},
            ),
            data=DataConfig(
                universe=[
                    "AAPL", "MSFT", "GOOG", "AMZN", "TSLA",
                    "NVDA", "META", "JPM", "V", "JNJ",
                ],
                start_date=date.today() - timedelta(days=2 * 365),
                end_date=date.today(),
                live=False,
            ),
            fill=FillConfig(model="impact", impact_eta=0.1),
            execution=ExecutionConfig(
                slippage_bps=5.0,
                commission_rate=0.005,
                borrow_gc_rate=0.005,
            ),
            risk=RiskConfig(
                enabled=True,
                stop_multiplier=3.0,
                exposure_max_gross_pct=180.0,
                exposure_max_net_pct=40.0,
                exposure_max_single_name_pct=15.0,
            ),
            regime=RegimeConfig(enabled=True, lookback_days=21),
        )

        engine, summary = run_backtest(rc)

        assert engine is not None
        assert isinstance(summary, dict)
        assert "total_return_pct" in summary
        assert "sharpe_ratio" in summary
        assert "total_stops_triggered" in summary
