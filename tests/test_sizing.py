"""Tests for sizing modes — fixed_dollar, fixed_shares, equal_weight."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from btconfig.run_config import (
    DataConfig,
    RunConfig,
    SizingConfig,
    VolTargetConfig,
)
from btconfig.runner import run_backtest


def _quick_config(**overrides) -> RunConfig:
    """Build a minimal RunConfig for fast tests."""
    defaults = dict(
        initial_capital=100_000,
        data=DataConfig(
            universe=["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"],
            start_date=date.today() - timedelta(days=365),
            end_date=date.today(),
            live=False,
        ),
        output={"markdown_tearsheet": False},
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


class TestSizingConfigModel:
    """SizingConfig Pydantic model tests."""

    def test_default_is_signal(self):
        sc = SizingConfig()
        assert sc.mode == "signal"
        assert sc.fixed_dollar_per_position is None
        assert sc.fixed_shares_per_position is None

    def test_fixed_dollar_mode(self):
        sc = SizingConfig(mode="fixed_dollar", fixed_dollar_per_position=50_000)
        assert sc.mode == "fixed_dollar"
        assert sc.fixed_dollar_per_position == 50_000

    def test_fixed_shares_mode(self):
        sc = SizingConfig(mode="fixed_shares", fixed_shares_per_position=100)
        assert sc.mode == "fixed_shares"
        assert sc.fixed_shares_per_position == 100

    def test_equal_weight_mode(self):
        sc = SizingConfig(mode="equal_weight")
        assert sc.mode == "equal_weight"

    def test_invalid_mode_rejected(self):
        with pytest.raises(Exception):
            SizingConfig(mode="invalid")


class TestFixedDollarSizing:
    """Integration: fixed dollar sizing runs backtest."""

    def test_fixed_dollar_runs(self):
        rc = _quick_config(
            sizing=SizingConfig(mode="fixed_dollar", fixed_dollar_per_position=10_000),
        )
        engine, summary = run_backtest(rc)
        assert engine is not None
        assert isinstance(summary, dict)

    def test_fixed_dollar_produces_trades(self):
        rc = _quick_config(
            sizing=SizingConfig(mode="fixed_dollar", fixed_dollar_per_position=10_000),
        )
        engine, summary = run_backtest(rc)
        assert summary.get("total_trades", 0) > 0


class TestFixedSharesSizing:
    """Integration: fixed shares sizing runs backtest."""

    def test_fixed_shares_runs(self):
        rc = _quick_config(
            sizing=SizingConfig(mode="fixed_shares", fixed_shares_per_position=50),
        )
        engine, summary = run_backtest(rc)
        assert engine is not None

    def test_fixed_shares_produces_trades(self):
        rc = _quick_config(
            sizing=SizingConfig(mode="fixed_shares", fixed_shares_per_position=50),
        )
        engine, summary = run_backtest(rc)
        assert summary.get("total_trades", 0) > 0


class TestEqualWeightSizing:
    """Integration: equal weight sizing runs backtest."""

    def test_equal_weight_runs(self):
        rc = _quick_config(
            sizing=SizingConfig(mode="equal_weight"),
        )
        engine, summary = run_backtest(rc)
        assert engine is not None

    def test_equal_weight_produces_trades(self):
        rc = _quick_config(
            sizing=SizingConfig(mode="equal_weight"),
        )
        engine, summary = run_backtest(rc)
        assert summary.get("total_trades", 0) > 0


class TestVolTargetConfigModel:
    """VolTargetConfig Pydantic model tests."""

    def test_default_disabled(self):
        vt = VolTargetConfig()
        assert vt.enabled is False
        assert vt.target_annual_vol_pct == 10.0
        assert vt.lookback_days == 63

    def test_enabled_with_params(self):
        vt = VolTargetConfig(
            enabled=True, target_annual_vol_pct=15.0,
            lookback_days=21, max_leverage=5.0,
        )
        assert vt.enabled is True
        assert vt.target_annual_vol_pct == 15.0

    def test_vol_target_integration(self):
        """Vol target enabled runs backtest."""
        rc = _quick_config(
            vol_target=VolTargetConfig(enabled=True, target_annual_vol_pct=10.0),
        )
        engine, summary = run_backtest(rc)
        assert engine is not None

    def test_vol_pct_validation(self):
        with pytest.raises(Exception):
            VolTargetConfig(target_annual_vol_pct=0.5)  # below ge=1.0

        with pytest.raises(Exception):
            VolTargetConfig(target_annual_vol_pct=60.0)  # above le=50.0
