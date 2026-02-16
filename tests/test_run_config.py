"""Tests for btconfig.run_config — the shared Pydantic config model."""

from __future__ import annotations

import argparse
import tempfile
from datetime import date
from pathlib import Path

import pytest

from btconfig.run_config import (
    DataConfig,
    ExecutionConfig,
    FillConfig,
    OutputConfig,
    RegimeConfig,
    RiskConfig,
    RunConfig,
    SignalConfig,
)


class TestRunConfigDefaults:
    """Test that default construction produces valid config."""

    def test_default_construction(self):
        rc = RunConfig()
        assert rc.initial_capital == 1_000_000.0
        assert rc.max_position_pct == 10.0
        assert rc.signal_threshold == 0.1
        assert rc.rebalance_frequency == "weekly"

    def test_default_signal(self):
        rc = RunConfig()
        assert rc.signal.type == "momentum"
        assert rc.signal.momentum_formation_days == 126

    def test_default_data(self):
        rc = RunConfig()
        assert len(rc.data.universe) == 10
        assert "AAPL" in rc.data.universe
        assert rc.data.live is True
        assert rc.data.provider == "yahoo"

    def test_default_fill(self):
        rc = RunConfig()
        assert rc.fill.model == "mid"

    def test_default_risk(self):
        rc = RunConfig()
        assert rc.risk.enabled is False
        assert rc.risk.stop_multiplier == 3.0

    def test_default_regime(self):
        rc = RunConfig()
        assert rc.regime.enabled is False


class TestRunConfigValidation:
    """Test Pydantic validators enforce ranges."""

    def test_negative_capital_rejected(self):
        with pytest.raises(Exception):
            RunConfig(initial_capital=-1000)

    def test_max_position_pct_too_high(self):
        with pytest.raises(Exception):
            RunConfig(max_position_pct=200.0)

    def test_invalid_signal_type(self):
        with pytest.raises(Exception):
            RunConfig(signal=SignalConfig(type="invalid"))

    def test_invalid_rebalance(self):
        with pytest.raises(Exception):
            RunConfig(rebalance_frequency="hourly")


class TestRunConfigFromCliArgs:
    """Test building RunConfig from argparse namespace."""

    def _make_args(self, **overrides):
        defaults = dict(
            signal="momentum",
            universe="AAPL,MSFT,GOOG",
            start=None,
            end=None,
            capital=100_000,
            live=False,
            provider="yahoo",
            fill="mid",
            risk=False,
            regime=False,
            tearsheet=None,
            rebalance="weekly",
            max_position_pct=10.0,
            verbose=False,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_basic_from_args(self):
        args = self._make_args()
        rc = RunConfig.from_cli_args(args)
        assert rc.initial_capital == 100_000
        assert rc.data.universe == ["AAPL", "MSFT", "GOOG"]
        assert rc.signal.type == "momentum"

    def test_risk_enabled(self):
        args = self._make_args(risk=True)
        rc = RunConfig.from_cli_args(args)
        assert rc.risk.enabled is True

    def test_regime_enabled(self):
        args = self._make_args(regime=True)
        rc = RunConfig.from_cli_args(args)
        assert rc.regime.enabled is True

    def test_dates_parsed(self):
        args = self._make_args(start="2022-01-01", end="2023-12-31")
        rc = RunConfig.from_cli_args(args)
        assert rc.data.start_date == date(2022, 1, 1)
        assert rc.data.end_date == date(2023, 12, 31)

    def test_fill_model_passed(self):
        args = self._make_args(fill="impact")
        rc = RunConfig.from_cli_args(args)
        assert rc.fill.model == "impact"

    def test_tearsheet_path(self):
        args = self._make_args(tearsheet="results.html")
        rc = RunConfig.from_cli_args(args)
        assert rc.output.tearsheet_path == "results.html"


class TestRunConfigYaml:
    """Test YAML round-trip."""

    def test_yaml_round_trip(self):
        rc = RunConfig(
            initial_capital=500_000,
            signal=SignalConfig(type="mean_reversion"),
            data=DataConfig(universe=["SPY", "QQQ"], live=True),
            risk=RiskConfig(enabled=True, stop_multiplier=2.5),
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            path = f.name

        rc.to_yaml(path)
        loaded = RunConfig.from_yaml(path)

        assert loaded.initial_capital == 500_000
        assert loaded.signal.type == "mean_reversion"
        assert loaded.data.universe == ["SPY", "QQQ"]
        assert loaded.data.live is True
        assert loaded.risk.enabled is True
        assert loaded.risk.stop_multiplier == 2.5

        Path(path).unlink()

    def test_partial_yaml(self):
        """Only override some fields; rest should take defaults."""
        import yaml

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            yaml.dump({"initial_capital": 250_000}, f)
            path = f.name

        rc = RunConfig.from_yaml(path)
        assert rc.initial_capital == 250_000
        assert rc.signal.type == "momentum"  # default
        assert rc.risk.enabled is False  # default

        Path(path).unlink()


class TestDataConfigParsing:
    """Test DataConfig universe parsing."""

    def test_string_universe_parsed(self):
        dc = DataConfig(universe="AAPL, MSFT, goog")
        assert dc.universe == ["AAPL", "MSFT", "GOOG"]

    def test_list_universe_uppercased(self):
        dc = DataConfig(universe=["aapl", "msft"])
        assert dc.universe == ["AAPL", "MSFT"]


class TestExampleConfigs:
    """Test that all example YAML configs load and validate."""

    @pytest.fixture(params=[
        "momentum_basic.yaml",
        "ls_with_risk.yaml",
        "mean_reversion.yaml",
        "composite_pod_style.yaml",
    ])
    def config_path(self, request):
        p = Path(__file__).parent.parent / "examples" / "configs" / request.param
        if not p.exists():
            pytest.skip(f"Example config not found: {p}")
        return str(p)

    def test_example_loads(self, config_path):
        rc = RunConfig.from_yaml(config_path)
        assert rc.initial_capital > 0
        assert len(rc.data.universe) > 0
