"""Tests for risk management module."""


from risk.stop_loss import ATRTrailingStop, NoStopLoss
from risk.position_sizer import PositionSizer
from risk.drawdown_control import CircuitState, DrawdownController
from risk.exposure_limits import ExposureLimits
from risk.risk_manager import RiskManager


# ---------------------------------------------------------------------------
# Stop-loss tests
# ---------------------------------------------------------------------------

class TestATRTrailingStop:
    def test_long_not_triggered(self):
        model = ATRTrailingStop(multiplier=3.0)
        result = model.check_stop(
            "AAPL", 100, entry_price=150.0, current_price=155.0,
            high_water_mark=156.0, atr=2.0,
        )
        assert not result.triggered

    def test_long_trailing_triggered(self):
        model = ATRTrailingStop(multiplier=3.0)
        # HWM=160, ATR=2, stop=160-6=154, current=153 → triggered
        result = model.check_stop(
            "AAPL", 100, entry_price=150.0, current_price=153.0,
            high_water_mark=160.0, atr=2.0,
        )
        assert result.triggered
        assert "trailing stop" in result.reason

    def test_long_hard_stop_triggered(self):
        model = ATRTrailingStop(multiplier=3.0, max_loss_pct=10.0)
        # Entry=150, 10% loss = $135, current=$130 → hard stop
        result = model.check_stop(
            "AAPL", 100, entry_price=150.0, current_price=130.0,
            high_water_mark=155.0, atr=2.0,
        )
        assert result.triggered
        assert "hard stop" in result.reason

    def test_short_not_triggered(self):
        model = ATRTrailingStop(multiplier=3.0)
        result = model.check_stop(
            "TSLA", -100, entry_price=200.0, current_price=190.0,
            high_water_mark=185.0, atr=3.0,
        )
        assert not result.triggered

    def test_short_triggered(self):
        model = ATRTrailingStop(multiplier=3.0)
        # Short entry=200, LWM=185, stop=185+9=194, current=195 → triggered
        result = model.check_stop(
            "TSLA", -100, entry_price=200.0, current_price=195.0,
            high_water_mark=185.0, atr=3.0,
        )
        assert result.triggered

    def test_no_stop_loss(self):
        model = NoStopLoss()
        result = model.check_stop("X", 100, 100.0, 50.0, 110.0, 5.0)
        assert not result.triggered


# ---------------------------------------------------------------------------
# Position sizer tests
# ---------------------------------------------------------------------------

class TestPositionSizer:
    def test_basic_sizing(self):
        sizer = PositionSizer(max_pct_equity=5.0, max_pct_adv=10.0)
        result = sizer.size_position(
            "AAPL", signal_score=1.0, price=150.0,
            equity=1_000_000, avg_daily_volume=50_000_000,
        )
        # Max from equity: 5% of $1M / $150 = 333 shares
        assert result.allowed_shares <= 333
        assert result.allowed_shares > 0

    def test_adv_constraint_binds(self):
        sizer = PositionSizer(max_pct_equity=50.0, max_pct_adv=10.0)
        result = sizer.size_position(
            "SMOL", signal_score=1.0, price=10.0,
            equity=10_000_000, avg_daily_volume=10_000,
        )
        # ADV constraint: 10% of 10K = 1000 shares
        assert result.allowed_shares <= 1000
        assert "adv_limit" in result.constraint_hit

    def test_notional_cap(self):
        sizer = PositionSizer(
            max_pct_equity=100.0, max_pct_adv=100.0,
            max_notional=100_000,
        )
        result = sizer.size_position(
            "BRK.A", signal_score=1.0, price=600_000.0,
            equity=100_000_000, avg_daily_volume=1000,
        )
        # $100K / $600K per share = 0 shares
        assert result.allowed_shares == 0

    def test_zero_price(self):
        sizer = PositionSizer()
        result = sizer.size_position("X", 1.0, 0.0, 1_000_000, 1_000_000)
        assert result.allowed_shares == 0


# ---------------------------------------------------------------------------
# Drawdown controller tests
# ---------------------------------------------------------------------------

class TestDrawdownController:
    def test_normal_state(self):
        ctrl = DrawdownController(warning_dd_pct=-5.0, halt_dd_pct=-10.0)
        result = ctrl.update(1_000_000, 0)
        assert result.state == CircuitState.NORMAL
        assert result.can_trade

    def test_warning_state(self):
        ctrl = DrawdownController(warning_dd_pct=-5.0, halt_dd_pct=-10.0)
        ctrl.update(1_000_000, 0)  # set peak
        result = ctrl.update(940_000, 1)  # -6% DD
        assert result.state == CircuitState.WARNING
        assert result.can_trade

    def test_halted_state(self):
        ctrl = DrawdownController(warning_dd_pct=-5.0, halt_dd_pct=-10.0)
        ctrl.update(1_000_000, 0)
        result = ctrl.update(890_000, 1)  # -11% DD
        assert result.state == CircuitState.HALTED
        assert not result.can_trade

    def test_recovery_from_halt(self):
        ctrl = DrawdownController(
            warning_dd_pct=-5.0, halt_dd_pct=-10.0, resume_dd_pct=-7.0,
        )
        ctrl.update(1_000_000, 0)
        ctrl.update(890_000, 1)  # halt at -11%
        assert ctrl.state == CircuitState.HALTED

        # Recover to -6% (above -7% resume)
        result = ctrl.update(940_000, 2)
        assert result.state == CircuitState.WARNING  # resumed to warning

    def test_duration_halt(self):
        ctrl = DrawdownController(
            warning_dd_pct=-5.0, halt_dd_pct=-10.0, max_dd_duration=3,
        )
        ctrl.update(1_000_000, 0)
        ctrl.update(999_000, 1)  # small DD, day 1
        ctrl.update(998_000, 2)  # day 2
        result = ctrl.update(997_000, 3)  # day 3 → halt
        assert result.state == CircuitState.HALTED

    def test_position_scale(self):
        ctrl = DrawdownController(
            warning_dd_pct=-5.0, halt_dd_pct=-10.0,
            position_scale_in_warning=0.5,
        )
        assert ctrl.position_scale == 1.0
        ctrl.update(1_000_000, 0)
        ctrl.update(940_000, 1)  # warning
        assert ctrl.position_scale == 0.5
        ctrl.update(890_000, 2)  # halted
        assert ctrl.position_scale == 0.0


# ---------------------------------------------------------------------------
# Exposure limits tests
# ---------------------------------------------------------------------------

class TestExposureLimits:
    def test_within_limits(self):
        limits = ExposureLimits(
            max_gross_pct=200.0, max_net_pct=50.0,
            max_single_name_pct=20.0,  # raise to accommodate test positions
        )
        positions = {
            "AAPL": (100, 150.0),   # $15K long = 15% of $100K
            "MSFT": (-50, 350.0),   # $17.5K short = 17.5% of $100K
        }
        result = limits.check(positions, equity=100_000)
        assert result.within_limits
        assert len(result.violations) == 0

    def test_gross_violation(self):
        limits = ExposureLimits(max_gross_pct=100.0)
        positions = {
            "AAPL": (1000, 150.0),   # $150K long
            "MSFT": (-500, 350.0),   # $175K short
        }
        result = limits.check(positions, equity=100_000)
        assert not result.within_limits
        assert any("gross" in v for v in result.violations)

    def test_single_name_violation(self):
        limits = ExposureLimits(max_single_name_pct=5.0)
        positions = {
            "AAPL": (1000, 150.0),  # $150K = 15% of $1M
        }
        result = limits.check(positions, equity=1_000_000)
        assert not result.within_limits
        assert any("single_name" in v for v in result.violations)

    def test_scale_to_fit(self):
        limits = ExposureLimits(max_gross_pct=100.0)
        targets = {"AAPL": 1000, "MSFT": -500}
        prices = {"AAPL": 150.0, "MSFT": 350.0}
        scaled = limits.scale_to_fit(targets, prices, equity=100_000)
        # Total notional = $150K + $175K = $325K, limit = $100K
        # Scale = 100/325 ≈ 0.31
        assert abs(scaled["AAPL"]) < 1000
        assert abs(scaled["MSFT"]) < 500


# ---------------------------------------------------------------------------
# Risk manager integration tests
# ---------------------------------------------------------------------------

class TestRiskManager:
    def test_approves_normal_trade(self):
        rm = RiskManager()
        rm.drawdown.update(1_000_000, 0)
        decision = rm.approve_trade(
            "AAPL", 100, signal_score=0.5, price=150.0,
            equity=1_000_000, avg_daily_volume=50_000_000,
        )
        assert decision.approved

    def test_rejects_during_halt(self):
        rm = RiskManager()
        rm.drawdown.update(1_000_000, 0)
        rm.drawdown.update(890_000, 1)  # halt
        decision = rm.approve_trade(
            "AAPL", 100, signal_score=0.5, price=150.0,
            equity=890_000, avg_daily_volume=50_000_000,
        )
        assert not decision.approved
        assert decision.approved_shares == 0
        assert "circuit_breaker_halted" in decision.reasons

    def test_scales_during_warning(self):
        rm = RiskManager(
            drawdown=DrawdownController(
                warning_dd_pct=-5.0, halt_dd_pct=-10.0,
                position_scale_in_warning=0.5,
            ),
        )
        rm.drawdown.update(1_000_000, 0)
        rm.drawdown.update(940_000, 1)  # warning
        decision = rm.approve_trade(
            "AAPL", 200, signal_score=0.5, price=150.0,
            equity=940_000, avg_daily_volume=50_000_000,
        )
        assert decision.approved_shares < 200
