"""Tests for risk/vol_target.py — VolTargetScaler."""

from __future__ import annotations

import math

import numpy as np

from risk.vol_target import VolTargetScaler


class TestVolTargetScalerBasic:
    """Core scaling logic."""

    def test_default_construction(self):
        scaler = VolTargetScaler()
        assert scaler.target_vol_pct == 10.0
        assert scaler.lookback_days == 63

    def test_custom_params(self):
        scaler = VolTargetScaler(
            target_vol_pct=15.0, lookback_days=21,
            max_leverage=5.0, min_leverage=0.2,
        )
        assert scaler.target_vol_pct == 15.0
        assert scaler.lookback_days == 21

    def test_insufficient_data_returns_one(self):
        scaler = VolTargetScaler(lookback_days=63)
        # Only 30 days of data — not enough
        returns = [0.001] * 30
        assert scaler.compute_scale(returns) == 1.0

    def test_exact_lookback_data(self):
        scaler = VolTargetScaler(lookback_days=63)
        # Exactly 63 days should work
        np.random.seed(42)
        returns = list(np.random.normal(0.0005, 0.01, 63))
        scale = scaler.compute_scale(returns)
        assert scale > 0
        assert isinstance(scale, float)

    def test_known_vol_scaling(self):
        """If realized vol is 20% and target is 10%, scale should be ~0.5."""
        scaler = VolTargetScaler(target_vol_pct=10.0, lookback_days=63)
        # 20% annualized vol ≈ 1.26% daily vol
        daily_vol = 0.20 / math.sqrt(252)
        np.random.seed(42)
        returns = list(np.random.normal(0, daily_vol, 63))
        scale = scaler.compute_scale(returns)
        # Should be approximately 0.5 (10% / 20%)
        assert 0.3 < scale < 0.8

    def test_low_vol_scales_up(self):
        """If realized vol is 5% and target is 10%, scale should be ~2.0."""
        scaler = VolTargetScaler(target_vol_pct=10.0, lookback_days=63)
        daily_vol = 0.05 / math.sqrt(252)
        np.random.seed(42)
        returns = list(np.random.normal(0, daily_vol, 63))
        scale = scaler.compute_scale(returns)
        assert scale > 1.5


class TestVolTargetScalerClamping:
    """Max/min leverage bounds."""

    def test_max_leverage_clamped(self):
        """If realized vol is very low, scale should cap at max_leverage."""
        scaler = VolTargetScaler(
            target_vol_pct=20.0, lookback_days=20, max_leverage=3.0,
        )
        # Very low vol → huge raw scale → should clamp to 3.0
        returns = [0.0001] * 20  # near-zero vol
        # Add tiny noise to avoid exact zero
        returns = [r + np.random.normal(0, 0.0001) for r in returns]
        np.random.seed(123)
        scale = scaler.compute_scale(returns)
        assert scale <= 3.0

    def test_min_leverage_clamped(self):
        """If realized vol is very high, scale should floor at min_leverage."""
        scaler = VolTargetScaler(
            target_vol_pct=5.0, lookback_days=20,
            max_leverage=3.0, min_leverage=0.1,
        )
        # Very high vol
        daily_vol = 1.0 / math.sqrt(252)  # 100% annualized
        np.random.seed(42)
        returns = list(np.random.normal(0, daily_vol, 20))
        scale = scaler.compute_scale(returns)
        assert scale >= 0.1


class TestVolTargetScalerEdgeCases:
    """Edge cases."""

    def test_zero_vol_returns_one(self):
        """Constant returns (zero vol) should return 1.0."""
        scaler = VolTargetScaler(lookback_days=20)
        returns = [0.001] * 20  # constant → zero std
        scale = scaler.compute_scale(returns)
        assert scale == 1.0

    def test_empty_returns(self):
        scaler = VolTargetScaler(lookback_days=20)
        assert scaler.compute_scale([]) == 1.0

    def test_negative_returns_still_work(self):
        """Vol is vol regardless of return direction."""
        scaler = VolTargetScaler(target_vol_pct=10.0, lookback_days=20)
        np.random.seed(42)
        returns = list(np.random.normal(-0.005, 0.01, 20))
        scale = scaler.compute_scale(returns)
        assert scale > 0
