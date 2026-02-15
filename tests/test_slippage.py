"""Tests for slippage models."""

import pytest

from execution.slippage import FixedSlippage, VolumeSlippage, ZeroSlippage


class TestZeroSlippage:
    def test_always_zero(self):
        model = ZeroSlippage()
        result = model.calculate(100.0, 1000, 1_000_000)
        assert result.cost_per_share == 0.0
        assert result.bps == 0.0


class TestFixedSlippage:
    def test_default_5_bps(self):
        model = FixedSlippage(bps=5.0)
        result = model.calculate(100.0, 1000, 1_000_000)
        # 5 bps of $100 = $0.05
        assert abs(result.cost_per_share - 0.05) < 1e-6
        assert result.bps == 5.0

    def test_higher_price_more_cost(self):
        model = FixedSlippage(bps=5.0)
        r1 = model.calculate(50.0, 1000, 1_000_000)
        r2 = model.calculate(200.0, 1000, 1_000_000)
        assert r2.cost_per_share > r1.cost_per_share

    def test_independent_of_volume(self):
        model = FixedSlippage(bps=5.0)
        r1 = model.calculate(100.0, 100, 1_000_000)
        r2 = model.calculate(100.0, 100_000, 1_000_000)
        assert r1.cost_per_share == r2.cost_per_share


class TestVolumeSlippage:
    def test_base_bps_at_zero_volume(self):
        model = VolumeSlippage(base_bps=2.0, scale=0.1)
        result = model.calculate(100.0, 0, 1_000_000)
        # 0 shares → only base_bps
        assert abs(result.bps - 2.0) < 0.01

    def test_more_shares_more_slippage(self):
        model = VolumeSlippage(base_bps=2.0, scale=0.1)
        r_small = model.calculate(100.0, 1_000, 1_000_000)
        r_large = model.calculate(100.0, 100_000, 1_000_000)
        assert r_large.bps > r_small.bps
        assert r_large.cost_per_share > r_small.cost_per_share

    def test_less_adv_more_slippage(self):
        model = VolumeSlippage(base_bps=2.0, scale=0.1)
        r_liquid = model.calculate(100.0, 10_000, 10_000_000)
        r_illiquid = model.calculate(100.0, 10_000, 100_000)
        assert r_illiquid.bps > r_liquid.bps

    def test_zero_adv_no_crash(self):
        model = VolumeSlippage(base_bps=2.0, scale=0.1)
        result = model.calculate(100.0, 1000, 0)
        assert result.bps == 2.0  # just base
