"""Tests for borrow cost models."""

import pytest

from execution.borrow import FixedBorrow, TieredBorrow, ZeroBorrow


class TestZeroBorrow:
    def test_always_zero(self):
        model = ZeroBorrow()
        result = model.daily_cost(1000, 100.0)
        assert result.daily_cost == 0.0
        assert result.annualized_rate == 0.0
        assert result.is_hard_to_borrow is False


class TestFixedBorrow:
    def test_default_gc_rate(self):
        model = FixedBorrow(annualized_rate=0.005)  # 50 bps
        result = model.daily_cost(1000, 100.0)
        # Notional = $100K, daily = $100K * 0.005 / 252 ≈ $1.98
        expected = 100_000 * 0.005 / 252
        assert abs(result.daily_cost - expected) < 0.01
        assert result.annualized_rate == 0.005

    def test_higher_rate_more_cost(self):
        gc = FixedBorrow(annualized_rate=0.005)
        htb = FixedBorrow(annualized_rate=0.15)

        r_gc = gc.daily_cost(1000, 100.0)
        r_htb = htb.daily_cost(1000, 100.0)
        assert r_htb.daily_cost > r_gc.daily_cost

    def test_more_shares_more_cost(self):
        model = FixedBorrow(annualized_rate=0.005)
        r1 = model.daily_cost(100, 100.0)
        r2 = model.daily_cost(10_000, 100.0)
        assert r2.daily_cost > r1.daily_cost


class TestTieredBorrow:
    def test_gc_rate_low_short_interest(self):
        model = TieredBorrow(gc_rate=0.0025)
        result = model.daily_cost(1000, 100.0, short_pct_of_float=2.0)
        assert result.annualized_rate == 0.0025
        assert result.is_hard_to_borrow is False

    def test_warm_rate_medium_si(self):
        model = TieredBorrow(warm_rate=0.015)
        result = model.daily_cost(1000, 100.0, short_pct_of_float=7.0)
        assert result.annualized_rate == 0.015
        assert result.is_hard_to_borrow is False

    def test_special_rate_high_si(self):
        model = TieredBorrow(special_rate=0.05)
        result = model.daily_cost(1000, 100.0, short_pct_of_float=15.0)
        assert result.annualized_rate == 0.05
        assert result.is_hard_to_borrow is True

    def test_htb_rate_very_high_si(self):
        model = TieredBorrow(htb_rate=0.15)
        result = model.daily_cost(1000, 100.0, short_pct_of_float=25.0)
        assert result.annualized_rate == 0.15
        assert result.is_hard_to_borrow is True

    def test_htb_costs_significantly_more(self):
        model = TieredBorrow()
        gc = model.daily_cost(1000, 100.0, short_pct_of_float=1.0)
        htb = model.daily_cost(1000, 100.0, short_pct_of_float=30.0)
        assert htb.daily_cost > 10 * gc.daily_cost  # HTB should be 60x GC
