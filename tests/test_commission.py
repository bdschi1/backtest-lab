"""Tests for commission models."""

import pytest

from execution.commission import (
    PerShareCommission,
    TieredCommission,
    ZeroCommission,
)


class TestZeroCommission:
    def test_always_zero(self):
        model = ZeroCommission()
        result = model.calculate(1000, 150.0)
        assert result.total == 0.0
        assert result.per_share == 0.0
        assert result.rate_type == "zero"


class TestPerShareCommission:
    def test_default_ib_rate(self):
        model = PerShareCommission()  # $0.005/share, $1 min
        result = model.calculate(1000, 150.0)
        assert result.total == 5.0  # 1000 * 0.005
        assert abs(result.per_share - 0.005) < 1e-6

    def test_minimum_applies(self):
        model = PerShareCommission(rate=0.005, minimum=1.0)
        result = model.calculate(10, 150.0)  # 10 * 0.005 = $0.05 < $1.00
        assert result.total == 1.0

    def test_maximum_pct_caps(self):
        model = PerShareCommission(rate=0.005, minimum=1.0, maximum_pct=1.0)
        # 100 shares at $1.00 = $100 trade value, 1% cap = $1.00
        result = model.calculate(100, 1.0)
        assert result.total <= 1.0  # 1% of $100

    def test_rate_type(self):
        model = PerShareCommission()
        result = model.calculate(100, 100.0)
        assert result.rate_type == "per_share"


class TestTieredCommission:
    def test_tier_0_retail(self):
        model = TieredCommission(tier=0)  # $0.0035/share
        result = model.calculate(10_000, 50.0)
        assert result.total == 35.0  # 10K * 0.0035

    def test_tier_3_institutional(self):
        model = TieredCommission(tier=3)  # $0.001/share
        result = model.calculate(10_000, 50.0)
        assert result.total == 10.0  # 10K * 0.001

    def test_institutional_cheaper_than_retail(self):
        retail = TieredCommission(tier=0)
        institutional = TieredCommission(tier=3)

        r = retail.calculate(10_000, 50.0)
        i = institutional.calculate(10_000, 50.0)
        assert i.total < r.total

    def test_minimum_applies(self):
        model = TieredCommission(tier=0, minimum=0.35)
        result = model.calculate(1, 50.0)
        assert result.total == 0.35

    def test_rate_type(self):
        model = TieredCommission(tier=1)
        result = model.calculate(100, 100.0)
        assert result.rate_type == "tiered"
