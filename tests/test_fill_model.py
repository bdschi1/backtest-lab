"""Tests for execution fill models."""

import pytest

from execution.fill_model import (
    BarData,
    MarketImpactFill,
    MidPriceFill,
    SpreadAwareFill,
)


@pytest.fixture
def sample_bar():
    return BarData(
        ticker="AAPL",
        open=150.0,
        high=155.0,
        low=148.0,
        close=152.0,
        volume=50_000_000,
        avg_daily_volume=45_000_000,
    )


@pytest.fixture
def bar_with_bid_ask():
    return BarData(
        ticker="AAPL",
        open=150.0,
        high=155.0,
        low=148.0,
        close=152.0,
        volume=50_000_000,
        avg_daily_volume=45_000_000,
        bid=151.98,
        ask=152.02,
    )


@pytest.fixture
def illiquid_bar():
    return BarData(
        ticker="SMOL",
        open=10.0,
        high=10.80,
        low=9.50,
        close=10.20,
        volume=50_000,
        avg_daily_volume=40_000,
    )


class TestMidPriceFill:
    def test_buy_fills_at_close(self, sample_bar):
        model = MidPriceFill()
        fill = model.simulate_fill(sample_bar, 100, "buy")
        assert fill.fill_price == 152.0
        assert fill.spread_cost == 0.0
        assert fill.impact_cost == 0.0
        assert fill.total_cost == 0.0

    def test_sell_fills_at_close(self, sample_bar):
        model = MidPriceFill()
        fill = model.simulate_fill(sample_bar, 100, "sell")
        assert fill.fill_price == 152.0
        assert fill.total_cost == 0.0

    def test_records_side_and_shares(self, sample_bar):
        model = MidPriceFill()
        fill = model.simulate_fill(sample_bar, 500, "buy")
        assert fill.side == "buy"
        assert fill.shares == 500
        assert fill.ticker == "AAPL"


class TestSpreadAwareFill:
    def test_buy_costs_more_than_close(self, sample_bar):
        model = SpreadAwareFill()
        fill = model.simulate_fill(sample_bar, 100, "buy")
        assert fill.fill_price > sample_bar.close
        assert fill.spread_cost > 0

    def test_sell_costs_less_than_close(self, sample_bar):
        model = SpreadAwareFill()
        fill = model.simulate_fill(sample_bar, 100, "sell")
        assert fill.fill_price < sample_bar.close
        assert fill.spread_cost > 0

    def test_uses_real_bid_ask_when_available(self, bar_with_bid_ask):
        model = SpreadAwareFill()
        fill = model.simulate_fill(bar_with_bid_ask, 100, "buy")
        # Real spread is $0.04, half = $0.02
        assert abs(fill.fill_price - 152.02) < 0.01

    def test_sell_with_real_bid_ask(self, bar_with_bid_ask):
        model = SpreadAwareFill()
        fill = model.simulate_fill(bar_with_bid_ask, 100, "sell")
        assert abs(fill.fill_price - 151.98) < 0.01

    def test_wider_range_means_wider_spread(self):
        narrow = BarData("X", 100.0, 100.5, 99.5, 100.0, 1e6, 1e6)
        wide = BarData("X", 100.0, 105.0, 95.0, 100.0, 1e6, 1e6)

        model = SpreadAwareFill()
        fill_narrow = model.simulate_fill(narrow, 100, "buy")
        fill_wide = model.simulate_fill(wide, 100, "buy")

        assert fill_wide.spread_cost > fill_narrow.spread_cost

    def test_default_spread_bps_fallback(self):
        flat_bar = BarData("X", 100.0, 100.0, 100.0, 100.0, 1e6, 1e6)
        model = SpreadAwareFill(default_spread_bps=10.0)
        fill = model.simulate_fill(flat_bar, 100, "buy")
        # 10 bps of $100 = $0.10, half = $0.05
        assert abs(fill.spread_cost - 0.05) < 0.01


class TestMarketImpactFill:
    def test_buy_costs_more_than_spread_only(self, sample_bar):
        spread_model = SpreadAwareFill()
        impact_model = MarketImpactFill(eta=0.1)

        fill_spread = spread_model.simulate_fill(sample_bar, 1000, "buy")
        fill_impact = impact_model.simulate_fill(sample_bar, 1000, "buy")

        assert fill_impact.fill_price >= fill_spread.fill_price
        assert fill_impact.impact_cost >= 0

    def test_larger_order_more_impact(self, sample_bar):
        model = MarketImpactFill(eta=0.1)
        fill_small = model.simulate_fill(sample_bar, 100, "buy")
        fill_large = model.simulate_fill(sample_bar, 1_000_000, "buy")

        assert fill_large.impact_cost > fill_small.impact_cost

    def test_illiquid_stock_more_impact(self, illiquid_bar):
        model = MarketImpactFill(eta=0.1)
        fill = model.simulate_fill(illiquid_bar, 5000, "buy")
        # 5000 shares on 40K ADV = 12.5% participation → significant impact
        assert fill.impact_cost > 0
        assert fill.fill_price > illiquid_bar.close

    def test_zero_volume_no_crash(self):
        bar = BarData("X", 100.0, 105.0, 95.0, 100.0, 0, 0)
        model = MarketImpactFill(eta=0.1)
        fill = model.simulate_fill(bar, 100, "buy")
        assert fill.impact_cost == 0.0

    def test_sell_price_floored(self, illiquid_bar):
        model = MarketImpactFill(eta=10.0)  # extreme eta
        fill = model.simulate_fill(illiquid_bar, 50000, "sell")
        assert fill.fill_price >= 0.01
