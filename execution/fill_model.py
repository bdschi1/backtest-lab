"""Fill models — simulate how orders get filled at realistic prices.

Three tiers of realism:
    1. MidPriceFill       — uses close price (yfinance baseline)
    2. SpreadAwareFill    — estimates bid/ask from daily range or uses real quotes
    3. MarketImpactFill   — Almgren-Chriss sqrt model using ADV

The engine selects the best available model based on the data provider.
All fill models are stateless — they take a bar of data and return an
execution price.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BarData:
    """Single bar of OHLCV data for one ticker on one date.

    All prices in local currency. Volume in shares.
    bid/ask are None when the provider doesn't support them.
    """

    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    avg_daily_volume: float = 0.0  # 20-day ADV in shares
    bid: float | None = None
    ask: float | None = None


@dataclass(frozen=True)
class Fill:
    """Result of a fill simulation."""

    ticker: str
    fill_price: float
    spread_cost: float    # cost attributed to the bid-ask spread
    impact_cost: float    # cost attributed to market impact
    total_cost: float     # spread_cost + impact_cost (per share, signed)
    shares: int
    side: str             # "buy" or "sell"


class FillModel(ABC):
    """Abstract fill model interface."""

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def simulate_fill(
        self,
        bar: BarData,
        shares: int,
        side: str,
    ) -> Fill:
        """Simulate order execution and return a Fill.

        Args:
            bar: Current bar data for the ticker.
            shares: Number of shares to trade (always positive).
            side: "buy" or "sell".

        Returns:
            Fill with execution price and cost breakdown.
        """
        ...


class MidPriceFill(FillModel):
    """Fill at the closing price — no spread or impact modeled.

    This is what most yfinance-based backtests do.
    Useful as a baseline to measure how much execution costs matter.
    """

    @property
    def name(self) -> str:
        return "Mid-Price (Close)"

    def simulate_fill(
        self,
        bar: BarData,
        shares: int,
        side: str,
    ) -> Fill:
        return Fill(
            ticker=bar.ticker,
            fill_price=bar.close,
            spread_cost=0.0,
            impact_cost=0.0,
            total_cost=0.0,
            shares=shares,
            side=side,
        )


class SpreadAwareFill(FillModel):
    """Fill with estimated or real bid-ask spread.

    If real bid/ask data is available (Bloomberg, IB), uses it directly.
    Otherwise, estimates the spread as a fraction of the daily range:
        estimated_spread = spread_pct * close

    Buys fill at ask (close + half_spread), sells at bid (close - half_spread).

    Args:
        default_spread_bps: Fallback spread in basis points when no bid/ask
                            and daily range is zero. Default 10 bps.
    """

    def __init__(self, default_spread_bps: float = 10.0):
        self._default_spread_bps = default_spread_bps

    @property
    def name(self) -> str:
        return "Spread-Aware"

    def _estimate_spread(self, bar: BarData) -> float:
        """Estimate half-spread in price terms."""
        if bar.bid is not None and bar.ask is not None and bar.ask > bar.bid:
            # Real bid/ask available
            return (bar.ask - bar.bid) / 2.0

        # Estimate from daily range — Corwin & Schultz (2012) inspired
        # Simple version: spread ~ 20% of daily range for liquid stocks
        daily_range = bar.high - bar.low
        if daily_range > 0 and bar.close > 0:
            # Scale: wider range → wider spread estimate
            spread_ratio = 0.2 * daily_range / bar.close
            return spread_ratio * bar.close / 2.0

        # Fallback: use default bps
        return (self._default_spread_bps / 10_000) * bar.close / 2.0

    def simulate_fill(
        self,
        bar: BarData,
        shares: int,
        side: str,
    ) -> Fill:
        half_spread = self._estimate_spread(bar)

        if side == "buy":
            fill_price = bar.close + half_spread
            spread_cost = half_spread  # per share
        else:
            fill_price = bar.close - half_spread
            spread_cost = half_spread  # per share (cost is always positive)

        return Fill(
            ticker=bar.ticker,
            fill_price=fill_price,
            spread_cost=spread_cost,
            impact_cost=0.0,
            total_cost=spread_cost,
            shares=shares,
            side=side,
        )


class MarketImpactFill(FillModel):
    """Fill with spread + market impact (Almgren-Chriss square-root model).

    Market impact = eta * sigma * sqrt(shares / ADV)

    Where:
        eta:    impact coefficient (calibrated, default 0.1)
        sigma:  daily volatility (estimated from bar range)
        shares: order size
        ADV:    average daily volume

    This is the model that separates a real backtest from a paper trading toy.
    Without it, you're implicitly assuming infinite liquidity.

    Args:
        eta: Impact coefficient. Higher = more conservative.
             0.05-0.10 for liquid large-caps, 0.15-0.25 for mid/small-caps.
        default_spread_bps: Fallback spread when no bid/ask available.
    """

    def __init__(
        self,
        eta: float = 0.1,
        default_spread_bps: float = 10.0,
    ):
        self._eta = eta
        self._default_spread_bps = default_spread_bps

    @property
    def name(self) -> str:
        return "Market Impact (Almgren-Chriss)"

    def _estimate_half_spread(self, bar: BarData) -> float:
        """Same spread estimation as SpreadAwareFill."""
        if bar.bid is not None and bar.ask is not None and bar.ask > bar.bid:
            return (bar.ask - bar.bid) / 2.0

        daily_range = bar.high - bar.low
        if daily_range > 0 and bar.close > 0:
            spread_ratio = 0.2 * daily_range / bar.close
            return spread_ratio * bar.close / 2.0

        return (self._default_spread_bps / 10_000) * bar.close / 2.0

    def _estimate_impact(self, bar: BarData, shares: int) -> float:
        """Estimate market impact in price terms (per share).

        Uses Almgren-Chriss square-root model:
            impact = eta * sigma_daily * sqrt(shares / ADV)
        """
        adv = bar.avg_daily_volume
        if adv <= 0 or bar.close <= 0:
            # Can't estimate impact without volume data
            return 0.0

        participation = shares / adv
        if participation <= 0:
            return 0.0

        # Estimate daily vol from intraday range (Parkinson estimator)
        if bar.high > bar.low and bar.close > 0:
            log_range = math.log(bar.high / bar.low)
            sigma = log_range / (2.0 * math.sqrt(math.log(2)))
        else:
            sigma = 0.02  # fallback 2% daily vol

        # Almgren-Chriss: impact = eta * sigma * sqrt(participation)
        impact_pct = self._eta * sigma * math.sqrt(participation)
        return impact_pct * bar.close

    def simulate_fill(
        self,
        bar: BarData,
        shares: int,
        side: str,
    ) -> Fill:
        half_spread = self._estimate_half_spread(bar)
        impact = self._estimate_impact(bar, shares)

        if side == "buy":
            fill_price = bar.close + half_spread + impact
        else:
            fill_price = bar.close - half_spread - impact
            # Floor at zero
            fill_price = max(fill_price, 0.01)

        return Fill(
            ticker=bar.ticker,
            fill_price=fill_price,
            spread_cost=half_spread,
            impact_cost=impact,
            total_cost=half_spread + impact,
            shares=shares,
            side=side,
        )
