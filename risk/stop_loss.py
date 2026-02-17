"""Volatility-adjusted stop-loss models.

Not arbitrary percentage stops — ATR-based trailing stops that adapt
to each stock's realized volatility.

The risk manager calls check_stops() before every rebalance.
Any position that has breached its stop is force-closed.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StopResult:
    """Result of a stop-loss check for one position."""

    ticker: str
    triggered: bool
    reason: str
    entry_price: float
    current_price: float
    stop_price: float


class StopLossModel(ABC):
    """Abstract stop-loss model."""

    @abstractmethod
    def check_stop(
        self,
        ticker: str,
        shares: int,
        entry_price: float,
        current_price: float,
        high_water_mark: float,
        atr: float,
    ) -> StopResult:
        """Check if a position should be stopped out.

        Args:
            ticker: Ticker symbol.
            shares: Current position (positive=long, negative=short).
            entry_price: Average cost basis.
            current_price: Latest price.
            high_water_mark: Best price since entry (for trailing stops).
            atr: Average True Range (14-day) for the stock.

        Returns:
            StopResult indicating whether the stop was triggered.
        """
        ...


class NoStopLoss(StopLossModel):
    """No stop-loss — baseline."""

    def check_stop(self, ticker, shares, entry_price, current_price,
                   high_water_mark, atr) -> StopResult:
        return StopResult(
            ticker=ticker, triggered=False, reason="no stop",
            entry_price=entry_price, current_price=current_price,
            stop_price=0.0,
        )


class ATRTrailingStop(StopLossModel):
    """ATR-based trailing stop.

    Stop price trails the high-water mark by N * ATR.

    For longs:  stop = HWM - (multiplier * ATR)
    For shorts: stop = LWM + (multiplier * ATR)

    Args:
        multiplier: Number of ATRs below HWM (default 3.0).
                    Wider = less likely to get stopped out by noise.
                    2.0 = tight (for short-term), 4.0 = loose (swing).
        max_loss_pct: Hard cap on maximum loss from entry (default 15%).
    """

    def __init__(self, multiplier: float = 3.0, max_loss_pct: float = 15.0):
        self._multiplier = multiplier
        self._max_loss_pct = max_loss_pct / 100.0

    def check_stop(self, ticker, shares, entry_price, current_price,
                   high_water_mark, atr) -> StopResult:
        if atr <= 0:
            atr = entry_price * 0.02  # fallback 2% daily range

        if shares > 0:
            # Long position
            trailing_stop = high_water_mark - (self._multiplier * atr)
            hard_stop = entry_price * (1 - self._max_loss_pct)
            stop_price = max(trailing_stop, hard_stop)

            if current_price <= stop_price:
                if current_price <= hard_stop:
                    reason = f"hard stop: loss exceeded {self._max_loss_pct:.0%}"
                else:
                    reason = f"trailing stop: {self._multiplier}x ATR from HWM"
                return StopResult(
                    ticker=ticker, triggered=True, reason=reason,
                    entry_price=entry_price, current_price=current_price,
                    stop_price=stop_price,
                )
        else:
            # Short position — stop above low-water mark
            low_water_mark = high_water_mark  # reused as LWM for shorts
            trailing_stop = low_water_mark + (self._multiplier * atr)
            hard_stop = entry_price * (1 + self._max_loss_pct)
            stop_price = min(trailing_stop, hard_stop)

            if current_price >= stop_price:
                if current_price >= hard_stop:
                    reason = f"hard stop: short loss exceeded {self._max_loss_pct:.0%}"
                else:
                    reason = f"trailing stop: {self._multiplier}x ATR from LWM"
                return StopResult(
                    ticker=ticker, triggered=True, reason=reason,
                    entry_price=entry_price, current_price=current_price,
                    stop_price=stop_price,
                )

        return StopResult(
            ticker=ticker, triggered=False, reason="within bounds",
            entry_price=entry_price, current_price=current_price,
            stop_price=stop_price if shares > 0 else stop_price,
        )
