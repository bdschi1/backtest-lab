"""Position sizing with liquidity constraints.

Determines maximum allowable position size based on:
    1. Max % of portfolio equity (concentration limit)
    2. Max % of average daily volume (liquidity constraint)
    3. Kelly fraction (optional, risk-adjusted sizing)

The tightest constraint wins — this prevents both concentration
blow-ups and liquidity-driven execution problems.

This is NOT a dashboard metric — it's an enforced gate that runs
before every trade.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SizingResult:
    """Position sizing result."""

    ticker: str
    requested_shares: int
    allowed_shares: int
    constraint_hit: str       # which constraint was binding
    max_pct_equity: float     # max allowed as % of equity
    max_pct_adv: float        # max allowed as % of ADV
    participation_rate: float # actual participation rate


class PositionSizer:
    """Liquidity-aware position sizer.

    Args:
        max_pct_equity: Max position as % of portfolio equity (default 5%).
        max_pct_adv: Max position as % of 20-day ADV (default 10%).
                     A PM at a pod shop might set this to 5% or lower.
                     Higher = more market impact risk.
        max_notional: Hard cap on notional value (default $5M).
                      Prevents any single name from getting too large.
        min_shares: Minimum trade size (default 100 = 1 round lot).
    """

    def __init__(
        self,
        max_pct_equity: float = 5.0,
        max_pct_adv: float = 10.0,
        max_notional: float = 5_000_000.0,
        min_shares: int = 100,
    ):
        self._max_pct_equity = max_pct_equity / 100.0
        self._max_pct_adv = max_pct_adv / 100.0
        self._max_notional = max_notional
        self._min_shares = min_shares

    def size_position(
        self,
        ticker: str,
        signal_score: float,
        price: float,
        equity: float,
        avg_daily_volume: float,
    ) -> SizingResult:
        """Compute allowed position size.

        Args:
            ticker: Ticker symbol.
            signal_score: Signal strength [-1, 1].
            price: Current price per share.
            equity: Total portfolio equity.
            avg_daily_volume: 20-day ADV in shares.

        Returns:
            SizingResult with allowed shares and binding constraint.
        """
        if price <= 0 or equity <= 0:
            return SizingResult(
                ticker=ticker, requested_shares=0, allowed_shares=0,
                constraint_hit="invalid_price_or_equity",
                max_pct_equity=0, max_pct_adv=0, participation_rate=0,
            )

        # Target notional from signal
        target_notional = abs(signal_score) * self._max_pct_equity * equity
        requested_shares = int(target_notional / price)

        # Constraint 1: Max % of equity
        max_from_equity = int((self._max_pct_equity * equity) / price)

        # Constraint 2: Max % of ADV (liquidity)
        if avg_daily_volume > 0:
            max_from_adv = int(self._max_pct_adv * avg_daily_volume)
        else:
            max_from_adv = self._min_shares  # fallback to minimum

        # Constraint 3: Max notional
        max_from_notional = int(self._max_notional / price)

        # Take the tightest constraint
        allowed = min(requested_shares, max_from_equity, max_from_adv, max_from_notional)
        allowed = max(allowed, 0)

        # Determine binding constraint
        if allowed == 0:
            constraint = "below_minimum"
        elif allowed == max_from_adv and max_from_adv < min(max_from_equity, max_from_notional):
            constraint = f"adv_limit ({self._max_pct_adv:.0%} of {avg_daily_volume:,.0f})"
        elif allowed == max_from_notional and max_from_notional < max_from_equity:
            constraint = f"notional_cap (${self._max_notional:,.0f})"
        elif allowed == max_from_equity:
            constraint = f"equity_limit ({self._max_pct_equity:.0%})"
        else:
            constraint = "signal_size"

        participation = allowed / avg_daily_volume if avg_daily_volume > 0 else 0

        return SizingResult(
            ticker=ticker,
            requested_shares=requested_shares,
            allowed_shares=allowed,
            constraint_hit=constraint,
            max_pct_equity=self._max_pct_equity * 100,
            max_pct_adv=self._max_pct_adv * 100,
            participation_rate=participation,
        )
