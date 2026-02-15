"""Short borrow cost models — critical for long/short strategies.

Models the cost of borrowing shares for short selling:
    - Annualized borrow rate applied daily
    - Hard-to-borrow detection from short interest

Without modeling borrow costs, a backtest overstates short-side P&L,
sometimes dramatically (10-30% annualized for crowded shorts).

Models:
    1. ZeroBorrow     — no borrow cost (long-only or baseline)
    2. FixedBorrow    — flat annualized rate for all names
    3. TieredBorrow   — rate depends on short interest / availability
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class BorrowResult:
    """Daily borrow cost for a short position."""

    daily_cost: float          # total daily cost in dollars
    annualized_rate: float     # rate used (as decimal, e.g., 0.005 = 50 bps)
    is_hard_to_borrow: bool    # flagged as hard-to-borrow


class BorrowModel(ABC):
    """Abstract borrow cost model."""

    @abstractmethod
    def daily_cost(
        self,
        shares: int,
        price: float,
        short_pct_of_float: float = 0.0,
    ) -> BorrowResult:
        """Calculate daily borrow cost for a short position.

        Args:
            shares: Number of shares held short (always positive).
            price: Current price per share.
            short_pct_of_float: Short interest as % of float (0-100).

        Returns:
            BorrowResult with daily cost and rate.
        """
        ...


class ZeroBorrow(BorrowModel):
    """No borrow cost — for long-only strategies or baseline."""

    def daily_cost(
        self, shares: int, price: float, short_pct_of_float: float = 0.0,
    ) -> BorrowResult:
        return BorrowResult(
            daily_cost=0.0,
            annualized_rate=0.0,
            is_hard_to_borrow=False,
        )


class FixedBorrow(BorrowModel):
    """Fixed annualized borrow rate for all names.

    Args:
        annualized_rate: Annual borrow rate as decimal.
                         Default 0.005 (50 bps) — general collateral rate.
                         Typical ranges:
                           - GC (easy to borrow): 25-75 bps
                           - Warm: 1-3%
                           - Hard-to-borrow: 5-30%+
    """

    def __init__(self, annualized_rate: float = 0.005):
        self._rate = annualized_rate

    def daily_cost(
        self, shares: int, price: float, short_pct_of_float: float = 0.0,
    ) -> BorrowResult:
        notional = shares * price
        daily = notional * self._rate / 252  # trading days
        return BorrowResult(
            daily_cost=daily,
            annualized_rate=self._rate,
            is_hard_to_borrow=False,
        )


class TieredBorrow(BorrowModel):
    """Borrow rate based on short interest / availability.

    Rate tiers based on short % of float:
        <5%:   GC rate (25 bps default)
        5-10%: Warm (150 bps default)
        10-20%: Special (500 bps default)
        >20%:  Hard-to-borrow (1500 bps default)

    Args:
        gc_rate: General collateral rate (default 0.0025 = 25 bps).
        warm_rate: Warm borrow rate (default 0.015 = 150 bps).
        special_rate: Special borrow rate (default 0.05 = 500 bps).
        htb_rate: Hard-to-borrow rate (default 0.15 = 1500 bps).
    """

    def __init__(
        self,
        gc_rate: float = 0.0025,
        warm_rate: float = 0.015,
        special_rate: float = 0.05,
        htb_rate: float = 0.15,
    ):
        self._gc_rate = gc_rate
        self._warm_rate = warm_rate
        self._special_rate = special_rate
        self._htb_rate = htb_rate

    def _get_rate(self, short_pct: float) -> tuple[float, bool]:
        """Determine borrow rate and HTB flag from short interest."""
        if short_pct < 5.0:
            return self._gc_rate, False
        elif short_pct < 10.0:
            return self._warm_rate, False
        elif short_pct < 20.0:
            return self._special_rate, True
        else:
            return self._htb_rate, True

    def daily_cost(
        self, shares: int, price: float, short_pct_of_float: float = 0.0,
    ) -> BorrowResult:
        rate, is_htb = self._get_rate(short_pct_of_float)
        notional = shares * price
        daily = notional * rate / 252

        return BorrowResult(
            daily_cost=daily,
            annualized_rate=rate,
            is_hard_to_borrow=is_htb,
        )
