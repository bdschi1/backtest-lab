"""Commission models — simulate trading costs.

Three models matching common broker structures:
    1. ZeroCommission     — no commission (for isolating spread/impact effects)
    2. PerShareCommission — e.g., $0.005/share (IB-style)
    3. TieredCommission   — volume-based tiers (institutional)

All models are stateless. They take trade details and return total cost.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class CommissionResult:
    """Commission calculation result."""

    total: float          # total commission for this trade
    per_share: float      # effective per-share rate
    rate_type: str        # "zero", "per_share", "tiered"


class CommissionModel(ABC):
    """Abstract commission model."""

    @abstractmethod
    def calculate(self, shares: int, price: float) -> CommissionResult:
        """Calculate commission for a trade.

        Args:
            shares: Number of shares traded (always positive).
            price: Execution price per share.

        Returns:
            CommissionResult with total cost and per-share breakdown.
        """
        ...


class ZeroCommission(CommissionModel):
    """No commission — useful for isolating spread and impact effects."""

    def calculate(self, shares: int, price: float) -> CommissionResult:
        return CommissionResult(total=0.0, per_share=0.0, rate_type="zero")


class PerShareCommission(CommissionModel):
    """Per-share commission (e.g., Interactive Brokers).

    Args:
        rate: Cost per share (default $0.005 = IB's standard).
        minimum: Minimum commission per trade (default $1.00).
        maximum_pct: Maximum as % of trade value (default 1.0%).
    """

    def __init__(
        self,
        rate: float = 0.005,
        minimum: float = 1.00,
        maximum_pct: float = 1.0,
    ):
        self._rate = rate
        self._minimum = minimum
        self._maximum_pct = maximum_pct / 100.0

    def calculate(self, shares: int, price: float) -> CommissionResult:
        raw = shares * self._rate
        trade_value = shares * price

        # Apply floor and ceiling
        commission = max(raw, self._minimum)
        if trade_value > 0:
            commission = min(commission, trade_value * self._maximum_pct)

        per_share = commission / shares if shares > 0 else 0.0
        return CommissionResult(
            total=commission,
            per_share=per_share,
            rate_type="per_share",
        )


class TieredCommission(CommissionModel):
    """Volume-based tiered commission (institutional pricing).

    Tiers are defined as (monthly_volume_threshold, per_share_rate).
    The rate applied is based on the trader's monthly volume tier.

    Default tiers (IB-inspired):
        <=300K shares/month:  $0.0035/share
        <=3M shares/month:    $0.0020/share
        <=20M shares/month:   $0.0015/share
        >20M shares/month:    $0.0010/share

    Args:
        tier: Which tier to use (0-3). Default 0 (retail).
        minimum: Minimum per trade (default $0.35).
        maximum_pct: Maximum as % of trade value (default 1.0%).
    """

    _TIERS = [
        0.0035,  # Tier 0: <=300K shares/month
        0.0020,  # Tier 1: <=3M shares/month
        0.0015,  # Tier 2: <=20M shares/month
        0.0010,  # Tier 3: >20M shares/month (institutional)
    ]

    def __init__(
        self,
        tier: int = 0,
        minimum: float = 0.35,
        maximum_pct: float = 1.0,
    ):
        self._rate = self._TIERS[min(tier, len(self._TIERS) - 1)]
        self._minimum = minimum
        self._maximum_pct = maximum_pct / 100.0

    def calculate(self, shares: int, price: float) -> CommissionResult:
        raw = shares * self._rate
        trade_value = shares * price

        commission = max(raw, self._minimum)
        if trade_value > 0:
            commission = min(commission, trade_value * self._maximum_pct)

        per_share = commission / shares if shares > 0 else 0.0
        return CommissionResult(
            total=commission,
            per_share=per_share,
            rate_type="tiered",
        )
