"""Slippage models — additional execution cost beyond spread and impact.

Slippage captures the difference between the expected fill price and
the actual fill price due to timing, order routing, and randomness.

These are additive on top of fill model costs:
    total_execution_cost = fill_model_cost + slippage + commission

Models:
    1. ZeroSlippage     — no additional slippage
    2. FixedSlippage    — constant bps per trade
    3. VolumeSlippage   — scales with participation rate (shares / ADV)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SlippageResult:
    """Slippage calculation result (per share, in price terms)."""

    cost_per_share: float   # slippage cost per share
    bps: float              # slippage in basis points


class SlippageModel(ABC):
    """Abstract slippage model."""

    @abstractmethod
    def calculate(
        self,
        price: float,
        shares: int,
        avg_daily_volume: float,
    ) -> SlippageResult:
        """Calculate slippage for a trade.

        Args:
            price: Expected fill price.
            shares: Order size (always positive).
            avg_daily_volume: 20-day average daily volume in shares.

        Returns:
            SlippageResult with per-share cost.
        """
        ...


class ZeroSlippage(SlippageModel):
    """No slippage — baseline for comparison."""

    def calculate(
        self, price: float, shares: int, avg_daily_volume: float,
    ) -> SlippageResult:
        return SlippageResult(cost_per_share=0.0, bps=0.0)


class FixedSlippage(SlippageModel):
    """Fixed slippage in basis points per trade.

    Args:
        bps: Slippage in basis points. Default 5 bps.
             Typical ranges: 2-5 bps for large-cap, 10-20 bps for small-cap.
    """

    def __init__(self, bps: float = 5.0):
        self._bps = bps

    def calculate(
        self, price: float, shares: int, avg_daily_volume: float,
    ) -> SlippageResult:
        cost = (self._bps / 10_000) * price
        return SlippageResult(cost_per_share=cost, bps=self._bps)


class VolumeSlippage(SlippageModel):
    """Volume-proportional slippage — scales with participation rate.

    slippage_bps = base_bps + scale * (shares / ADV) * 10_000

    The idea: trading 0.1% of ADV has minimal extra slippage,
    but trading 5% of ADV has significant slippage.

    Args:
        base_bps: Minimum slippage regardless of size (default 2 bps).
        scale: Sensitivity to participation rate (default 0.1).
    """

    def __init__(self, base_bps: float = 2.0, scale: float = 0.1):
        self._base_bps = base_bps
        self._scale = scale

    def calculate(
        self, price: float, shares: int, avg_daily_volume: float,
    ) -> SlippageResult:
        participation = shares / avg_daily_volume if avg_daily_volume > 0 else 0.0
        bps = self._base_bps + self._scale * participation * 10_000
        cost = (bps / 10_000) * price
        return SlippageResult(cost_per_share=cost, bps=bps)
