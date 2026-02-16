"""Volatility-target scaling for portfolio positions.

At each rebalance, the scaler computes trailing realized portfolio volatility
and produces a scalar multiplier that scales all position sizes to achieve a
target annualized volatility.

    scale = target_vol / realized_vol

Clamped to [min_leverage, max_leverage] to avoid extreme positions.

Reference:
    Moskowitz, Ooi, Pedersen (2012) - "Time Series Momentum"
    Barroso & Santa-Clara (2015) - "Momentum has its moments"
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


class VolTargetScaler:
    """Scales portfolio positions to target a specific annualized volatility.

    Args:
        target_vol_pct: Target annualized portfolio volatility as percentage
                        (e.g. 10.0 for 10% annual vol).
        lookback_days: Number of trailing days for realized vol calculation.
                       63 (quarterly) is a common default; shorter = more responsive.
        max_leverage: Maximum scaling factor (caps vol-target scaling).
                      3.0 means positions can scale up to 3x.
        min_leverage: Minimum scaling factor (floor on scaling).
                      0.1 means positions scale down to at least 10%.
    """

    def __init__(
        self,
        target_vol_pct: float = 10.0,
        lookback_days: int = 63,
        max_leverage: float = 3.0,
        min_leverage: float = 0.1,
    ):
        self._target_vol = target_vol_pct / 100.0  # convert to decimal
        self._lookback = lookback_days
        self._max_leverage = max_leverage
        self._min_leverage = min_leverage

    @property
    def target_vol_pct(self) -> float:
        return self._target_vol * 100.0

    @property
    def lookback_days(self) -> int:
        return self._lookback

    def compute_scale(self, daily_returns: list[float]) -> float:
        """Compute the vol-target scaling factor.

        Args:
            daily_returns: List of portfolio daily returns (most recent last).
                          Needs at least lookback_days entries for a valid estimate.

        Returns:
            Scaling factor to apply to all target position sizes.
            Returns 1.0 if insufficient data.
        """
        if len(daily_returns) < self._lookback:
            logger.debug(
                "Vol-target: insufficient data (%d < %d), returning scale=1.0",
                len(daily_returns), self._lookback,
            )
            return 1.0

        # Trailing realized vol (annualized)
        recent = np.array(daily_returns[-self._lookback:])
        daily_vol = np.std(recent, ddof=1)

        if daily_vol < 1e-10:
            logger.debug("Vol-target: near-zero realized vol, returning scale=1.0")
            return 1.0

        realized_annual_vol = daily_vol * math.sqrt(252)
        raw_scale = self._target_vol / realized_annual_vol

        # Clamp
        scale = max(self._min_leverage, min(raw_scale, self._max_leverage))

        logger.debug(
            "Vol-target: realized=%.2f%%, target=%.2f%%, raw_scale=%.2f, clamped=%.2f",
            realized_annual_vol * 100, self._target_vol * 100, raw_scale, scale,
        )

        return scale
