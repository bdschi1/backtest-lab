"""Regime detection — classify the current market environment.

Two approaches:
    1. VolatilityRegime: Simple rolling-volatility thresholds
       (fast, interpretable, no fitting required)
    2. HMMRegime: Hidden Markov Model with 2-3 states
       (more sophisticated, requires fitting, can overfit)

The detector runs once per bar and returns the current regime.
The adapter then adjusts strategy parameters accordingly.

Regime classification from multi-agent-investment-committee:
    LOW:      vol < 12% ann → risk-on, full sizing
    NORMAL:   12-20% → standard parameters
    ELEVATED: 20-30% → reduce gross, tighten stops
    CRISIS:   >30% → defensive mode, minimal new positions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class Regime(Enum):
    """Market regime classification."""
    LOW = "low_vol"
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRISIS = "crisis"


@dataclass(frozen=True)
class RegimeState:
    """Current regime state with metadata."""

    regime: Regime
    annualized_vol: float
    vol_percentile: float    # where current vol sits in history (0-100)
    regime_duration: int     # days in current regime
    transition_from: Regime | None  # previous regime (None if first)


class VolatilityRegimeDetector:
    """Classify market regime based on rolling realized volatility.

    Uses annualized realized volatility of a broad index or
    portfolio returns.

    Args:
        lookback_days: Window for vol calculation (default 21 = 1 month).
        low_threshold: Annualized vol below this = LOW (default 12%).
        elevated_threshold: Vol above this = ELEVATED (default 20%).
        crisis_threshold: Vol above this = CRISIS (default 30%).
        history_days: Days of vol history for percentile calc (default 252).
    """

    def __init__(
        self,
        lookback_days: int = 21,
        low_threshold: float = 12.0,
        elevated_threshold: float = 20.0,
        crisis_threshold: float = 30.0,
        history_days: int = 252,
    ):
        self._lookback = lookback_days
        self._low = low_threshold / 100.0
        self._elevated = elevated_threshold / 100.0
        self._crisis = crisis_threshold / 100.0
        self._history_days = history_days

        # State
        self._vol_history: list[float] = []
        self._current_regime = Regime.NORMAL
        self._regime_duration = 0
        self._prev_regime: Regime | None = None

    @property
    def current_regime(self) -> Regime:
        return self._current_regime

    def update(self, daily_returns: list[float]) -> RegimeState:
        """Update regime with latest daily returns.

        Args:
            daily_returns: List of recent daily returns (at least lookback_days).
                          Most recent return last.

        Returns:
            Current RegimeState.
        """
        if len(daily_returns) < self._lookback:
            return RegimeState(
                regime=Regime.NORMAL,
                annualized_vol=0.0,
                vol_percentile=50.0,
                regime_duration=0,
                transition_from=None,
            )

        # Compute realized vol
        window = daily_returns[-self._lookback:]
        daily_vol = np.std(window)
        ann_vol = daily_vol * np.sqrt(252)

        # Track history
        self._vol_history.append(ann_vol)
        if len(self._vol_history) > self._history_days:
            self._vol_history = self._vol_history[-self._history_days:]

        # Percentile
        if len(self._vol_history) > 1:
            percentile = (
                np.searchsorted(sorted(self._vol_history), ann_vol)
                / len(self._vol_history) * 100
            )
        else:
            percentile = 50.0

        # Classify
        prev = self._current_regime
        if ann_vol < self._low:
            new_regime = Regime.LOW
        elif ann_vol < self._elevated:
            new_regime = Regime.NORMAL
        elif ann_vol < self._crisis:
            new_regime = Regime.ELEVATED
        else:
            new_regime = Regime.CRISIS

        if new_regime != prev:
            self._prev_regime = prev
            self._regime_duration = 1
            logger.info(
                "Regime change: %s → %s (vol=%.1f%%)",
                prev.value, new_regime.value, ann_vol * 100,
            )
        else:
            self._regime_duration += 1

        self._current_regime = new_regime

        return RegimeState(
            regime=new_regime,
            annualized_vol=ann_vol,
            vol_percentile=percentile,
            regime_duration=self._regime_duration,
            transition_from=self._prev_regime,
        )
