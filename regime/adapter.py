"""Strategy adaptation based on detected regime.

Maps regime → parameter overrides for the backtest engine.
When the market environment changes, the strategy adjusts:

    LOW vol:      Full risk budget, wider stops, more positions
    NORMAL:       Standard parameters
    ELEVATED:     Reduce gross, tighten stops, fewer positions
    CRISIS:       Defensive — minimal new trades, tight stops, low gross

This is how strategies behave when the investing world changes.
A strategy that doesn't adapt to regime is implicitly assuming
the regime never changes — which is always wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from regime.detector import Regime, RegimeState

logger = logging.getLogger(__name__)


@dataclass
class RegimeOverrides:
    """Parameter overrides for a given regime.

    These scale the base parameters from BacktestConfig:
        gross_scale: multiplier on max gross exposure (1.0 = no change)
        position_scale: multiplier on individual position sizes
        stop_multiplier: ATR multiplier adjustment
        signal_threshold: minimum signal to trade (higher = pickier)
        max_new_positions: limit on new positions per rebalance
    """

    gross_scale: float = 1.0
    position_scale: float = 1.0
    stop_multiplier: float = 1.0
    signal_threshold: float = 0.1
    max_new_positions: int = 50


# Default regime parameter mappings
_DEFAULT_OVERRIDES: dict[Regime, RegimeOverrides] = {
    Regime.LOW: RegimeOverrides(
        gross_scale=1.2,        # can run higher gross in low-vol
        position_scale=1.1,     # slightly larger positions
        stop_multiplier=1.2,    # wider stops (less noise-triggered)
        signal_threshold=0.05,  # accept weaker signals
        max_new_positions=50,
    ),
    Regime.NORMAL: RegimeOverrides(
        gross_scale=1.0,
        position_scale=1.0,
        stop_multiplier=1.0,
        signal_threshold=0.1,
        max_new_positions=30,
    ),
    Regime.ELEVATED: RegimeOverrides(
        gross_scale=0.7,        # cut gross by 30%
        position_scale=0.7,     # smaller positions
        stop_multiplier=0.8,    # tighter stops
        signal_threshold=0.2,   # only strong signals
        max_new_positions=10,
    ),
    Regime.CRISIS: RegimeOverrides(
        gross_scale=0.4,        # cut gross by 60%
        position_scale=0.4,     # much smaller positions
        stop_multiplier=0.6,    # very tight stops
        signal_threshold=0.4,   # only highest conviction
        max_new_positions=3,    # almost no new positions
    ),
}


class RegimeAdapter:
    """Adapt strategy parameters to current market regime.

    Args:
        overrides: Custom regime → parameter mapping.
                   Defaults to _DEFAULT_OVERRIDES.
    """

    def __init__(
        self,
        overrides: dict[Regime, RegimeOverrides] | None = None,
    ):
        self._overrides = overrides or _DEFAULT_OVERRIDES.copy()
        self._current_overrides = self._overrides[Regime.NORMAL]

    def adapt(self, regime_state: RegimeState) -> RegimeOverrides:
        """Get parameter overrides for the current regime.

        Args:
            regime_state: Current regime state from the detector.

        Returns:
            RegimeOverrides to apply to the strategy.
        """
        overrides = self._overrides.get(
            regime_state.regime,
            self._overrides[Regime.NORMAL],
        )

        if overrides != self._current_overrides:
            logger.info(
                "Regime adapter: %s → gross_scale=%.1f, "
                "position_scale=%.1f, stop_mult=%.1f, threshold=%.2f",
                regime_state.regime.value,
                overrides.gross_scale,
                overrides.position_scale,
                overrides.stop_multiplier,
                overrides.signal_threshold,
            )
            self._current_overrides = overrides

        return overrides

    @property
    def current_overrides(self) -> RegimeOverrides:
        return self._current_overrides
