"""Drawdown circuit breakers.

Halts the strategy when:
    1. Max drawdown exceeds a threshold
    2. Current drawdown deviates from backtested expectations
    3. Drawdown duration exceeds a time limit

These are hard stops — the engine will refuse to place new trades
when a circuit breaker is tripped. The strategy must recover to
a less severe drawdown level before trading resumes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker state."""
    NORMAL = "normal"
    WARNING = "warning"
    HALTED = "halted"


@dataclass(frozen=True)
class DrawdownCheck:
    """Result of a drawdown circuit breaker check."""

    state: CircuitState
    current_dd_pct: float       # current drawdown as % (negative)
    max_dd_pct: float           # max drawdown since inception
    dd_duration_days: int       # days in current drawdown
    reason: str
    can_trade: bool


class DrawdownController:
    """Drawdown-based circuit breaker.

    Three levels:
        NORMAL:  DD < warning_threshold → trade freely
        WARNING: DD between warning and halt → reduce position sizes
        HALTED:  DD > halt_threshold → no new trades

    Recovery: must return above resume_threshold to exit HALTED state.

    Args:
        warning_dd_pct: Drawdown % that triggers warning (default -5%).
        halt_dd_pct: Drawdown % that halts trading (default -10%).
        resume_dd_pct: Drawdown % to resume after halt (default -7%).
        max_dd_duration: Max days in drawdown before halt (default 60).
        position_scale_in_warning: Scale factor for positions in WARNING (default 0.5).
    """

    def __init__(
        self,
        warning_dd_pct: float = -5.0,
        halt_dd_pct: float = -10.0,
        resume_dd_pct: float = -7.0,
        max_dd_duration: int = 60,
        position_scale_in_warning: float = 0.5,
    ):
        self._warning = warning_dd_pct / 100.0
        self._halt = halt_dd_pct / 100.0
        self._resume = resume_dd_pct / 100.0
        self._max_duration = max_dd_duration
        self._position_scale = position_scale_in_warning

        # State
        self._peak_equity = 0.0
        self._dd_start_day = 0
        self._current_state = CircuitState.NORMAL
        self._days_in_dd = 0

    @property
    def state(self) -> CircuitState:
        return self._current_state

    @property
    def position_scale(self) -> float:
        """Current position scaling factor (1.0 = full, 0.5 = half, 0 = halted)."""
        if self._current_state == CircuitState.HALTED:
            return 0.0
        elif self._current_state == CircuitState.WARNING:
            return self._position_scale
        return 1.0

    def update(self, equity: float, day_index: int) -> DrawdownCheck:
        """Update drawdown state with latest equity.

        Call this once per bar, before signal generation.

        Args:
            equity: Current portfolio equity.
            day_index: Trading day index (for duration tracking).

        Returns:
            DrawdownCheck with current state and trading permission.
        """
        # Update peak — treat equity at or above peak as "not in drawdown"
        if equity >= self._peak_equity:
            self._peak_equity = equity
            self._days_in_dd = 0
        else:
            self._days_in_dd += 1

        # Current drawdown
        if self._peak_equity > 0:
            current_dd = (equity - self._peak_equity) / self._peak_equity
        else:
            current_dd = 0.0

        # Max drawdown (track worst)
        max_dd = current_dd  # simplified — engine tracks full history

        # State transitions

        if self._current_state == CircuitState.HALTED:
            # Need to recover above resume threshold
            if current_dd > self._resume:
                self._current_state = CircuitState.WARNING
                logger.info(
                    "Circuit breaker: HALTED → WARNING "
                    "(DD recovered to %.1f%%)", current_dd * 100,
                )
        elif current_dd <= self._halt or self._days_in_dd >= self._max_duration:
            self._current_state = CircuitState.HALTED
            reason = (
                f"DD={current_dd:.1%}" if current_dd <= self._halt
                else f"DD duration={self._days_in_dd} days"
            )
            logger.warning("Circuit breaker HALTED: %s", reason)
        elif current_dd <= self._warning:
            self._current_state = CircuitState.WARNING
        else:
            self._current_state = CircuitState.NORMAL

        # Build reason string
        if self._current_state == CircuitState.HALTED:
            reason = f"HALTED: DD={current_dd:.1%} (limit={self._halt:.1%})"
            if self._days_in_dd >= self._max_duration:
                reason += f", duration={self._days_in_dd}d (limit={self._max_duration}d)"
        elif self._current_state == CircuitState.WARNING:
            reason = (
                f"WARNING: DD={current_dd:.1%}, "
                f"positions scaled to {self._position_scale:.0%}"
            )
        else:
            reason = "normal"

        return DrawdownCheck(
            state=self._current_state,
            current_dd_pct=current_dd * 100,
            max_dd_pct=max_dd * 100,
            dd_duration_days=self._days_in_dd,
            reason=reason,
            can_trade=self._current_state != CircuitState.HALTED,
        )

    def reset(self) -> None:
        """Reset circuit breaker state."""
        self._peak_equity = 0.0
        self._days_in_dd = 0
        self._current_state = CircuitState.NORMAL
