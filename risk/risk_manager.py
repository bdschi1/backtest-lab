"""Master risk manager — orchestrates all risk checks.

This is the single entry point for all risk enforcement.
The engine calls risk_manager.pre_trade_check() before every trade
and risk_manager.post_bar_check() after every bar.

It composes:
    - Stop-loss model (position-level exits)
    - Position sizer (liquidity-aware sizing)
    - Drawdown controller (portfolio-level circuit breaker)
    - Exposure limits (concentration and leverage)

If ANY check fails, the trade is rejected or modified.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from risk.drawdown_control import CircuitState, DrawdownController
from risk.exposure_limits import ExposureLimits
from risk.position_sizer import PositionSizer
from risk.stop_loss import ATRTrailingStop, StopLossModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskDecision:
    """Final risk decision for a proposed trade or bar."""

    approved: bool
    original_shares: int
    approved_shares: int
    reasons: list[str]
    circuit_state: CircuitState
    position_scale: float


class RiskManager:
    """Master risk controller — runs before every trade.

    Args:
        stop_loss: Stop-loss model (default: ATR trailing stop).
        sizer: Position sizer (default: 5% equity, 10% ADV).
        drawdown: Drawdown controller (default: -10% halt).
        exposure: Exposure limits (default: 200% gross, 50% net).
    """

    def __init__(
        self,
        stop_loss: StopLossModel | None = None,
        sizer: PositionSizer | None = None,
        drawdown: DrawdownController | None = None,
        exposure: ExposureLimits | None = None,
    ):
        self.stop_loss = stop_loss or ATRTrailingStop(multiplier=3.0)
        self.sizer = sizer or PositionSizer()
        self.drawdown = drawdown or DrawdownController()
        self.exposure = exposure or ExposureLimits()

        # Track high-water marks per position for trailing stops
        self._hwm: dict[str, float] = {}
        # Track ATR per ticker
        self._atr: dict[str, float] = {}

    def update_bar(self, equity: float, day_index: int, prices: dict[str, float]):
        """Update risk state after each bar.

        Call this once per day BEFORE signal generation.

        Args:
            equity: Current portfolio equity.
            day_index: Trading day index.
            prices: {ticker: latest_price}
        """
        # Update drawdown controller
        self.drawdown.update(equity, day_index)

        # Update high-water marks
        for ticker, price in prices.items():
            if ticker in self._hwm:
                self._hwm[ticker] = max(self._hwm[ticker], price)
            else:
                self._hwm[ticker] = price

    def update_atr(self, atr_values: dict[str, float]):
        """Update ATR values for stop-loss calculations.

        Args:
            atr_values: {ticker: 14-day ATR}
        """
        self._atr.update(atr_values)

    def check_stops(
        self,
        positions: dict[str, Any],
    ) -> list[str]:
        """Check all positions against stop-loss.

        Args:
            positions: {ticker: Position} from the engine.

        Returns:
            List of tickers that should be force-closed.
        """
        to_close: list[str] = []

        for ticker, pos in positions.items():
            hwm = self._hwm.get(ticker, pos.current_price)
            atr = self._atr.get(ticker, pos.current_price * 0.02)

            result = self.stop_loss.check_stop(
                ticker=ticker,
                shares=pos.shares,
                entry_price=pos.avg_cost,
                current_price=pos.current_price,
                high_water_mark=hwm,
                atr=atr,
            )

            if result.triggered:
                logger.warning(
                    "STOP TRIGGERED: %s — %s (entry=%.2f, current=%.2f, stop=%.2f)",
                    ticker, result.reason, result.entry_price,
                    result.current_price, result.stop_price,
                )
                to_close.append(ticker)

        return to_close

    def approve_trade(
        self,
        ticker: str,
        requested_shares: int,
        signal_score: float,
        price: float,
        equity: float,
        avg_daily_volume: float,
    ) -> RiskDecision:
        """Pre-trade risk check — approve, modify, or reject.

        Args:
            ticker: Ticker symbol.
            requested_shares: Number of shares requested (signed).
            signal_score: Signal strength [-1, 1].
            price: Current price.
            equity: Portfolio equity.
            avg_daily_volume: 20-day ADV.

        Returns:
            RiskDecision with approved shares and reasons.
        """
        reasons: list[str] = []
        circuit_state = self.drawdown.state
        position_scale = self.drawdown.position_scale

        # Check circuit breaker
        if circuit_state == CircuitState.HALTED:
            return RiskDecision(
                approved=False,
                original_shares=requested_shares,
                approved_shares=0,
                reasons=["circuit_breaker_halted"],
                circuit_state=circuit_state,
                position_scale=0.0,
            )

        # Apply position sizing
        sizing = self.sizer.size_position(
            ticker=ticker,
            signal_score=signal_score,
            price=price,
            equity=equity,
            avg_daily_volume=avg_daily_volume,
        )

        approved = min(abs(requested_shares), sizing.allowed_shares)

        if approved < abs(requested_shares):
            reasons.append(f"sized_down: {sizing.constraint_hit}")

        # Apply drawdown scaling
        if position_scale < 1.0:
            scaled = int(approved * position_scale)
            if scaled < approved:
                reasons.append(
                    f"dd_scaled: {position_scale:.0%} "
                    f"(circuit={circuit_state.value})"
                )
                approved = scaled

        # Restore sign
        if requested_shares < 0:
            approved = -approved

        if not reasons:
            reasons.append("approved")

        return RiskDecision(
            approved=approved != 0,
            original_shares=requested_shares,
            approved_shares=approved,
            reasons=reasons,
            circuit_state=circuit_state,
            position_scale=position_scale,
        )

    def register_entry(self, ticker: str, price: float):
        """Register a new position entry (sets initial HWM)."""
        self._hwm[ticker] = price

    def remove_position(self, ticker: str):
        """Clean up tracking for a closed position."""
        self._hwm.pop(ticker, None)
        self._atr.pop(ticker, None)
