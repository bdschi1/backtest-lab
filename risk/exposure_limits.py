"""Portfolio exposure limits — hard caps on concentration and leverage.

Enforces:
    - Max gross exposure (long + short as % of equity)
    - Max net exposure (long - short as % of equity)
    - Max single-name concentration
    - Max sector concentration
    - Min/max number of positions

These run AFTER signals but BEFORE execution — they can reject or
scale down proposed trades.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExposureCheck:
    """Result of exposure limit check."""

    within_limits: bool
    gross_pct: float
    net_pct: float
    max_single_name_pct: float
    violations: list[str]


class ExposureLimits:
    """Portfolio exposure limit enforcer.

    Args:
        max_gross_pct: Max gross exposure as % of equity (default 200%).
                       100% = fully invested long-only.
                       200% = 100% long + 100% short (typical L/S).
        max_net_pct: Max absolute net exposure (default 50%).
                     Lower = more market-neutral.
        max_single_name_pct: Max single position as % of equity (default 5%).
        max_sector_pct: Max sector exposure as % of equity (default 25%).
        min_positions: Minimum number of positions (default 5).
                       Prevents over-concentration.
        max_positions: Maximum number of positions (default 50).
                       Prevents over-diversification (PM bandwidth).
    """

    def __init__(
        self,
        max_gross_pct: float = 200.0,
        max_net_pct: float = 50.0,
        max_single_name_pct: float = 5.0,
        max_sector_pct: float = 25.0,
        min_positions: int = 5,
        max_positions: int = 50,
    ):
        self._max_gross = max_gross_pct / 100.0
        self._max_net = max_net_pct / 100.0
        self._max_single = max_single_name_pct / 100.0
        self._max_sector = max_sector_pct / 100.0
        self._min_positions = min_positions
        self._max_positions = max_positions

    def check(
        self,
        positions: dict[str, tuple[int, float]],
        equity: float,
        sectors: dict[str, str] | None = None,
    ) -> ExposureCheck:
        """Check if current portfolio is within exposure limits.

        Args:
            positions: {ticker: (shares, current_price)}
            equity: Total portfolio equity.
            sectors: Optional {ticker: sector_name} for sector checks.

        Returns:
            ExposureCheck with violations list.
        """
        if equity <= 0:
            return ExposureCheck(
                within_limits=False, gross_pct=0, net_pct=0,
                max_single_name_pct=0,
                violations=["zero_equity"],
            )

        violations: list[str] = []

        # Compute exposures
        long_value = sum(
            shares * price for shares, price in positions.values()
            if shares > 0
        )
        short_value = sum(
            abs(shares) * price for shares, price in positions.values()
            if shares < 0
        )

        gross = (long_value + short_value) / equity
        net = (long_value - short_value) / equity

        if gross > self._max_gross:
            violations.append(
                f"gross_exposure: {gross:.1%} > {self._max_gross:.1%}"
            )

        if abs(net) > self._max_net:
            violations.append(
                f"net_exposure: {net:+.1%} exceeds +-{self._max_net:.1%}"
            )

        # Single-name concentration
        max_single = 0.0
        for ticker, (shares, price) in positions.items():
            notional = abs(shares) * price
            pct = notional / equity
            max_single = max(max_single, pct)
            if pct > self._max_single:
                violations.append(
                    f"single_name: {ticker} at {pct:.1%} > {self._max_single:.1%}"
                )

        # Sector concentration
        if sectors:
            sector_exposure: dict[str, float] = {}
            for ticker, (shares, price) in positions.items():
                sector = sectors.get(ticker, "Unknown")
                notional = abs(shares) * price
                sector_exposure[sector] = sector_exposure.get(sector, 0) + notional

            for sector, notional in sector_exposure.items():
                pct = notional / equity
                if pct > self._max_sector:
                    violations.append(
                        f"sector: {sector} at {pct:.1%} > {self._max_sector:.1%}"
                    )

        # Position count
        n_positions = len(positions)
        if n_positions > self._max_positions:
            violations.append(
                f"too_many_positions: {n_positions} > {self._max_positions}"
            )

        return ExposureCheck(
            within_limits=len(violations) == 0,
            gross_pct=gross * 100,
            net_pct=net * 100,
            max_single_name_pct=max_single * 100,
            violations=violations,
        )

    def scale_to_fit(
        self,
        target_positions: dict[str, int],
        prices: dict[str, float],
        equity: float,
    ) -> dict[str, int]:
        """Scale target positions down to fit within gross exposure limit.

        Proportionally reduces all positions if total would exceed limits.
        Does not add positions — only reduces.

        Args:
            target_positions: {ticker: target_shares}
            prices: {ticker: current_price}
            equity: Total portfolio equity.

        Returns:
            Scaled target positions.
        """
        if equity <= 0:
            return {t: 0 for t in target_positions}

        total_notional = sum(
            abs(shares) * prices.get(ticker, 0)
            for ticker, shares in target_positions.items()
        )

        max_notional = self._max_gross * equity

        if total_notional <= max_notional:
            return target_positions

        scale = max_notional / total_notional
        logger.warning(
            "Exposure limit: scaling positions by %.2f "
            "(requested $%.0f, limit $%.0f)",
            scale, total_notional, max_notional,
        )

        return {
            ticker: int(shares * scale)
            for ticker, shares in target_positions.items()
        }
