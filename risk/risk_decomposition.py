"""Risk decomposition — position, theme, and portfolio-level analytics.

Decomposes portfolio risk into systematic (market/beta) and idiosyncratic
components. Tracks exposures at position, sector/theme, and portfolio levels.

Key concepts:
    - Portfolio vol: annualized standard deviation of daily P&L
    - Beta-net exposure: sum of (position_beta * position_weight)
    - Idiosyncratic returns: total_return - beta * market_return
    - Concentration: HHI and top-N contribution
    - Exposure time series: gross, net, long, short over time

These metrics are computed AFTER the backtest completes, using the
engine's snapshots and position history.

Usage:
    from risk.risk_decomposition import RiskDecomposition
    rd = RiskDecomposition(engine)
    summary = rd.compute_summary()
    exposure_ts = rd.exposure_time_series()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PositionRiskContribution:
    """Risk contribution of a single position."""
    ticker: str
    weight_pct: float           # % of gross exposure
    notional: float             # dollar value
    side: str                   # "long" or "short"
    contribution_to_var: float  # marginal contribution to portfolio variance


class RiskDecomposition:
    """Post-hoc risk decomposition of a completed backtest.

    Analyzes the portfolio's risk characteristics across three levels:
        1. Position-level: individual name contributions, concentration
        2. Theme-level: sector/factor groupings (when sector data available)
        3. Portfolio-level: total vol, beta, idiosyncratic ratio

    Args:
        engine: Completed BacktestEngine instance.
        benchmark_returns: Optional daily benchmark returns for beta decomposition.
        sector_map: Optional {ticker: sector_name} for theme-level grouping.
    """

    def __init__(
        self,
        engine: Any,
        benchmark_returns: list[float] | None = None,
        sector_map: dict[str, str] | None = None,
    ):
        self._engine = engine
        self._benchmark_returns = benchmark_returns
        self._sector_map = sector_map or {}
        self._snapshots = engine.snapshots
        self._trades = engine.trades

    def compute_summary(self) -> dict[str, Any]:
        """Compute full risk decomposition summary.

        Returns dict with keys prefixed by 'risk_' for namespacing.
        """
        if len(self._snapshots) < 5:
            return self._empty_summary()

        result: dict[str, Any] = {}

        # Portfolio-level volatility metrics
        result.update(self._portfolio_vol_metrics())

        # Exposure metrics (from final snapshot + time series stats)
        result.update(self._exposure_metrics())

        # Beta decomposition (if benchmark available)
        result.update(self._beta_decomposition())

        # Concentration metrics
        result.update(self._concentration_metrics())

        # Idiosyncratic return ratio
        result.update(self._idiosyncratic_metrics())

        return result

    def _portfolio_vol_metrics(self) -> dict[str, Any]:
        """Portfolio-level volatility at different horizons."""
        returns = np.array([s.daily_return for s in self._snapshots])

        daily_vol = float(np.std(returns))
        ann_vol = daily_vol * np.sqrt(252)

        # Rolling 21-day vol (monthly)
        rolling_vols = []
        for i in range(21, len(returns)):
            window = returns[i-21:i]
            rolling_vols.append(np.std(window) * np.sqrt(252))

        # Rolling 63-day vol (quarterly)
        rolling_63 = []
        for i in range(63, len(returns)):
            window = returns[i-63:i]
            rolling_63.append(np.std(window) * np.sqrt(252))

        # Downside vol (semi-deviation)
        neg_returns = returns[returns < 0]
        downside_vol = float(np.std(neg_returns) * np.sqrt(252)) if len(neg_returns) > 0 else 0.0

        return {
            "risk_daily_vol_pct": round(daily_vol * 100, 4),
            "risk_ann_vol_pct": round(ann_vol * 100, 2),
            "risk_downside_vol_pct": round(downside_vol * 100, 2),
            "risk_vol_21d_current_pct": round(rolling_vols[-1] * 100, 2) if rolling_vols else 0.0,
            "risk_vol_63d_current_pct": round(rolling_63[-1] * 100, 2) if rolling_63 else 0.0,
            "risk_vol_21d_avg_pct": round(float(np.mean(rolling_vols)) * 100, 2) if rolling_vols else 0.0,
            "risk_vol_21d_max_pct": round(float(np.max(rolling_vols)) * 100, 2) if rolling_vols else 0.0,
            "risk_vol_21d_min_pct": round(float(np.min(rolling_vols)) * 100, 2) if rolling_vols else 0.0,
        }

    def _exposure_metrics(self) -> dict[str, Any]:
        """Gross/net/long/short exposure statistics over time."""
        if not self._snapshots:
            return {}

        equities = [s.equity for s in self._snapshots]
        gross_pcts = []
        net_pcts = []
        long_pcts = []
        short_pcts = []

        for s in self._snapshots:
            eq = s.equity if s.equity > 0 else 1.0
            gross_pcts.append(s.gross_value / eq * 100)
            net_pcts.append((s.long_value - s.short_value) / eq * 100)
            long_pcts.append(s.long_value / eq * 100)
            short_pcts.append(s.short_value / eq * 100)

        return {
            "risk_avg_gross_exposure_pct": round(float(np.mean(gross_pcts)), 1),
            "risk_max_gross_exposure_pct": round(float(np.max(gross_pcts)), 1),
            "risk_avg_net_exposure_pct": round(float(np.mean(net_pcts)), 1),
            "risk_max_net_exposure_pct": round(float(np.max(np.abs(net_pcts))), 1),
            "risk_avg_long_exposure_pct": round(float(np.mean(long_pcts)), 1),
            "risk_avg_short_exposure_pct": round(float(np.mean(short_pcts)), 1),
            "risk_final_gross_pct": round(gross_pcts[-1], 1) if gross_pcts else 0.0,
            "risk_final_net_pct": round(net_pcts[-1], 1) if net_pcts else 0.0,
            "risk_final_long_pct": round(long_pcts[-1], 1) if long_pcts else 0.0,
            "risk_final_short_pct": round(short_pcts[-1], 1) if short_pcts else 0.0,
            "risk_avg_positions": round(float(np.mean([s.num_positions for s in self._snapshots])), 1),
            "risk_max_positions": max(s.num_positions for s in self._snapshots),
        }

    def _beta_decomposition(self) -> dict[str, Any]:
        """Decompose returns into beta and alpha components."""
        if self._benchmark_returns is None:
            return {
                "risk_portfolio_beta": None,
                "risk_beta_pnl_pct": None,
                "risk_alpha_pnl_pct": None,
                "risk_idio_ratio_pct": None,
            }

        strat = np.array([s.daily_return for s in self._snapshots])
        n = min(len(strat), len(self._benchmark_returns))
        if n < 10:
            return {
                "risk_portfolio_beta": None,
                "risk_beta_pnl_pct": None,
                "risk_alpha_pnl_pct": None,
                "risk_idio_ratio_pct": None,
            }

        strat = strat[:n]
        bench = np.array(self._benchmark_returns[:n])

        # Portfolio beta
        bench_var = np.var(bench)
        if bench_var > 0:
            beta = float(np.cov(strat, bench)[0, 1] / bench_var)
        else:
            beta = 0.0

        # Decompose cumulative returns
        total_return = float((1 + strat).prod() - 1)
        bench_return = float((1 + bench).prod() - 1)
        beta_return = beta * bench_return
        alpha_return = total_return - beta_return

        # Idiosyncratic ratio: what % of variance is NOT explained by market
        r_squared = float(np.corrcoef(strat, bench)[0, 1]) ** 2 if bench_var > 0 else 0.0
        idio_ratio = (1 - r_squared) * 100

        return {
            "risk_portfolio_beta": round(beta, 3),
            "risk_beta_pnl_pct": round(beta_return * 100, 2),
            "risk_alpha_pnl_pct": round(alpha_return * 100, 2),
            "risk_idio_ratio_pct": round(idio_ratio, 1),
        }

    def _concentration_metrics(self) -> dict[str, Any]:
        """Position concentration (HHI and top-N)."""
        # Use final snapshot positions
        positions = self._engine.portfolio.positions
        if not positions:
            return {
                "risk_hhi": 0.0,
                "risk_top1_pct": 0.0,
                "risk_top5_pct": 0.0,
            }

        equity = self._snapshots[-1].equity if self._snapshots else 1.0
        weights = []
        for pos in positions.values():
            w = abs(pos.market_value) / equity if equity > 0 else 0.0
            weights.append(w)

        weights.sort(reverse=True)

        # HHI (Herfindahl-Hirschman Index): sum of squared weights
        hhi = float(sum(w ** 2 for w in weights))

        top1 = weights[0] * 100 if weights else 0.0
        top5 = sum(weights[:5]) * 100 if weights else 0.0

        return {
            "risk_hhi": round(hhi, 4),
            "risk_top1_pct": round(top1, 1),
            "risk_top5_pct": round(top5, 1),
        }

    def _idiosyncratic_metrics(self) -> dict[str, Any]:
        """Compute idiosyncratic return characteristics."""
        if self._benchmark_returns is None:
            return {"risk_idio_vol_pct": None, "risk_idio_sharpe": None}

        strat = np.array([s.daily_return for s in self._snapshots])
        n = min(len(strat), len(self._benchmark_returns))
        if n < 10:
            return {"risk_idio_vol_pct": None, "risk_idio_sharpe": None}

        strat = strat[:n]
        bench = np.array(self._benchmark_returns[:n])

        # Beta
        bench_var = np.var(bench)
        if bench_var > 0:
            beta = float(np.cov(strat, bench)[0, 1] / bench_var)
        else:
            beta = 0.0

        # Idiosyncratic returns = strategy - beta * benchmark
        idio_returns = strat - beta * bench
        idio_vol = float(np.std(idio_returns) * np.sqrt(252))
        idio_mean = float(np.mean(idio_returns) * 252)
        idio_sharpe = idio_mean / idio_vol if idio_vol > 0 else 0.0

        return {
            "risk_idio_vol_pct": round(idio_vol * 100, 2),
            "risk_idio_sharpe": round(idio_sharpe, 3),
        }

    def exposure_time_series(self) -> list[dict[str, Any]]:
        """Return daily exposure time series for charting.

        Returns list of dicts with: date, gross_pct, net_pct, long_pct, short_pct, num_positions
        """
        result = []
        for s in self._snapshots:
            eq = s.equity if s.equity > 0 else 1.0
            result.append({
                "date": s.date,
                "gross_pct": s.gross_value / eq * 100,
                "net_pct": (s.long_value - s.short_value) / eq * 100,
                "long_pct": s.long_value / eq * 100,
                "short_pct": s.short_value / eq * 100,
                "num_positions": s.num_positions,
            })
        return result

    def _empty_summary(self) -> dict[str, Any]:
        """Return empty risk decomposition when data is insufficient."""
        return {
            "risk_daily_vol_pct": 0.0,
            "risk_ann_vol_pct": 0.0,
            "risk_downside_vol_pct": 0.0,
            "risk_vol_21d_current_pct": 0.0,
            "risk_vol_63d_current_pct": 0.0,
            "risk_vol_21d_avg_pct": 0.0,
            "risk_vol_21d_max_pct": 0.0,
            "risk_vol_21d_min_pct": 0.0,
            "risk_avg_gross_exposure_pct": 0.0,
            "risk_max_gross_exposure_pct": 0.0,
            "risk_avg_net_exposure_pct": 0.0,
            "risk_max_net_exposure_pct": 0.0,
            "risk_avg_long_exposure_pct": 0.0,
            "risk_avg_short_exposure_pct": 0.0,
            "risk_final_gross_pct": 0.0,
            "risk_final_net_pct": 0.0,
            "risk_final_long_pct": 0.0,
            "risk_final_short_pct": 0.0,
            "risk_avg_positions": 0.0,
            "risk_max_positions": 0,
            "risk_portfolio_beta": None,
            "risk_beta_pnl_pct": None,
            "risk_alpha_pnl_pct": None,
            "risk_idio_ratio_pct": None,
            "risk_hhi": 0.0,
            "risk_top1_pct": 0.0,
            "risk_top5_pct": 0.0,
            "risk_idio_vol_pct": None,
            "risk_idio_sharpe": None,
        }
