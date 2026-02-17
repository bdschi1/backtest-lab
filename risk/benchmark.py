"""Benchmark comparison — SPY-relative performance analytics.

Computes:
    - Relative return (strategy vs benchmark)
    - Beta (CAPM regression slope)
    - Alpha (Jensen's alpha: strategy_return - rf - beta * (benchmark_return - rf))
    - Information ratio (active_return / tracking_error)
    - Tracking error (std of active returns, annualized)
    - Up capture ratio (strategy return in up-benchmark periods / benchmark return in same)
    - Down capture ratio (strategy return in down-benchmark periods / benchmark return in same)
    - Correlation to benchmark

Usage:
    from risk.benchmark import compute_benchmark_metrics
    metrics = compute_benchmark_metrics(strategy_returns, benchmark_returns, risk_free_rate=0.05)
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def fetch_benchmark_returns(
    start: date,
    end: date,
    benchmark_ticker: str = "SPY",
) -> list[float] | None:
    """Fetch daily returns for benchmark.

    Tries yfinance first, falls back to None if unavailable.
    Returns list of daily returns aligned to trading days.
    """
    try:
        from data.yahoo_provider import YahooProvider
        provider = YahooProvider()
        prices = provider.fetch_daily_prices([benchmark_ticker], start, end)

        if prices.height == 0:
            logger.warning("No benchmark data for %s", benchmark_ticker)
            return None

        closes = prices.sort("date").get_column("close").to_list()
        if len(closes) < 2:
            return None

        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        return returns

    except Exception as e:
        logger.warning("Failed to fetch benchmark %s: %s", benchmark_ticker, e)
        return None


def compute_benchmark_metrics(
    strategy_returns: list[float],
    benchmark_returns: list[float],
    risk_free_rate: float = 0.05,
) -> dict[str, Any]:
    """Compute benchmark-relative performance metrics.

    Args:
        strategy_returns: Daily strategy returns.
        benchmark_returns: Daily benchmark returns (same length).
        risk_free_rate: Annualized risk-free rate (default 5%).

    Returns:
        Dict with benchmark comparison metrics.
    """
    # Align lengths (truncate to shorter)
    n = min(len(strategy_returns), len(benchmark_returns))
    if n < 10:
        logger.warning("Not enough overlapping data for benchmark comparison (%d days)", n)
        return _empty_metrics()

    strat = np.array(strategy_returns[:n])
    bench = np.array(benchmark_returns[:n])

    (1 + risk_free_rate) ** (1/252) - 1

    # Cumulative returns
    strat_cum = (1 + strat).prod() - 1
    bench_cum = (1 + bench).prod() - 1

    # Annualized returns
    ann_factor = 252 / n
    strat_ann = (1 + strat_cum) ** ann_factor - 1
    bench_ann = (1 + bench_cum) ** ann_factor - 1

    # Active returns
    active = strat - bench

    # Tracking error (annualized std of active returns)
    tracking_error = float(np.std(active) * np.sqrt(252))

    # Information ratio
    mean_active = float(np.mean(active) * 252)  # annualized
    info_ratio = mean_active / tracking_error if tracking_error > 0 else 0.0

    # Beta (CAPM regression)
    # beta = cov(strat, bench) / var(bench)
    bench_var = np.var(bench)
    if bench_var > 0:
        beta = float(np.cov(strat, bench)[0, 1] / bench_var)
    else:
        beta = 0.0

    # Alpha (Jensen's alpha, annualized)
    # alpha = strategy_ann - rf - beta * (bench_ann - rf)
    alpha = float((strat_ann - risk_free_rate) - beta * (bench_ann - risk_free_rate))
    # Convert to percentage
    alpha_pct = alpha * 100

    # Correlation
    correlation = float(np.corrcoef(strat, bench)[0, 1]) if bench_var > 0 else 0.0

    # Up/Down capture ratios
    up_mask = bench > 0
    down_mask = bench < 0

    if np.sum(up_mask) > 0:
        up_capture = float(np.mean(strat[up_mask]) / np.mean(bench[up_mask]) * 100)
    else:
        up_capture = 0.0

    if np.sum(down_mask) > 0:
        down_capture = float(np.mean(strat[down_mask]) / np.mean(bench[down_mask]) * 100)
    else:
        down_capture = 0.0

    # R-squared
    r_squared = correlation ** 2 * 100  # as percentage

    return {
        "benchmark_ticker": "SPY",
        "benchmark_return_pct": float(bench_cum * 100),
        "benchmark_ann_return_pct": float(bench_ann * 100),
        "relative_return_pct": float((strat_cum - bench_cum) * 100),
        "beta": round(beta, 3),
        "alpha_pct": round(alpha_pct, 2),
        "information_ratio": round(info_ratio, 3),
        "tracking_error_pct": round(tracking_error * 100, 2),
        "up_capture_pct": round(up_capture, 1),
        "down_capture_pct": round(down_capture, 1),
        "benchmark_correlation": round(correlation, 3),
        "r_squared_pct": round(r_squared, 1),
    }


def _empty_metrics() -> dict[str, Any]:
    """Return empty benchmark metrics when comparison is not possible."""
    return {
        "benchmark_ticker": "SPY",
        "benchmark_return_pct": 0.0,
        "benchmark_ann_return_pct": 0.0,
        "relative_return_pct": 0.0,
        "beta": 0.0,
        "alpha_pct": 0.0,
        "information_ratio": 0.0,
        "tracking_error_pct": 0.0,
        "up_capture_pct": 0.0,
        "down_capture_pct": 0.0,
        "benchmark_correlation": 0.0,
        "r_squared_pct": 0.0,
    }
