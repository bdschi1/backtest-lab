"""Rademacher complexity — uniform performance bounds for backtesting.

Implements the framework from Paleologo, "The Elements of Quantitative
Investing" (2024), Chapter 6.3: using Rademacher complexity to compute
distribution-free lower bounds on strategy performance metrics (Sharpe
ratio and Information Coefficient).

The key insight: Rademacher complexity measures how well a family of
strategies can fit random noise. If your strategy set can fit noise well,
the gap between backtest and real performance is large. The bound:

    θ ≥ θ̂ - 2R̂ - 2√(log(2/δ) / T)

where:
    θ̂  = observed metric (Sharpe, IC)
    R̂  = empirical Rademacher complexity (Monte Carlo estimate)
    δ   = confidence level (e.g., 0.05 for 95%)
    T   = number of time periods

The data-snooping penalty (2R̂) captures how many strategies you
implicitly tested. The estimation error term captures finite-sample noise.

Reference:
    Paleologo, G. (2024). The Elements of Quantitative Investing.
    Chapter 6.3: Rademacher Complexity and Backtesting.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RademacherReport:
    """Results of Rademacher complexity analysis.

    Attributes:
        observed_metric: The backtest metric value (Sharpe or IC).
        rademacher_complexity: Empirical R̂ from Monte Carlo sign flips.
        data_snooping_penalty: 2 * R̂ — penalty for strategy search.
        estimation_error: 2 * sqrt(log(2/δ) / T) — finite-sample noise.
        lower_bound: θ̂ - 2R̂ - estimation_error (Paleologo eq. 6.3).
        confidence_level: 1 - δ (e.g., 0.95).
        num_periods: T, number of time periods.
        num_simulations: Monte Carlo iterations used.
        metric_type: "sharpe" or "ic".
    """

    observed_metric: float
    rademacher_complexity: float
    data_snooping_penalty: float
    estimation_error: float
    lower_bound: float
    confidence_level: float
    num_periods: int
    num_simulations: int
    metric_type: str

    @property
    def is_significant(self) -> bool:
        """True if the lower bound is positive (strategy beats noise)."""
        return self.lower_bound > 0


def rademacher_bound(
    returns_matrix: np.ndarray,
    metric: str = "sharpe",
    delta: float = 0.05,
    n_simulations: int = 10_000,
    seed: int | None = None,
) -> RademacherReport:
    """Compute Rademacher complexity bound on strategy performance.

    Uses Monte Carlo estimation: generate random Rademacher vectors
    (±1 with equal probability), compute the supremum of ε'x / T
    across all strategies, and average across simulations.

    Args:
        returns_matrix: (T, N) array where T = time periods and
            N = candidate strategies. Each column is a strategy's
            return series.
        metric: "sharpe" for Sharpe ratio bound or "ic" for
            Information Coefficient bound.
        delta: Confidence parameter. Bound holds with probability
            1 - delta (default 0.05 → 95% confidence).
        n_simulations: Number of Monte Carlo sign-flip iterations.
        seed: Random seed for reproducibility.

    Returns:
        RademacherReport with bound and decomposition.

    Raises:
        ValueError: If returns_matrix is empty or has wrong shape.

    Reference:
        Paleologo (2024), Ch 6.3. The empirical Rademacher complexity
        is R̂ = E_ε[sup_n(ε'x^n / T)] estimated via Monte Carlo.
    """
    if returns_matrix.ndim == 1:
        returns_matrix = returns_matrix.reshape(-1, 1)

    T, N = returns_matrix.shape
    if T < 2 or N < 1:
        raise ValueError(
            f"returns_matrix must have T >= 2 periods and N >= 1 strategies, "
            f"got shape ({T}, {N})"
        )

    rng = np.random.default_rng(seed)

    # Compute observed metric for each strategy
    if metric == "sharpe":
        means = returns_matrix.mean(axis=0)
        stds = returns_matrix.std(axis=0, ddof=1)
        stds = np.where(stds > 1e-12, stds, 1e-12)
        observed_metrics = means / stds * math.sqrt(252)
        observed_best = float(np.max(observed_metrics))
    elif metric == "ic":
        # IC: cross-sectional correlation between signal and returns
        # For a single returns matrix, use mean return as the "best" metric
        means = returns_matrix.mean(axis=0)
        stds = returns_matrix.std(axis=0, ddof=1)
        stds = np.where(stds > 1e-12, stds, 1e-12)
        observed_metrics = means / stds
        observed_best = float(np.max(observed_metrics))
    else:
        raise ValueError(f"metric must be 'sharpe' or 'ic', got '{metric}'")

    # Monte Carlo Rademacher complexity estimation
    # R̂ = (1/M) * Σ_m sup_n (ε_m' * x^n / T)
    suprema = np.empty(n_simulations)
    for m in range(n_simulations):
        # Rademacher vector: ±1 with equal probability
        epsilon = rng.choice([-1.0, 1.0], size=T)

        # For each strategy n, compute ε' * x^n / T
        # Shape: (N,) — dot product of epsilon with each column
        scores = epsilon @ returns_matrix / T

        # Take supremum across strategies
        suprema[m] = np.max(scores)

    rademacher_complexity = float(np.mean(suprema))

    # Penalty decomposition (Paleologo eq. 6.3)
    data_snooping_penalty = 2.0 * rademacher_complexity
    estimation_error = 2.0 * math.sqrt(math.log(2.0 / delta) / T)

    # Lower bound: θ ≥ θ̂ - 2R̂ - 2√(log(2/δ)/T)
    lower_bound = observed_best - data_snooping_penalty - estimation_error

    logger.info(
        "Rademacher bound (%s): observed=%.4f, R̂=%.4f, "
        "penalty=%.4f, est_error=%.4f, lower_bound=%.4f",
        metric, observed_best, rademacher_complexity,
        data_snooping_penalty, estimation_error, lower_bound,
    )

    return RademacherReport(
        observed_metric=observed_best,
        rademacher_complexity=rademacher_complexity,
        data_snooping_penalty=data_snooping_penalty,
        estimation_error=estimation_error,
        lower_bound=lower_bound,
        confidence_level=1.0 - delta,
        num_periods=T,
        num_simulations=n_simulations,
        metric_type=metric,
    )


def rademacher_bound_from_returns(
    strategy_returns: list[np.ndarray],
    metric: str = "sharpe",
    delta: float = 0.05,
    n_simulations: int = 10_000,
    seed: int | None = None,
) -> RademacherReport:
    """Convenience wrapper: compute bound from a list of return arrays.

    Args:
        strategy_returns: List of 1D arrays, each a strategy's return series.
            All must have the same length T.
        metric: "sharpe" or "ic".
        delta: Confidence parameter.
        n_simulations: Monte Carlo iterations.
        seed: Random seed.

    Returns:
        RademacherReport.
    """
    if not strategy_returns:
        raise ValueError("strategy_returns must be non-empty")

    lengths = {len(r) for r in strategy_returns}
    if len(lengths) > 1:
        raise ValueError(
            f"All strategy return arrays must have the same length, "
            f"got lengths: {lengths}"
        )

    matrix = np.column_stack(strategy_returns)
    return rademacher_bound(
        matrix, metric=metric, delta=delta,
        n_simulations=n_simulations, seed=seed,
    )
