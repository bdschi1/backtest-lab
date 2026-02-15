"""Overfitting detection — quantify how much of your backtest is noise.

Two methods:
    1. Sharpe Decay: Compare in-sample vs out-of-sample Sharpe ratios
       across walk-forward folds. Consistent decay = overfit.

    2. Deflated Sharpe Ratio (Harvey, Liu & Zhu, 2016): Adjusts the
       Sharpe ratio for the number of strategies tested. If you tried
       100 parameter combinations and picked the best, the "real" Sharpe
       is much lower than the observed one.

These are the tools that separate rigorous quant work from
curve-fitting disguised as research.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from bias.walk_forward import WalkForwardResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OverfitReport:
    """Overfitting assessment report."""

    is_sharpe_ratio: float        # in-sample Sharpe (average across folds)
    oos_sharpe_ratio: float       # out-of-sample Sharpe (average across folds)
    sharpe_decay_pct: float       # (IS - OOS) / IS * 100
    num_folds: int
    pct_folds_positive_oos: float # % of folds with positive OOS Sharpe
    deflated_sharpe: float | None # adjusted for multiple testing
    is_likely_overfit: bool       # True if strong evidence of overfitting
    diagnosis: str


def assess_overfitting(
    fold_results: list[WalkForwardResult],
    num_strategies_tested: int = 1,
    significance_level: float = 0.05,
) -> OverfitReport:
    """Assess overfitting from walk-forward results.

    Args:
        fold_results: Results from each walk-forward fold.
        num_strategies_tested: How many parameter combos / strategies
                               were tested before selecting this one.
                               Honest accounting is critical here.
        significance_level: For deflated Sharpe test (default 5%).

    Returns:
        OverfitReport with diagnosis.
    """
    if not fold_results:
        return OverfitReport(
            is_sharpe_ratio=0, oos_sharpe_ratio=0, sharpe_decay_pct=0,
            num_folds=0, pct_folds_positive_oos=0, deflated_sharpe=None,
            is_likely_overfit=False, diagnosis="no_data",
        )

    is_sharpes = [r.in_sample_sharpe for r in fold_results]
    oos_sharpes = [r.out_of_sample_sharpe for r in fold_results]

    avg_is = np.mean(is_sharpes)
    avg_oos = np.mean(oos_sharpes)

    # Sharpe decay
    if abs(avg_is) > 1e-10:
        decay_pct = (avg_is - avg_oos) / abs(avg_is) * 100
    else:
        decay_pct = 0.0

    # % of folds with positive OOS
    pct_positive = sum(1 for s in oos_sharpes if s > 0) / len(oos_sharpes) * 100

    # Deflated Sharpe Ratio
    deflated = _deflated_sharpe(
        observed_sharpe=avg_oos,
        num_trials=num_strategies_tested,
        num_observations=sum(r.fold.test_days for r in fold_results),
    )

    # Diagnosis
    indicators = []
    is_overfit = False

    if decay_pct > 50:
        indicators.append(f"severe Sharpe decay ({decay_pct:.0f}%)")
        is_overfit = True
    elif decay_pct > 25:
        indicators.append(f"moderate Sharpe decay ({decay_pct:.0f}%)")

    if pct_positive < 50:
        indicators.append(
            f"only {pct_positive:.0f}% of folds profitable OOS"
        )
        is_overfit = True

    if avg_oos < 0:
        indicators.append(f"negative OOS Sharpe ({avg_oos:.2f})")
        is_overfit = True

    if deflated is not None and deflated < 0.5:
        indicators.append(
            f"deflated Sharpe {deflated:.2f} (tested {num_strategies_tested} strategies)"
        )
        is_overfit = True

    if not indicators:
        diagnosis = (
            f"Appears robust: IS Sharpe={avg_is:.2f}, "
            f"OOS Sharpe={avg_oos:.2f}, "
            f"decay={decay_pct:.0f}%, "
            f"{pct_positive:.0f}% folds profitable"
        )
    else:
        diagnosis = "OVERFIT WARNING: " + "; ".join(indicators)

    return OverfitReport(
        is_sharpe_ratio=avg_is,
        oos_sharpe_ratio=avg_oos,
        sharpe_decay_pct=decay_pct,
        num_folds=len(fold_results),
        pct_folds_positive_oos=pct_positive,
        deflated_sharpe=deflated,
        is_likely_overfit=is_overfit,
        diagnosis=diagnosis,
    )


def _deflated_sharpe(
    observed_sharpe: float,
    num_trials: int,
    num_observations: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float | None:
    """Compute the Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Adjusts the observed Sharpe for:
        - Number of strategies tested (multiple testing)
        - Non-normality of returns (skew, kurtosis)
        - Sample size

    Args:
        observed_sharpe: The Sharpe ratio you observed.
        num_trials: Number of strategies / param combos tested.
        num_observations: Total number of return observations.
        skew: Skewness of returns (default 0 = symmetric).
        kurtosis: Kurtosis of returns (default 3 = normal).

    Returns:
        Deflated Sharpe Ratio, or None if inputs are invalid.
    """
    if num_trials <= 0 or num_observations <= 0:
        return None

    try:
        # Expected maximum Sharpe under null (all strategies are noise)
        # E[max(Z_1, ..., Z_N)] ≈ sqrt(2 * ln(N)) for iid standard normals
        e_max_sharpe = math.sqrt(2 * math.log(num_trials))

        # Standard error of Sharpe estimate
        se = math.sqrt(
            (1 + 0.5 * observed_sharpe ** 2
             - skew * observed_sharpe
             + (kurtosis - 3) / 4 * observed_sharpe ** 2)
            / num_observations
        )

        if se <= 0:
            return None

        # Deflated = (observed - E[max]) / SE
        deflated = (observed_sharpe - e_max_sharpe) / se

        # Convert to probability (one-sided test)
        from math import erf
        prob = 0.5 * (1 + erf(deflated / math.sqrt(2)))

        return prob

    except (ValueError, ZeroDivisionError):
        return None
