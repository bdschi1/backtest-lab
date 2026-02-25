"""Sharpe ratio inference — confidence intervals and autocorrelation adjustment.

Implements the standard error formula from Lo (2002) and Paleologo (2024),
upgraded with full non-normality corrections from Bailey & Lopez de Prado
(2014) via the sharpe_inference module.

The base SE formula (Gaussian case) is:

    SE(SR) = √((1 + SR²/2) / T)

The full formula (sharpe_inference.sharpe_ratio_variance) adds corrections
for skewness (γ₃), kurtosis (γ₄), and autocorrelation (ρ):

    Var(SR̂) ≈ (1/T_eff) × [1 - γ₃·SR + (γ₄-1)/4 · SR²]

where T_eff = T × (1-ρ)/(1+ρ).

The autocorrelation-adjusted Sharpe ratio:

    SR_adj = (μ̂/σ̂) × √((1-ρ) / (1+ρ))

Reference:
    Paleologo, G. (2024). The Elements of Quantitative Investing.
    Lo, A. (2002). The Statistics of Sharpe Ratios.
    Bailey, D. H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import norm, skew as scipy_skew

from bias.sharpe_inference import (
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_ratio_variance,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SharpeConfidenceInterval:
    """Sharpe ratio with confidence interval and autocorrelation adjustment.

    Attributes:
        sharpe_ratio: Annualized Sharpe ratio (naïve, unadjusted).
        standard_error: SE(SR) with non-normality corrections.
        ci_lower: Lower bound of confidence interval.
        ci_upper: Upper bound of confidence interval.
        confidence_level: e.g. 0.95 for 95% CI.
        autocorrelation: First-order autocorrelation of returns.
        sharpe_adjusted: Autocorrelation-adjusted Sharpe (if ρ ≠ 0).
        se_adjusted: SE of adjusted Sharpe.
        ci_lower_adjusted: Lower CI for adjusted Sharpe.
        ci_upper_adjusted: Upper CI for adjusted Sharpe.
        effective_observations: T_eff = T × (1-ρ)/(1+ρ).
        num_observations: Raw number of observations T.
        annualization_factor: √252 by default.
        psr: Probabilistic Sharpe Ratio — P(true SR > 0).
        min_track_record: Minimum observations needed for significance at 95%.
    """

    sharpe_ratio: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    autocorrelation: float
    sharpe_adjusted: float
    se_adjusted: float
    ci_lower_adjusted: float
    ci_upper_adjusted: float
    effective_observations: float
    num_observations: int
    annualization_factor: float
    psr: float
    min_track_record: float


def sharpe_confidence_interval(
    returns: np.ndarray,
    confidence: float = 0.95,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> SharpeConfidenceInterval:
    """Compute Sharpe ratio with confidence interval and autocorrelation adjustment.

    Uses the full non-normality-corrected variance formula from
    sharpe_inference.sharpe_ratio_variance(), which accounts for
    skewness, kurtosis, and autocorrelation.

    Args:
        returns: Array of periodic returns (daily, monthly, etc.).
        confidence: Confidence level for interval (default 0.95).
        annualize: Whether to annualize the Sharpe ratio.
        periods_per_year: Trading periods per year (252 for daily).

    Returns:
        SharpeConfidenceInterval with naïve and adjusted estimates,
        plus PSR and MinTRL.

    Raises:
        ValueError: If returns has fewer than 3 observations.

    Reference:
        Lo (2002), Paleologo (2024) FAQ 4.2, Bailey & Lopez de Prado (2014).
    """
    returns = np.asarray(returns, dtype=float)
    T = len(returns)
    if T < 3:
        raise ValueError(f"Need at least 3 observations, got {T}")

    ann_factor = math.sqrt(periods_per_year) if annualize else 1.0

    # Naïve Sharpe ratio
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    if sigma < 1e-12:
        sigma = 1e-12
    sr_periodic = mu / sigma
    sr = sr_periodic * ann_factor

    # Return distribution moments
    rho = _autocorrelation(returns)
    gamma3 = float(scipy_skew(returns, bias=False)) if T >= 8 else 0.0
    gamma4 = float(scipy_kurtosis(returns, bias=False, fisher=False)) if T >= 8 else 3.0

    # Full variance from sharpe_inference (non-normality + autocorrelation)
    var_periodic = sharpe_ratio_variance(
        sr_periodic, T, skew=gamma3, kurtosis=gamma4, autocorr=rho,
    )
    se_periodic = math.sqrt(max(var_periodic, 1e-20))
    se = se_periodic * ann_factor

    # Z-score for confidence level
    alpha = 1.0 - confidence
    z = float(norm.ppf(1.0 - alpha / 2.0))

    ci_lower = sr - z * se
    ci_upper = sr + z * se

    # Autocorrelation-adjusted Sharpe (Paleologo FAQ 4.2)
    if abs(1.0 + rho) > 1e-10:
        adjustment = math.sqrt(max(0.0, (1.0 - rho) / (1.0 + rho)))
    else:
        adjustment = 1.0

    sr_adj = sr * adjustment

    # Effective sample size
    if abs(1.0 + rho) > 1e-10:
        t_eff = T * (1.0 - rho) / (1.0 + rho)
    else:
        t_eff = float(T)
    t_eff = max(t_eff, 2.0)

    # SE for adjusted Sharpe (using full variance formula on adjusted SR)
    sr_adj_periodic = sr_periodic * adjustment
    var_adj_periodic = sharpe_ratio_variance(
        sr_adj_periodic, T, skew=gamma3, kurtosis=gamma4, autocorr=rho,
    )
    se_adj_periodic = math.sqrt(max(var_adj_periodic, 1e-20))
    se_adj = se_adj_periodic * ann_factor

    ci_lower_adj = sr_adj - z * se_adj
    ci_upper_adj = sr_adj + z * se_adj

    # PSR: P(true SR > 0) using the annualized SR
    psr = probabilistic_sharpe_ratio(
        sr, sr0=0.0, t=T, skew=gamma3, kurtosis=gamma4, autocorr=rho,
    )

    # MinTRL: minimum observations for significance at 95%
    min_trl = minimum_track_record_length(
        sr, sr0=0.0, skew=gamma3, kurtosis=gamma4, autocorr=rho, alpha=0.05,
    )

    logger.info(
        "Sharpe CI: SR=%.3f [%.3f, %.3f], ρ=%.3f, SR_adj=%.3f [%.3f, %.3f], "
        "T=%d, T_eff=%.0f, PSR=%.3f, MinTRL=%.0f",
        sr, ci_lower, ci_upper, rho,
        sr_adj, ci_lower_adj, ci_upper_adj,
        T, t_eff, psr, min_trl,
    )

    return SharpeConfidenceInterval(
        sharpe_ratio=sr,
        standard_error=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence,
        autocorrelation=rho,
        sharpe_adjusted=sr_adj,
        se_adjusted=se_adj,
        ci_lower_adjusted=ci_lower_adj,
        ci_upper_adjusted=ci_upper_adj,
        effective_observations=t_eff,
        num_observations=T,
        annualization_factor=ann_factor,
        psr=psr,
        min_track_record=min_trl,
    )


def _autocorrelation(returns: np.ndarray) -> float:
    """Compute first-order autocorrelation (lag-1)."""
    T = len(returns)
    if T < 3:
        return 0.0
    mean = np.mean(returns)
    centered = returns - mean
    numerator = float(np.sum(centered[1:] * centered[:-1]))
    denominator = float(np.sum(centered ** 2))
    if abs(denominator) < 1e-15:
        return 0.0
    return numerator / denominator
