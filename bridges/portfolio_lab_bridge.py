"""Bridge to ls-portfolio-lab — import risk metrics and factor models.

ls-portfolio-lab has 277 tests and production-ready analytics:
    - 20+ portfolio metrics (Sharpe, Sortino, VaR, CVaR, drawdown)
    - CAPM/FF3/FF4 factor models
    - Trade impact simulator
    - SLSQP constrained optimizer

This bridge lets backtest-lab use those analytics on its results.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to ls-portfolio-lab
_PORTFOLIO_LAB_PATH = Path(
    "/Users/bdsm4/Library/CloudStorage/Dropbox/bds_repos/Tier_1/ls-portfolio-lab"
)


def _ensure_import():
    """Add ls-portfolio-lab to sys.path if needed."""
    path_str = str(_PORTFOLIO_LAB_PATH)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def get_portfolio_lab_provider(name: str = "Yahoo Finance"):
    """Get a data provider from ls-portfolio-lab's factory.

    Reuses their provider abstraction instead of duplicating it.
    """
    _ensure_import()
    from data.provider_factory import get_provider
    return get_provider(name)


def compute_factor_exposure(returns_series, factor_model: str = "FF3"):
    """Compute factor exposure using ls-portfolio-lab's factor models.

    Args:
        returns_series: Daily returns (list or numpy array).
        factor_model: "CAPM", "FF3", or "FF4".

    Returns:
        Dict with alpha, betas, r_squared.
    """
    _ensure_import()
    try:
        from core.factor_model import FactorModel
        model = FactorModel()
        return model.analyze(returns_series, model_type=factor_model)
    except ImportError:
        logger.warning("ls-portfolio-lab factor_model not available")
        return {"error": "ls-portfolio-lab not found at expected path"}
