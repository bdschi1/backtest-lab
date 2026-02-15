"""Bridge to fund-tracker-13f — use 13F conviction signals as universe filter.

fund-tracker-13f tracks 52 hedge funds across 5 tiers and surfaces:
    - New positions (fund bought for first time)
    - Exits (fund sold entirely)
    - Adds >50% (high-conviction increases)
    - Consensus buys (3+ funds buying same name)
    - Divergences (one fund buying what another sells)

This bridge lets backtest-lab use those signals to filter
the investment universe or boost signal conviction.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_FUND_TRACKER_PATH = Path(
    "/Users/bdsm4/Library/CloudStorage/Dropbox/bds_repos/Tier_1/fund-tracker-13f"
)


def _ensure_import():
    path_str = str(_FUND_TRACKER_PATH)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def get_consensus_buys(min_funds: int = 3) -> list[str]:
    """Get tickers that N+ tracked funds are buying.

    Args:
        min_funds: Minimum number of funds buying (default 3).

    Returns:
        List of ticker symbols with consensus buying activity.
    """
    _ensure_import()
    try:
        from core.aggregator import SignalAggregator
        agg = SignalAggregator()
        signals = agg.get_consensus_buys(min_funds=min_funds)
        return [s["ticker"] for s in signals]
    except ImportError:
        logger.warning("fund-tracker-13f not available")
        return []


def get_high_conviction_adds() -> list[dict]:
    """Get positions where a tracked fund increased by >50%.

    Returns:
        List of dicts with ticker, fund_name, pct_change.
    """
    _ensure_import()
    try:
        from core.aggregator import SignalAggregator
        agg = SignalAggregator()
        return agg.get_adds_above_threshold(threshold=50.0)
    except ImportError:
        logger.warning("fund-tracker-13f not available")
        return []


def get_new_positions() -> list[dict]:
    """Get tickers that a tracked fund initiated this quarter.

    Returns:
        List of dicts with ticker, fund_name, shares, market_value.
    """
    _ensure_import()
    try:
        from core.aggregator import SignalAggregator
        agg = SignalAggregator()
        return agg.get_new_positions()
    except ImportError:
        logger.warning("fund-tracker-13f not available")
        return []
