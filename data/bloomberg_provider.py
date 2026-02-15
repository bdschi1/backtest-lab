"""Bloomberg data provider stub for backtest-lab.

Requires Bloomberg Terminal or B-PIPE and blpapi Python package.
Install: pip install backtest-lab[bloomberg]

This is a production-ready skeleton. A Point72-class shop would fill in
the blpapi session logic for BDP/BDH/BDS calls.
"""

from __future__ import annotations

import logging
from datetime import date

import polars as pl

from data.provider import DataProvider

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Check if blpapi is importable and a session can be started."""
    try:
        import blpapi  # noqa: F401
        return True
    except ImportError:
        return False


class BloombergProvider(DataProvider):
    """Fetch market data from Bloomberg Terminal via blpapi.

    Provides real bid/ask data — enables realistic spread-aware fills.
    """

    def __init__(self, host: str = "localhost", port: int = 8194):
        self._host = host
        self._port = port
        self._session = None
        # In production: self._session = self._start_session()

    @property
    def name(self) -> str:
        return "Bloomberg"

    @property
    def supports_bid_ask(self) -> bool:
        return True

    def fetch_daily_prices(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> pl.DataFrame:
        """Fetch daily OHLCV + bid/ask from Bloomberg.

        Uses BDH (Bloomberg Data History) request.
        Fields: PX_OPEN, PX_HIGH, PX_LOW, PX_LAST, PX_VOLUME,
                EQY_WEIGHTED_AVG_PX (VWAP), BID, ASK.
        """
        raise NotImplementedError(
            "Bloomberg provider requires blpapi session configuration. "
            "See Bloomberg DAPI documentation for BDH request setup. "
            "Fields needed: PX_OPEN, PX_HIGH, PX_LOW, PX_LAST, PX_VOLUME, BID, ASK"
        )

    def fetch_ticker_info(self, ticker: str) -> dict:
        """Fetch fundamentals via BDP (Bloomberg Data Point).

        Fields: CUR_MKT_CAP, GICS_SECTOR_NAME, GICS_INDUSTRY_NAME,
                BETA_ADJ_OVERRIDABLE, EQY_SH_OUT, VOLUME_AVG_20D,
                EQY_DVD_YLD_IND, SHORT_INT_PCT.
        """
        raise NotImplementedError(
            "Bloomberg provider requires blpapi session configuration."
        )

    def fetch_current_prices(self, tickers: list[str]) -> dict[str, float]:
        """Fetch latest prices via BDP PX_LAST."""
        raise NotImplementedError(
            "Bloomberg provider requires blpapi session configuration."
        )

    def fetch_risk_free_rate(self) -> float:
        """Fetch from GB0003M Index (3-month T-bill) via BDP."""
        raise NotImplementedError(
            "Bloomberg provider requires blpapi session configuration."
        )
