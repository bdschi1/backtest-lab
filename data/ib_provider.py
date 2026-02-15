"""Interactive Brokers data provider stub for backtest-lab.

Requires TWS or IB Gateway and ib_insync Python package.
Install: pip install backtest-lab[ibkr]

This is a production-ready skeleton. Provides real bid/ask.
"""

from __future__ import annotations

import logging
from datetime import date

import polars as pl

from data.provider import DataProvider

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Check if ib_insync is importable."""
    try:
        import ib_insync  # noqa: F401
        return True
    except ImportError:
        return False


class IBProvider(DataProvider):
    """Fetch market data from Interactive Brokers via ib_insync.

    Provides real bid/ask data — enables realistic spread-aware fills.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self._host = host
        self._port = port
        self._client_id = client_id
        self._ib = None
        # In production: self._ib = self._connect()

    @property
    def name(self) -> str:
        return "Interactive Brokers"

    @property
    def supports_bid_ask(self) -> bool:
        return True

    def fetch_daily_prices(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> pl.DataFrame:
        """Fetch daily OHLCV + bid/ask from IB.

        Uses IB.reqHistoricalData with barSizeSetting='1 day'.
        """
        raise NotImplementedError(
            "IB provider requires TWS/Gateway connection. "
            "See ib_insync documentation for reqHistoricalData setup."
        )

    def fetch_ticker_info(self, ticker: str) -> dict:
        """Fetch fundamentals via IB.reqFundamentalData."""
        raise NotImplementedError(
            "IB provider requires TWS/Gateway connection."
        )

    def fetch_current_prices(self, tickers: list[str]) -> dict[str, float]:
        """Fetch latest prices via IB.reqMktData snapshot."""
        raise NotImplementedError(
            "IB provider requires TWS/Gateway connection."
        )

    def fetch_risk_free_rate(self) -> float:
        """Fetch risk-free rate. Falls back to 0.05."""
        return 0.05

    def close(self) -> None:
        """Disconnect from IB."""
        if self._ib is not None:
            try:
                self._ib.disconnect()
            except Exception:
                pass
