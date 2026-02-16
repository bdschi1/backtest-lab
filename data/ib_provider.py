"""Interactive Brokers data provider for backtest-lab.

Requires TWS or IB Gateway and ib_insync Python package.
Install: pip install backtest-lab[ibkr]

Implementation uses ib_insync's async-wrapped API for:
    - reqHistoricalData: daily OHLCV + bid/ask
    - reqFundamentalData: fundamentals (ReportFinancialStatements)
    - reqMktData: real-time snapshots

Connection management:
    - Lazy connect on first data request
    - Auto-reconnect on disconnect
    - Configurable host/port/clientId for TWS vs Gateway
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

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

    Args:
        host: TWS/Gateway host (default: 127.0.0.1).
        port: TWS=7496/7497, Gateway=4001/4002 (default: 7497 = TWS paper).
        client_id: Client ID for this connection (default: 1).
        timeout: Timeout for requests in seconds (default: 60).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        timeout: int = 60,
    ):
        self._host = host
        self._port = port
        self._client_id = client_id
        self._timeout = timeout
        self._ib = None

    def _ensure_connected(self):
        """Lazily connect to IB on first use."""
        if self._ib is not None and self._ib.isConnected():
            return

        try:
            from ib_insync import IB
            self._ib = IB()
            self._ib.connect(
                self._host,
                self._port,
                clientId=self._client_id,
                timeout=self._timeout,
            )
            logger.info(
                "IB connected: %s:%d (clientId=%d)",
                self._host, self._port, self._client_id,
            )
        except ImportError:
            raise ImportError(
                "ib_insync not installed. Run: pip install backtest-lab[ibkr]"
            )
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to IB at {self._host}:{self._port} — "
                f"is TWS/Gateway running? Error: {e}"
            )

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
        """Fetch daily OHLCV from IB via reqHistoricalData.

        IB returns bars with OHLCV. Bid/ask are fetched separately
        as daily close bid/ask if available.

        Args:
            tickers: List of ticker symbols.
            start: Start date.
            end: End date.

        Returns:
            Polars DataFrame in long format.
        """
        self._ensure_connected()
        from ib_insync import Stock, util

        all_rows: list[dict] = []
        duration_days = (end - start).days + 1
        duration_str = f"{duration_days} D"

        # IB caps at 365 days per request — chunk if needed
        if duration_days > 365:
            duration_str = "365 D"
            logger.warning(
                "IB limits to 365 days per request. "
                "Fetching last 365 days from end date."
            )

        for ticker in tickers:
            contract = Stock(ticker, "SMART", "USD")
            self._ib.qualifyContracts(contract)

            try:
                bars = self._ib.reqHistoricalData(
                    contract,
                    endDateTime=end.strftime("%Y%m%d %H:%M:%S"),
                    durationStr=duration_str,
                    barSizeSetting="1 day",
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1,
                )

                for bar in bars:
                    bar_date = bar.date
                    if isinstance(bar_date, datetime):
                        bar_date = bar_date.date()
                    elif isinstance(bar_date, str):
                        bar_date = datetime.strptime(bar_date, "%Y%m%d").date()

                    if bar_date < start:
                        continue

                    all_rows.append({
                        "date": bar_date,
                        "ticker": ticker,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "adj_close": bar.close,
                        "volume": float(bar.volume),
                    })

                logger.info("IB: fetched %d bars for %s", len(bars), ticker)

            except Exception as e:
                logger.error("IB: failed to fetch %s — %s", ticker, e)
                continue

        if not all_rows:
            logger.warning("IB returned no data for %s", tickers)
            return pl.DataFrame()

        # Try to fetch bid/ask data
        try:
            all_rows = self._add_bid_ask(all_rows, tickers, start, end)
        except Exception as e:
            logger.debug("IB: bid/ask fetch failed (non-critical) — %s", e)

        df = pl.DataFrame(all_rows).with_columns(
            pl.col("date").cast(pl.Date)
        ).sort(["ticker", "date"])

        logger.info(
            "IB: fetched %d total rows for %d tickers (%s to %s)",
            df.height, len(tickers), start, end,
        )
        return df

    def _add_bid_ask(
        self,
        rows: list[dict],
        tickers: list[str],
        start: date,
        end: date,
    ) -> list[dict]:
        """Attempt to add bid/ask columns from IB BID_ASK data."""
        from ib_insync import Stock

        bid_ask_map: dict[tuple[str, date], tuple[float, float]] = {}

        for ticker in tickers:
            contract = Stock(ticker, "SMART", "USD")
            self._ib.qualifyContracts(contract)

            duration_days = (end - start).days + 1
            duration_str = f"{min(duration_days, 365)} D"

            try:
                bars = self._ib.reqHistoricalData(
                    contract,
                    endDateTime=end.strftime("%Y%m%d %H:%M:%S"),
                    durationStr=duration_str,
                    barSizeSetting="1 day",
                    whatToShow="BID_ASK",
                    useRTH=True,
                    formatDate=1,
                )

                for bar in bars:
                    bar_date = bar.date
                    if isinstance(bar_date, datetime):
                        bar_date = bar_date.date()
                    elif isinstance(bar_date, str):
                        bar_date = datetime.strptime(bar_date, "%Y%m%d").date()

                    # For BID_ASK, open=bid_open, close=bid_close,
                    # high=ask_high, low=ask_low (approximately)
                    # More precisely, we use close as representative bid/ask
                    bid_ask_map[(ticker, bar_date)] = (bar.open, bar.close)

            except Exception:
                continue

        # Add bid/ask to rows
        for row in rows:
            key = (row["ticker"], row["date"])
            if key in bid_ask_map:
                row["bid"], row["ask"] = bid_ask_map[key]

        return rows

    def fetch_ticker_info(self, ticker: str) -> dict:
        """Fetch fundamentals via IB.

        Uses contract details and fundamental data where available.
        """
        self._ensure_connected()
        from ib_insync import Stock

        contract = Stock(ticker, "SMART", "USD")
        self._ib.qualifyContracts(contract)

        details = self._ib.reqContractDetails(contract)

        result: dict[str, Any] = {
            "market_cap": None,
            "sector": None,
            "industry": None,
            "beta": None,
            "shares_outstanding": None,
            "avg_volume": None,
            "dividend_yield": None,
            "short_pct_of_float": None,
        }

        if details:
            d = details[0]
            result["sector"] = getattr(d, "category", None)
            result["industry"] = getattr(d, "subcategory", None)
            result["market_cap"] = getattr(d, "marketCap", None) or None

        # Try fundamental data (requires market data subscription)
        try:
            fundamentals = self._ib.reqFundamentalData(
                contract, "ReportSnapshot"
            )
            if fundamentals:
                # Parse XML — basic extraction
                import xml.etree.ElementTree as ET
                root = ET.fromstring(fundamentals)

                for ratio in root.iter("Ratio"):
                    field_name = ratio.get("FieldName", "")
                    value = ratio.text
                    if value and field_name:
                        try:
                            float_val = float(value)
                            if field_name == "BETA":
                                result["beta"] = float_val
                            elif field_name == "ADIVSHR":
                                result["dividend_yield"] = float_val
                            elif field_name == "TTMSHRCUR":
                                result["shares_outstanding"] = float_val * 1e6
                        except ValueError:
                            pass

        except Exception as e:
            logger.debug("IB fundamentals not available for %s: %s", ticker, e)

        return result

    def fetch_current_prices(self, tickers: list[str]) -> dict[str, float]:
        """Fetch latest prices via IB.reqMktData snapshot."""
        self._ensure_connected()
        from ib_insync import Stock

        prices: dict[str, float] = {}

        for ticker in tickers:
            contract = Stock(ticker, "SMART", "USD")
            self._ib.qualifyContracts(contract)

            try:
                snapshot = self._ib.reqMktData(contract, snapshot=True)
                self._ib.sleep(2)  # Wait for data

                if snapshot.last and snapshot.last > 0:
                    prices[ticker] = snapshot.last
                elif snapshot.close and snapshot.close > 0:
                    prices[ticker] = snapshot.close
                else:
                    logger.warning("IB: no price for %s", ticker)

                self._ib.cancelMktData(contract)

            except Exception as e:
                logger.error("IB: failed to get price for %s — %s", ticker, e)

        return prices

    def fetch_risk_free_rate(self) -> float:
        """Fetch risk-free rate. Falls back to 0.05."""
        # IB doesn't have a direct T-bill feed — use fallback
        return 0.05

    def close(self) -> None:
        """Disconnect from IB."""
        if self._ib is not None:
            try:
                self._ib.disconnect()
                logger.info("IB disconnected")
            except Exception:
                pass
            self._ib = None
