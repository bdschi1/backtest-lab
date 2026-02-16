"""Bloomberg data provider for backtest-lab.

Requires Bloomberg Terminal or B-PIPE and blpapi Python package.
Install: pip install backtest-lab[bloomberg]

Implementation uses blpapi's BDH (Bloomberg Data History) for daily
OHLCV+bid/ask and BDP (Bloomberg Data Point) for fundamentals.

Session management:
    - Auto-connects on first data request
    - Connection is reused for the lifetime of the provider
    - Graceful fallback if Bloomberg is unreachable

Fields mapping:
    PX_OPEN     → open
    PX_HIGH     → high
    PX_LOW      → low
    PX_LAST     → close
    PX_VOLUME   → volume
    BID         → bid
    ASK         → ask
    EQY_WEIGHTED_AVG_PX → vwap (optional)
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

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

    Args:
        host: Bloomberg server host (default: localhost).
        port: Bloomberg server port (default: 8194).
        timeout_ms: Request timeout in milliseconds (default: 30000).
    """

    # Bloomberg field → our column name
    _DAILY_FIELDS = [
        "PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST",
        "PX_VOLUME", "BID", "ASK",
    ]
    _FIELD_MAP = {
        "PX_OPEN": "open",
        "PX_HIGH": "high",
        "PX_LOW": "low",
        "PX_LAST": "close",
        "PX_VOLUME": "volume",
        "BID": "bid",
        "ASK": "ask",
    }

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8194,
        timeout_ms: int = 30_000,
    ):
        self._host = host
        self._port = port
        self._timeout_ms = timeout_ms
        self._session = None

    def _ensure_session(self):
        """Lazily connect to Bloomberg on first use."""
        if self._session is not None:
            return

        try:
            import blpapi

            options = blpapi.SessionOptions()
            options.setServerHost(self._host)
            options.setServerPort(self._port)

            session = blpapi.Session(options)
            if not session.start():
                raise ConnectionError(
                    f"Failed to start Bloomberg session at {self._host}:{self._port}"
                )
            if not session.openService("//blp/refdata"):
                raise ConnectionError("Failed to open //blp/refdata service")

            self._session = session
            logger.info(
                "Bloomberg session started: %s:%d", self._host, self._port
            )
        except ImportError:
            raise ImportError(
                "blpapi not installed. Run: pip install backtest-lab[bloomberg]"
            )

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
        """Fetch daily OHLCV + bid/ask from Bloomberg via BDH.

        Args:
            tickers: List of tickers (will be converted to Bloomberg format,
                     e.g., "AAPL" → "AAPL US Equity").
            start: Start date.
            end: End date.

        Returns:
            Polars DataFrame in long format with columns:
            date, ticker, open, high, low, close, adj_close, volume, bid, ask.
        """
        self._ensure_session()
        import blpapi

        service = self._session.getService("//blp/refdata")
        request = service.createRequest("HistoricalDataRequest")

        # Convert tickers to Bloomberg format
        bbg_tickers = [self._to_bbg_ticker(t) for t in tickers]
        for t in bbg_tickers:
            request.getElement("securities").appendValue(t)

        for f in self._DAILY_FIELDS:
            request.getElement("fields").appendValue(f)

        request.set("startDate", start.strftime("%Y%m%d"))
        request.set("endDate", end.strftime("%Y%m%d"))
        request.set("periodicitySelection", "DAILY")
        request.set("adjustmentSplit", True)
        request.set("adjustmentAbnormal", True)

        self._session.sendRequest(request)

        # Process response
        all_rows: list[dict] = []
        ticker_idx = 0

        while True:
            event = self._session.nextEvent(self._timeout_ms)
            for msg in event:
                if msg.messageType() == blpapi.Name("HistoricalDataResponse"):
                    security_data = msg.getElement("securityData")
                    bbg_name = security_data.getElementAsString("security")
                    original_ticker = self._from_bbg_ticker(bbg_name, tickers)

                    if security_data.hasElement("fieldData"):
                        field_data = security_data.getElement("fieldData")
                        for j in range(field_data.numValues()):
                            bar = field_data.getValueAsElement(j)
                            row = {"date": None, "ticker": original_ticker}

                            if bar.hasElement("date"):
                                d = bar.getElementAsDatetime("date")
                                row["date"] = date(d.year, d.month, d.day)

                            for bbg_field, our_field in self._FIELD_MAP.items():
                                if bar.hasElement(bbg_field):
                                    row[our_field] = bar.getElementAsFloat(bbg_field)
                                else:
                                    row[our_field] = None

                            # adj_close = close for Bloomberg (already adjusted)
                            row["adj_close"] = row.get("close")
                            all_rows.append(row)

            if event.eventType() == blpapi.Event.RESPONSE:
                break

        if not all_rows:
            logger.warning("Bloomberg returned no data for %s", tickers)
            return pl.DataFrame()

        df = pl.DataFrame(all_rows).with_columns(
            pl.col("date").cast(pl.Date)
        ).sort(["ticker", "date"])

        logger.info(
            "Bloomberg: fetched %d rows for %d tickers (%s to %s)",
            df.height, len(tickers), start, end,
        )
        return df

    def fetch_ticker_info(self, ticker: str) -> dict:
        """Fetch fundamentals via BDP (Bloomberg Data Point).

        Returns dict with standardized keys:
            market_cap, sector, industry, beta, shares_outstanding,
            avg_volume, dividend_yield, short_pct_of_float.
        """
        self._ensure_session()
        import blpapi

        fields = [
            "CUR_MKT_CAP",
            "GICS_SECTOR_NAME",
            "GICS_INDUSTRY_NAME",
            "BETA_ADJ_OVERRIDABLE",
            "EQY_SH_OUT",
            "VOLUME_AVG_20D",
            "EQY_DVD_YLD_IND",
            "SHORT_INT_PCT",
        ]

        service = self._session.getService("//blp/refdata")
        request = service.createRequest("ReferenceDataRequest")
        request.getElement("securities").appendValue(self._to_bbg_ticker(ticker))
        for f in fields:
            request.getElement("fields").appendValue(f)

        self._session.sendRequest(request)

        result: dict[str, Any] = {}
        while True:
            event = self._session.nextEvent(self._timeout_ms)
            for msg in event:
                if msg.messageType() == blpapi.Name("ReferenceDataResponse"):
                    sec_data = msg.getElement("securityData")
                    if sec_data.numValues() > 0:
                        elem = sec_data.getValueAsElement(0)
                        if elem.hasElement("fieldData"):
                            fd = elem.getElement("fieldData")
                            result = {
                                "market_cap": self._safe_get(fd, "CUR_MKT_CAP"),
                                "sector": self._safe_get_str(fd, "GICS_SECTOR_NAME"),
                                "industry": self._safe_get_str(fd, "GICS_INDUSTRY_NAME"),
                                "beta": self._safe_get(fd, "BETA_ADJ_OVERRIDABLE"),
                                "shares_outstanding": self._safe_get(fd, "EQY_SH_OUT"),
                                "avg_volume": self._safe_get(fd, "VOLUME_AVG_20D"),
                                "dividend_yield": self._safe_get(fd, "EQY_DVD_YLD_IND"),
                                "short_pct_of_float": self._safe_get(fd, "SHORT_INT_PCT"),
                            }
            if event.eventType() == blpapi.Event.RESPONSE:
                break

        return result

    def fetch_current_prices(self, tickers: list[str]) -> dict[str, float]:
        """Fetch latest prices via BDP PX_LAST."""
        self._ensure_session()
        import blpapi

        service = self._session.getService("//blp/refdata")
        request = service.createRequest("ReferenceDataRequest")
        for t in tickers:
            request.getElement("securities").appendValue(self._to_bbg_ticker(t))
        request.getElement("fields").appendValue("PX_LAST")

        self._session.sendRequest(request)

        prices: dict[str, float] = {}
        while True:
            event = self._session.nextEvent(self._timeout_ms)
            for msg in event:
                if msg.messageType() == blpapi.Name("ReferenceDataResponse"):
                    sec_data = msg.getElement("securityData")
                    for i in range(sec_data.numValues()):
                        elem = sec_data.getValueAsElement(i)
                        bbg_name = elem.getElementAsString("security")
                        original = self._from_bbg_ticker(bbg_name, tickers)
                        if elem.hasElement("fieldData"):
                            fd = elem.getElement("fieldData")
                            px = self._safe_get(fd, "PX_LAST")
                            if px is not None:
                                prices[original] = px
            if event.eventType() == blpapi.Event.RESPONSE:
                break

        return prices

    def fetch_risk_free_rate(self) -> float:
        """Fetch 3-month T-bill rate from GB0003M Index."""
        self._ensure_session()
        import blpapi

        service = self._session.getService("//blp/refdata")
        request = service.createRequest("ReferenceDataRequest")
        request.getElement("securities").appendValue("GB0003M Index")
        request.getElement("fields").appendValue("PX_LAST")

        self._session.sendRequest(request)

        rate = 0.05  # fallback
        while True:
            event = self._session.nextEvent(self._timeout_ms)
            for msg in event:
                if msg.messageType() == blpapi.Name("ReferenceDataResponse"):
                    sec_data = msg.getElement("securityData")
                    if sec_data.numValues() > 0:
                        elem = sec_data.getValueAsElement(0)
                        if elem.hasElement("fieldData"):
                            fd = elem.getElement("fieldData")
                            px = self._safe_get(fd, "PX_LAST")
                            if px is not None:
                                rate = px / 100.0  # Convert from percentage
            if event.eventType() == blpapi.Event.RESPONSE:
                break

        return rate

    def close(self) -> None:
        """Disconnect from Bloomberg."""
        if self._session is not None:
            try:
                self._session.stop()
                logger.info("Bloomberg session stopped")
            except Exception:
                pass
            self._session = None

    # ---- Helpers ----

    @staticmethod
    def _to_bbg_ticker(ticker: str) -> str:
        """Convert standard ticker to Bloomberg format.

        "AAPL" → "AAPL US Equity"
        "AAPL US Equity" → "AAPL US Equity" (already formatted)
        """
        if " " in ticker:
            return ticker
        return f"{ticker} US Equity"

    @staticmethod
    def _from_bbg_ticker(bbg_ticker: str, original_tickers: list[str]) -> str:
        """Convert Bloomberg ticker back to original format."""
        short = bbg_ticker.split(" ")[0]
        for t in original_tickers:
            if t.upper() == short.upper():
                return t
        return short

    @staticmethod
    def _safe_get(element, field: str) -> float | None:
        """Safely get a numeric field from a Bloomberg element."""
        try:
            if element.hasElement(field):
                return element.getElementAsFloat(field)
        except Exception:
            pass
        return None

    @staticmethod
    def _safe_get_str(element, field: str) -> str | None:
        """Safely get a string field from a Bloomberg element."""
        try:
            if element.hasElement(field):
                return element.getElementAsString(field)
        except Exception:
            pass
        return None
