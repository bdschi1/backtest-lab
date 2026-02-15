"""Yahoo Finance data provider for backtest-lab.

Free, no API key required. All data is EOD.
Does NOT provide bid/ask — the engine uses mid-price fills when
this provider is active, or estimates spread from daily range.

Matches ls-portfolio-lab's YahooProvider pattern with retry + timeout.
"""

from __future__ import annotations

import logging
import platform
import threading
from datetime import date, timedelta
from functools import wraps
from typing import Any, Callable, TypeVar

import polars as pl
import yfinance as yf

from data.provider import DataProvider

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Timeout helper — thread-based (portable)
# ---------------------------------------------------------------------------

class _YFinanceTimeout(Exception):
    """Raised when a yfinance call exceeds the timeout."""


def _timeout(seconds: int = 60) -> Callable[[F], F]:
    """Decorator that aborts a function after *seconds* using a daemon thread."""
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result: list[Any] = []
            exc: list[BaseException] = []

            def _target() -> None:
                try:
                    result.append(func(*args, **kwargs))
                except BaseException as e:
                    exc.append(e)

            t = threading.Thread(target=_target, daemon=True)
            t.start()
            t.join(timeout=seconds)
            if t.is_alive():
                raise _YFinanceTimeout(
                    f"{func.__name__} timed out after {seconds}s"
                )
            if exc:
                raise exc[0]
            return result[0]
        return wrapper  # type: ignore[return-value]
    return decorator


# ---------------------------------------------------------------------------
# Retry — simple exponential backoff (no tenacity dependency for core)
# ---------------------------------------------------------------------------

def _retry(max_attempts: int = 3, base_delay: float = 2.0):
    """Simple retry decorator with exponential backoff."""
    import time

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < max_attempts - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "%s attempt %d/%d failed: %s. Retrying in %.1fs",
                            func.__name__, attempt + 1, max_attempts, e, delay,
                        )
                        time.sleep(delay)
            raise last_exc  # type: ignore[misc]
        return wrapper  # type: ignore[return-value]
    return decorator


class YahooProvider(DataProvider):
    """Fetch market data from Yahoo Finance via yfinance."""

    @property
    def name(self) -> str:
        return "Yahoo Finance"

    @property
    def supports_bid_ask(self) -> bool:
        return False

    def fetch_daily_prices(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> pl.DataFrame:
        """Fetch daily OHLCV + adjusted close for multiple tickers.

        Returns long-format DataFrame: date, ticker, open, high, low,
        close, adj_close, volume.
        """
        empty_schema = {
            "date": pl.Date,
            "ticker": pl.Utf8,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "adj_close": pl.Float64,
            "volume": pl.Float64,
        }

        if not tickers:
            return pl.DataFrame(schema=empty_schema)

        # Buffer start for edge cases
        start_str = (start - timedelta(days=5)).isoformat()
        end_str = (end + timedelta(days=1)).isoformat()

        logger.info(
            "Fetching prices for %d tickers: %s to %s",
            len(tickers), start_str, end_str,
        )

        try:
            raw = self._download(tickers, start_str, end_str)
        except Exception:
            logger.exception("yfinance download failed after retries")
            return pl.DataFrame(schema=empty_schema)

        if raw.empty:
            logger.warning("yfinance returned empty data for %s", tickers)
            return pl.DataFrame(schema=empty_schema)

        frames = []

        if len(tickers) == 1:
            ticker = tickers[0]
            pdf = raw.reset_index()
            pdf.columns = [c.lower().replace(" ", "_") for c in pdf.columns]
            df = pl.from_pandas(pdf)
            rename_map = {c: "date" for c in df.columns if "date" in c.lower()}
            if rename_map:
                df = df.rename(rename_map)
            df = df.with_columns(pl.lit(ticker).alias("ticker"))
            frames.append(df)
        else:
            for ticker in tickers:
                try:
                    if hasattr(raw.columns, "levels"):
                        ticker_data = raw.xs(ticker, axis=1, level=1)
                    else:
                        ticker_data = raw
                    pdf = ticker_data.reset_index()
                    pdf.columns = [c.lower().replace(" ", "_") for c in pdf.columns]
                    df = pl.from_pandas(pdf)
                    rename_map = {c: "date" for c in df.columns if "date" in c.lower()}
                    if rename_map:
                        df = df.rename(rename_map)
                    df = df.with_columns(pl.lit(ticker).alias("ticker"))
                    df = df.drop_nulls(subset=["close"])
                    frames.append(df)
                except (KeyError, ValueError):
                    logger.warning("No data for ticker %s, skipping", ticker)

        if not frames:
            return pl.DataFrame(schema=empty_schema)

        result = pl.concat(frames, how="diagonal")

        # Standardize adj_close column name
        for old in ["adjclose", "adjusted_close"]:
            if old in result.columns and "adj_close" not in result.columns:
                result = result.rename({old: "adj_close"})

        # Ensure date is Date type and filter to range
        if "date" in result.columns:
            result = result.with_columns(pl.col("date").cast(pl.Date))

        result = result.filter(
            (pl.col("date") >= start) & (pl.col("date") <= end)
        )

        desired = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
        final_cols = [c for c in desired if c in result.columns]

        return result.select(final_cols).sort(["ticker", "date"])

    @staticmethod
    @_retry(max_attempts=3, base_delay=2.0)
    @_timeout(60)
    def _download(tickers: list[str], start_str: str, end_str: str):
        """Resilient yfinance download."""
        return yf.download(
            tickers=tickers,
            start=start_str,
            end=end_str,
            auto_adjust=False,
            progress=False,
            threads=True,
        )

    def fetch_ticker_info(self, ticker: str) -> dict:
        """Fetch fundamental info for a single ticker."""
        try:
            info = self._fetch_info(ticker)
        except Exception:
            logger.exception("Failed to fetch info for %s", ticker)
            return {
                "market_cap": 0.0,
                "sector": "",
                "industry": "",
                "beta": 1.0,
                "shares_outstanding": 0,
                "avg_volume": 0,
                "dividend_yield": 0.0,
                "short_pct_of_float": 0.0,
            }

        return {
            "market_cap": info.get("marketCap", 0) or 0,
            "sector": info.get("sector", "") or "",
            "industry": info.get("industry", "") or "",
            "beta": info.get("beta", 1.0) or 1.0,
            "shares_outstanding": info.get("sharesOutstanding", 0) or 0,
            "avg_volume": info.get("averageVolume", 0) or 0,
            "dividend_yield": info.get("dividendYield", 0.0) or 0.0,
            "short_pct_of_float": info.get("shortPercentOfFloat", 0.0) or 0.0,
        }

    @staticmethod
    @_retry(max_attempts=3, base_delay=2.0)
    @_timeout(30)
    def _fetch_info(ticker: str) -> dict:
        """Resilient yfinance Ticker.info."""
        return yf.Ticker(ticker).info

    def fetch_current_prices(self, tickers: list[str]) -> dict[str, float]:
        """Fetch latest closing price for a list of tickers."""
        prices: dict[str, float] = {}
        if not tickers:
            return prices

        try:
            raw = self._download_current(tickers)
        except Exception:
            logger.exception("Failed to fetch current prices")
            return prices

        if raw.empty:
            return prices

        if len(tickers) == 1:
            close_col = raw["Close"].dropna()
            if not close_col.empty:
                prices[tickers[0]] = float(close_col.iloc[-1])
        else:
            for ticker in tickers:
                try:
                    col = raw["Close"][ticker].dropna()
                    if not col.empty:
                        prices[ticker] = float(col.iloc[-1])
                except (KeyError, IndexError):
                    logger.warning("No price data for %s", ticker)

        return prices

    @staticmethod
    @_retry(max_attempts=3, base_delay=2.0)
    @_timeout(60)
    def _download_current(tickers: list[str]):
        """Resilient yfinance download for current prices."""
        return yf.download(
            tickers=tickers,
            period="5d",
            auto_adjust=False,
            progress=False,
            threads=True,
        )

    def fetch_risk_free_rate(self) -> float:
        """Fetch 13-week T-bill rate (^IRX). Falls back to 0.05."""
        try:
            irx = yf.Ticker("^IRX")
            hist = irx.history(period="5d")
            if not hist.empty:
                rate = hist["Close"].dropna().iloc[-1]
                return float(rate) / 100.0
        except Exception:
            logger.warning("Failed to fetch risk-free rate, using 0.05 default")
        return 0.05
