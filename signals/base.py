"""Signal framework — abstract interface for alpha signals.

Every signal takes a universe of historical price data and returns
a score per ticker on [-1, +1]:
    +1 = maximum long conviction
     0 = no signal
    -1 = maximum short conviction

Signals are stateless functions of data. The engine calls
generate_signals() once per bar with all available history
up to (and including) that bar — never future data.

The look-ahead guard (bias module, coming in build 3) wraps the
data to enforce this at runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import polars as pl


class Signal(ABC):
    """Abstract signal interface.

    Subclass this and implement generate_signals() to create
    your own alpha signal.
    """

    @property
    def name(self) -> str:
        """Human-readable signal name."""
        return type(self).__name__

    @property
    def lookback_days(self) -> int:
        """Minimum trading days of history needed before first signal.

        The engine will not call generate_signals() until at least
        this many bars are available. Override in subclass.
        """
        return 20

    @abstractmethod
    def generate_signals(
        self,
        prices: pl.DataFrame,
        current_date: date,
    ) -> dict[str, float]:
        """Generate signals for all tickers in the universe.

        Args:
            prices: Historical daily prices up to and including current_date.
                    Long format: date, ticker, open, high, low, close,
                    adj_close, volume. NEVER contains future data.
            current_date: The date being evaluated.

        Returns:
            {ticker: score} where score is in [-1, +1].
            Tickers not in the dict are treated as 0 (no signal).
        """
        ...
