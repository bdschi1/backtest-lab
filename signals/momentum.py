"""Cross-sectional momentum signal.

Classic Jegadeesh & Titman (1993) momentum:
    - Formation period: look back N months of returns
    - Skip the most recent month (reversal effect)
    - Rank tickers cross-sectionally
    - Long the top quintile, short the bottom quintile

Signal output is continuous [-1, +1] based on z-score of
momentum returns across the universe.

This is NOT a production alpha signal — it's a well-documented,
academically grounded example to demonstrate the engine works
with real signal logic (not a toy moving-average crossover).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import polars as pl

from signals.base import Signal

logger = logging.getLogger(__name__)


class MomentumSignal(Signal):
    """Cross-sectional momentum with skip-month.

    Args:
        formation_days: Lookback period in trading days (default 252 = 12 months).
        skip_days: Recent days to skip (default 21 = 1 month).
                   Skipping avoids short-term reversal contamination.
        zscore_clip: Clip z-scores to this range (default 3.0).
    """

    def __init__(
        self,
        formation_days: int = 252,
        skip_days: int = 21,
        zscore_clip: float = 3.0,
    ):
        self._formation_days = formation_days
        self._skip_days = skip_days
        self._zscore_clip = zscore_clip

    @property
    def name(self) -> str:
        return f"Momentum({self._formation_days}d, skip={self._skip_days}d)"

    @property
    def lookback_days(self) -> int:
        return self._formation_days + self._skip_days + 5  # buffer

    def generate_signals(
        self,
        prices: pl.DataFrame,
        current_date: date,
    ) -> dict[str, float]:
        """Compute cross-sectional momentum z-scores."""
        # Filter to data up to current_date (defense against look-ahead)
        data = prices.filter(pl.col("date") <= current_date)

        if data.height == 0:
            return {}

        tickers = data.get_column("ticker").unique().to_list()
        momentum_returns: dict[str, float] = {}

        for ticker in tickers:
            ticker_data = (
                data
                .filter(pl.col("ticker") == ticker)
                .sort("date")
            )

            if ticker_data.height < self._formation_days + self._skip_days:
                continue

            closes = ticker_data.get_column("adj_close").to_numpy()

            # Skip the most recent skip_days
            end_idx = len(closes) - 1 - self._skip_days
            start_idx = end_idx - self._formation_days

            if start_idx < 0 or end_idx < 0:
                continue

            price_end = closes[end_idx]
            price_start = closes[start_idx]

            if price_start <= 0:
                continue

            ret = (price_end - price_start) / price_start
            momentum_returns[ticker] = ret

        if len(momentum_returns) < 2:
            # Need at least 2 tickers for cross-sectional ranking
            return {}

        # Cross-sectional z-score
        values = np.array(list(momentum_returns.values()))
        mean = np.mean(values)
        std = np.std(values)

        if std < 1e-10:
            return {}

        signals: dict[str, float] = {}
        for ticker, ret in momentum_returns.items():
            z = (ret - mean) / std
            # Clip and normalize to [-1, 1]
            z = np.clip(z, -self._zscore_clip, self._zscore_clip)
            signals[ticker] = float(z / self._zscore_clip)

        return signals


class MeanReversionSignal(Signal):
    """Mean-reversion signal — z-score of price relative to rolling mean.

    Short-term contrarian: buy oversold, sell overbought.
    Inverted sign from momentum — negative z-score → buy signal.

    Args:
        lookback_days: Rolling window for mean/std (default 20 = 1 month).
        zscore_clip: Clip z-scores (default 3.0).
    """

    def __init__(
        self,
        lookback_days: int = 20,
        zscore_clip: float = 3.0,
    ):
        self._lookback_days = lookback_days
        self._zscore_clip = zscore_clip

    @property
    def name(self) -> str:
        return f"MeanReversion({self._lookback_days}d)"

    @property
    def lookback_days(self) -> int:
        return self._lookback_days + 5

    def generate_signals(
        self,
        prices: pl.DataFrame,
        current_date: date,
    ) -> dict[str, float]:
        data = prices.filter(pl.col("date") <= current_date)

        if data.height == 0:
            return {}

        tickers = data.get_column("ticker").unique().to_list()
        signals: dict[str, float] = {}

        for ticker in tickers:
            ticker_data = (
                data
                .filter(pl.col("ticker") == ticker)
                .sort("date")
            )

            if ticker_data.height < self._lookback_days:
                continue

            closes = ticker_data.get_column("adj_close").to_numpy()
            window = closes[-self._lookback_days:]

            mean = np.mean(window)
            std = np.std(window)

            if std < 1e-10 or mean <= 0:
                continue

            current_price = closes[-1]
            z = (current_price - mean) / std

            # Invert: negative z (oversold) → positive signal (buy)
            z = np.clip(-z, -self._zscore_clip, self._zscore_clip)
            signals[ticker] = float(z / self._zscore_clip)

        return signals


class CompositeSignal(Signal):
    """Combine multiple signals with weights.

    Args:
        signals: List of (Signal, weight) tuples.
                 Weights are normalized to sum to 1.
    """

    def __init__(self, signals: list[tuple[Signal, float]]):
        self._signals = signals
        total_weight = sum(w for _, w in signals)
        self._weights = [(s, w / total_weight) for s, w in signals]

    @property
    def name(self) -> str:
        parts = [f"{s.name}*{w:.1f}" for s, w in self._weights]
        return f"Composite({', '.join(parts)})"

    @property
    def lookback_days(self) -> int:
        return max(s.lookback_days for s, _ in self._weights)

    def generate_signals(
        self,
        prices: pl.DataFrame,
        current_date: date,
    ) -> dict[str, float]:
        combined: dict[str, float] = {}

        for signal, weight in self._weights:
            scores = signal.generate_signals(prices, current_date)
            for ticker, score in scores.items():
                combined[ticker] = combined.get(ticker, 0.0) + score * weight

        # Clip combined signals to [-1, 1]
        return {t: max(-1.0, min(1.0, s)) for t, s in combined.items()}
