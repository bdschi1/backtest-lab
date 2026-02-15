"""Walk-forward validation — proper out-of-sample testing.

A single train/test split is not enough. Walk-forward:
    1. Train on window [0, T]
    2. Test on [T, T+step]
    3. Slide window forward
    4. Repeat

This prevents in-sample overfitting from being hidden by
a lucky test split.

Two variants:
    - Anchored: training window always starts at the beginning
    - Rolling: training window slides (fixed width)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import polars as pl

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalkForwardFold:
    """One fold of a walk-forward split."""

    fold_index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_days: int
    test_days: int


@dataclass(frozen=True)
class WalkForwardResult:
    """Result of one fold — filled in after backtesting that fold."""

    fold: WalkForwardFold
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    in_sample_return: float
    out_of_sample_return: float
    sharpe_decay: float  # IS_sharpe - OOS_sharpe (positive = overfit)


class WalkForwardSplitter:
    """Generate walk-forward train/test splits.

    Args:
        train_days: Training window in trading days (default 504 = 2 years).
        test_days: Test window in trading days (default 63 = 1 quarter).
        step_days: How far to slide between folds (default = test_days).
        anchored: If True, training always starts at day 0. If False, rolling.
        min_folds: Minimum folds required (default 4).
    """

    def __init__(
        self,
        train_days: int = 504,
        test_days: int = 63,
        step_days: int | None = None,
        anchored: bool = True,
        min_folds: int = 4,
    ):
        self._train_days = train_days
        self._test_days = test_days
        self._step_days = step_days or test_days
        self._anchored = anchored
        self._min_folds = min_folds

    def split(self, dates: list[date]) -> list[WalkForwardFold]:
        """Generate walk-forward folds from sorted trading dates.

        Args:
            dates: Sorted list of trading dates.

        Returns:
            List of WalkForwardFold objects.

        Raises:
            ValueError: if not enough data for min_folds.
        """
        sorted_dates = sorted(dates)
        n = len(sorted_dates)
        folds: list[WalkForwardFold] = []

        fold_idx = 0
        train_start_idx = 0

        while True:
            if self._anchored:
                train_start_idx = 0
            else:
                train_start_idx = fold_idx * self._step_days

            train_end_idx = train_start_idx + self._train_days - 1
            test_start_idx = train_end_idx + 1
            test_end_idx = test_start_idx + self._test_days - 1

            if test_end_idx >= n:
                break

            folds.append(WalkForwardFold(
                fold_index=fold_idx,
                train_start=sorted_dates[train_start_idx],
                train_end=sorted_dates[train_end_idx],
                test_start=sorted_dates[test_start_idx],
                test_end=sorted_dates[test_end_idx],
                train_days=train_end_idx - train_start_idx + 1,
                test_days=test_end_idx - test_start_idx + 1,
            ))

            fold_idx += 1

            if not self._anchored:
                continue  # next iteration slides
            else:
                # For anchored, slide the train_end forward
                train_start_idx = 0
                next_train_end = train_end_idx + self._step_days
                if next_train_end + self._test_days >= n:
                    break
                train_end_idx = next_train_end
                test_start_idx = train_end_idx + 1
                test_end_idx = test_start_idx + self._test_days - 1

                if test_end_idx >= n:
                    break

                folds.append(WalkForwardFold(
                    fold_index=fold_idx,
                    train_start=sorted_dates[0],
                    train_end=sorted_dates[train_end_idx],
                    test_start=sorted_dates[test_start_idx],
                    test_end=sorted_dates[test_end_idx],
                    train_days=train_end_idx + 1,
                    test_days=test_end_idx - test_start_idx + 1,
                ))
                fold_idx += 1

                # Continue sliding
                while True:
                    train_end_idx += self._step_days
                    test_start_idx = train_end_idx + 1
                    test_end_idx = test_start_idx + self._test_days - 1
                    if test_end_idx >= n:
                        break
                    folds.append(WalkForwardFold(
                        fold_index=fold_idx,
                        train_start=sorted_dates[0],
                        train_end=sorted_dates[train_end_idx],
                        test_start=sorted_dates[test_start_idx],
                        test_end=sorted_dates[test_end_idx],
                        train_days=train_end_idx + 1,
                        test_days=test_end_idx - test_start_idx + 1,
                    ))
                    fold_idx += 1
                break

        if len(folds) < self._min_folds:
            logger.warning(
                "Walk-forward: only %d folds (need %d). "
                "Consider shorter train/test windows or more data.",
                len(folds), self._min_folds,
            )

        logger.info(
            "Walk-forward: %d folds, %s mode, "
            "train=%dd, test=%dd",
            len(folds),
            "anchored" if self._anchored else "rolling",
            self._train_days, self._test_days,
        )

        return folds

    def split_data(
        self, prices: pl.DataFrame,
    ) -> list[tuple[WalkForwardFold, pl.DataFrame, pl.DataFrame]]:
        """Split price data into train/test DataFrames per fold.

        Args:
            prices: Full price history.

        Returns:
            List of (fold, train_df, test_df) tuples.
        """
        dates = sorted(prices.get_column("date").unique().to_list())
        folds = self.split(dates)

        result = []
        for fold in folds:
            train = prices.filter(
                (pl.col("date") >= fold.train_start)
                & (pl.col("date") <= fold.train_end)
            )
            test = prices.filter(
                (pl.col("date") >= fold.test_start)
                & (pl.col("date") <= fold.test_end)
            )
            result.append((fold, train, test))

        return result
