"""Tests for bias prevention module."""

from datetime import date, timedelta

import polars as pl
import pytest

from bias.lookahead_guard import LookaheadGuard
from bias.walk_forward import WalkForwardSplitter
from bias.overfit_detector import assess_overfitting
from bias.walk_forward import WalkForwardResult, WalkForwardFold


def _make_dates(n: int = 1000) -> list[date]:
    """Generate n trading dates."""
    dates = []
    d = date(2020, 1, 1)
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return dates


def _make_price_df(n_days: int = 100) -> pl.DataFrame:
    rows = []
    d = date(2024, 1, 1)
    day_count = 0
    while day_count < n_days:
        d += timedelta(days=1)
        if d.weekday() >= 5:
            continue
        day_count += 1
        rows.append({
            "date": d,
            "ticker": "AAPL",
            "close": 150.0 + day_count * 0.1,
        })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


# ---------------------------------------------------------------------------
# Look-ahead guard tests
# ---------------------------------------------------------------------------

class TestLookaheadGuard:
    def test_filters_future_data(self):
        df = _make_price_df(100)
        guard = LookaheadGuard(df)

        dates = sorted(df.get_column("date").to_list())
        mid = dates[50]
        guard.set_date(mid)
        data = guard.get_data()

        # All dates should be <= mid
        max_date = data.get_column("date").max()
        assert max_date <= mid

    def test_no_date_returns_empty(self):
        df = _make_price_df(100)
        guard = LookaheadGuard(df)
        data = guard.get_data()
        assert data.height == 0

    def test_advancing_date_gives_more_data(self):
        df = _make_price_df(100)
        guard = LookaheadGuard(df)

        dates = sorted(df.get_column("date").to_list())

        guard.set_date(dates[20])
        d1 = guard.get_data().height

        guard.set_date(dates[50])
        d2 = guard.get_data().height

        assert d2 > d1

    def test_access_log(self):
        df = _make_price_df(50)
        guard = LookaheadGuard(df)
        dates = sorted(df.get_column("date").to_list())
        guard.set_date(dates[10])
        guard.get_data()
        guard.set_date(dates[20])
        guard.get_data()

        assert len(guard.access_log) == 2

    def test_requires_date_column(self):
        df = pl.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError):
            LookaheadGuard(df)


# ---------------------------------------------------------------------------
# Walk-forward tests
# ---------------------------------------------------------------------------

class TestWalkForwardSplitter:
    def test_generates_folds(self):
        dates = _make_dates(1000)
        splitter = WalkForwardSplitter(
            train_days=252, test_days=63, anchored=True,
        )
        folds = splitter.split(dates)
        assert len(folds) >= 1

    def test_no_overlap(self):
        dates = _make_dates(1000)
        splitter = WalkForwardSplitter(
            train_days=252, test_days=63, anchored=True,
        )
        folds = splitter.split(dates)

        for fold in folds:
            # Train end should be before test start
            assert fold.train_end < fold.test_start

    def test_split_data(self):
        dates = _make_dates(800)
        rows = [{"date": d, "ticker": "X", "close": 100.0} for d in dates]
        prices = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))

        splitter = WalkForwardSplitter(
            train_days=252, test_days=63, anchored=True,
        )
        splits = splitter.split_data(prices)
        assert len(splits) >= 1

        for fold, train, test in splits:
            assert train.height > 0
            assert test.height > 0


# ---------------------------------------------------------------------------
# Overfit detector tests
# ---------------------------------------------------------------------------

class TestOverfitDetector:
    def _make_fold(self, is_sharpe, oos_sharpe):
        fold = WalkForwardFold(
            fold_index=0,
            train_start=date(2020, 1, 1),
            train_end=date(2021, 12, 31),
            test_start=date(2022, 1, 1),
            test_end=date(2022, 3, 31),
            train_days=504,
            test_days=63,
        )
        return WalkForwardResult(
            fold=fold,
            in_sample_sharpe=is_sharpe,
            out_of_sample_sharpe=oos_sharpe,
            in_sample_return=0.0,
            out_of_sample_return=0.0,
            sharpe_decay=is_sharpe - oos_sharpe,
        )

    def test_robust_strategy(self):
        results = [
            self._make_fold(1.5, 1.2),
            self._make_fold(1.4, 1.1),
            self._make_fold(1.6, 1.3),
            self._make_fold(1.3, 1.0),
        ]
        report = assess_overfitting(results)
        assert not report.is_likely_overfit
        assert report.pct_folds_positive_oos == 100.0

    def test_overfit_strategy(self):
        results = [
            self._make_fold(2.0, -0.5),
            self._make_fold(1.8, -0.3),
            self._make_fold(2.2, 0.1),
            self._make_fold(1.9, -0.2),
        ]
        report = assess_overfitting(results)
        assert report.is_likely_overfit
        assert report.sharpe_decay_pct > 50

    def test_multiple_testing_adjustment(self):
        results = [self._make_fold(1.5, 1.2)] * 4
        report_1 = assess_overfitting(results, num_strategies_tested=1)
        report_100 = assess_overfitting(results, num_strategies_tested=100)

        # More strategies tested → lower deflated Sharpe
        if report_1.deflated_sharpe and report_100.deflated_sharpe:
            assert report_100.deflated_sharpe < report_1.deflated_sharpe

    def test_empty_results(self):
        report = assess_overfitting([])
        assert report.num_folds == 0
        assert not report.is_likely_overfit
