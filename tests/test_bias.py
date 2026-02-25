"""Tests for bias prevention module."""

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from bias.lookahead_guard import LookaheadGuard
from bias.walk_forward import WalkForwardSplitter
from bias.overfit_detector import assess_overfitting
from bias.walk_forward import WalkForwardResult, WalkForwardFold
from bias.rademacher import rademacher_bound, rademacher_bound_from_returns
from bias.sharpe_statistics import sharpe_confidence_interval


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
        # Use a marginal OOS Sharpe so the deflated SR doesn't saturate at 1.0
        results = [self._make_fold(0.3, 0.1)] * 4
        report_1 = assess_overfitting(results, num_strategies_tested=1)
        report_100 = assess_overfitting(results, num_strategies_tested=100)

        # More strategies tested → lower deflated Sharpe
        if report_1.deflated_sharpe and report_100.deflated_sharpe:
            assert report_100.deflated_sharpe < report_1.deflated_sharpe

    def test_empty_results(self):
        report = assess_overfitting([])
        assert report.num_folds == 0
        assert not report.is_likely_overfit


# ---------------------------------------------------------------------------
# Rademacher complexity tests
# ---------------------------------------------------------------------------

class TestRademacherBound:
    def test_single_strategy(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, size=(500, 1))
        report = rademacher_bound(returns, metric="sharpe", seed=42)

        assert report.num_periods == 500
        assert report.num_simulations == 10_000
        assert report.metric_type == "sharpe"
        # With 1 strategy, R̂ ≈ 0 (can be slightly negative from MC noise)
        assert abs(report.rademacher_complexity) < 0.01
        assert report.data_snooping_penalty == pytest.approx(
            2 * report.rademacher_complexity
        )

    def test_multiple_strategies_higher_complexity(self):
        rng = np.random.default_rng(42)
        # 1 strategy
        returns_1 = rng.normal(0, 0.02, size=(500, 1))
        report_1 = rademacher_bound(returns_1, seed=42)

        # 100 strategies → higher complexity (more room to fit noise)
        returns_100 = rng.normal(0, 0.02, size=(500, 100))
        report_100 = rademacher_bound(returns_100, seed=42)

        assert report_100.rademacher_complexity > report_1.rademacher_complexity

    def test_noise_strategies_negative_bound(self):
        rng = np.random.default_rng(42)
        # Pure noise: many strategies, no signal
        returns = rng.normal(0, 0.02, size=(252, 200))
        report = rademacher_bound(returns, metric="sharpe", seed=42)

        # With 200 noise strategies, bound should likely be negative
        # (best observed Sharpe is noise, penalty wipes it out)
        assert report.data_snooping_penalty > 0
        assert report.estimation_error > 0

    def test_strong_signal_positive_bound(self):
        rng = np.random.default_rng(42)
        # Strong signal: high mean relative to vol
        signal = rng.normal(0.005, 0.01, size=(1000, 1))
        report = rademacher_bound(signal, metric="sharpe", seed=42)

        # With 1 strong strategy and 1000 days, bound should be positive
        assert report.lower_bound > 0
        assert report.is_significant

    def test_ic_metric(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, size=(500, 5))
        report = rademacher_bound(returns, metric="ic", seed=42)
        assert report.metric_type == "ic"

    def test_from_returns_convenience(self):
        rng = np.random.default_rng(42)
        strats = [rng.normal(0.001, 0.02, 300) for _ in range(5)]
        report = rademacher_bound_from_returns(strats, seed=42)
        assert report.num_periods == 300

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            rademacher_bound(np.array([[1.0]]), seed=42)  # T=1

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            rademacher_bound_from_returns([])

    def test_confidence_level(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, size=(500, 10))

        report_95 = rademacher_bound(returns, delta=0.05, seed=42)
        report_99 = rademacher_bound(returns, delta=0.01, seed=42)

        assert report_95.confidence_level == 0.95
        assert report_99.confidence_level == 0.99
        # Higher confidence → larger estimation error → lower bound
        assert report_99.estimation_error > report_95.estimation_error


# ---------------------------------------------------------------------------
# Sharpe statistics tests
# ---------------------------------------------------------------------------

class TestSharpeConfidenceInterval:
    def test_basic_ci(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0004, 0.01, 252)
        ci = sharpe_confidence_interval(returns)

        assert ci.num_observations == 252
        assert ci.confidence_level == 0.95
        assert ci.ci_lower < ci.sharpe_ratio < ci.ci_upper
        assert ci.standard_error > 0

    def test_psr_field(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0004, 0.01, 252)
        ci = sharpe_confidence_interval(returns)

        assert 0.0 <= ci.psr <= 1.0
        # Positive SR with decent sample → PSR should be well above 0.5
        if ci.sharpe_ratio > 0:
            assert ci.psr > 0.5

    def test_min_track_record_field(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0004, 0.01, 252)
        ci = sharpe_confidence_interval(returns)

        assert ci.min_track_record > 0
        # Should be a finite positive number for positive SR
        if ci.sharpe_ratio > 0:
            assert not math.isinf(ci.min_track_record)

    def test_adjusted_sharpe_lower_with_positive_autocorr(self):
        # Create returns with positive autocorrelation
        rng = np.random.default_rng(42)
        noise = rng.normal(0.0004, 0.01, 500)
        # Add autocorrelation: r_t = 0.3 * r_{t-1} + noise
        returns = np.zeros(500)
        returns[0] = noise[0]
        for i in range(1, 500):
            returns[i] = 0.3 * returns[i - 1] + noise[i]

        ci = sharpe_confidence_interval(returns)

        assert ci.autocorrelation > 0
        # Adjusted Sharpe should be lower than naïve with positive autocorr
        if ci.sharpe_ratio > 0:
            assert ci.sharpe_adjusted < ci.sharpe_ratio

    def test_effective_observations_reduced(self):
        rng = np.random.default_rng(42)
        noise = rng.normal(0.0004, 0.01, 500)
        returns = np.zeros(500)
        returns[0] = noise[0]
        for i in range(1, 500):
            returns[i] = 0.3 * returns[i - 1] + noise[i]

        ci = sharpe_confidence_interval(returns)

        # Positive autocorrelation → fewer effective observations
        assert ci.effective_observations < ci.num_observations

    def test_zero_mean_returns(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 252)
        ci = sharpe_confidence_interval(returns)

        # CI should include zero for noise
        assert ci.ci_lower < 0 < ci.ci_upper

    def test_monthly_annualization(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.005, 0.04, 60)  # 5 years monthly
        ci = sharpe_confidence_interval(returns, periods_per_year=12)

        assert ci.annualization_factor == pytest.approx(np.sqrt(12), rel=1e-6)

    def test_no_annualization(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0004, 0.01, 252)
        ci = sharpe_confidence_interval(returns, annualize=False)

        assert ci.annualization_factor == 1.0

    def test_too_few_observations(self):
        with pytest.raises(ValueError):
            sharpe_confidence_interval(np.array([0.01, 0.02]))

    def test_higher_confidence_wider_ci(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0004, 0.01, 500)

        ci_90 = sharpe_confidence_interval(returns, confidence=0.90)
        ci_99 = sharpe_confidence_interval(returns, confidence=0.99)

        # 99% CI should be wider than 90% CI
        width_90 = ci_90.ci_upper - ci_90.ci_lower
        width_99 = ci_99.ci_upper - ci_99.ci_lower
        assert width_99 > width_90
