"""Tests for benchmark comparison metrics."""

import numpy as np
from risk.benchmark import compute_benchmark_metrics, _empty_metrics


class TestBenchmarkMetrics:
    """Tests for compute_benchmark_metrics."""

    def test_identical_returns_zero_alpha(self):
        """If strategy = benchmark, alpha should be ~0."""
        returns = [0.01, -0.005, 0.003, 0.002, -0.001] * 10
        result = compute_benchmark_metrics(returns, returns)
        assert abs(result["alpha_pct"]) < 2.0  # small residual from rf rate adjustment
        assert abs(result["relative_return_pct"]) < 0.01
        assert result["benchmark_correlation"] > 0.99

    def test_outperforming_strategy(self):
        """Strategy outperforming benchmark should show positive alpha."""
        np.random.seed(42)
        bench = np.random.normal(0.0003, 0.01, 200).tolist()
        strat = [b + 0.001 for b in bench]  # consistent outperformance
        result = compute_benchmark_metrics(strat, bench)
        assert result["relative_return_pct"] > 0
        assert result["alpha_pct"] > 0

    def test_beta_calculation(self):
        """Beta should be close to 1 when strategy tracks benchmark."""
        np.random.seed(42)
        bench = np.random.normal(0.0005, 0.01, 200).tolist()
        strat = [b * 1.0 + np.random.normal(0, 0.001) for b in bench]
        result = compute_benchmark_metrics(strat, bench)
        assert 0.8 < result["beta"] < 1.2

    def test_high_beta_strategy(self):
        """2x levered strategy should have beta ~2."""
        np.random.seed(42)
        bench = np.random.normal(0.0003, 0.01, 200).tolist()
        strat = [b * 2.0 for b in bench]
        result = compute_benchmark_metrics(strat, bench)
        assert 1.7 < result["beta"] < 2.3

    def test_uncorrelated_strategy(self):
        """Uncorrelated strategy should have low beta and low correlation."""
        np.random.seed(42)
        bench = np.random.normal(0.0003, 0.01, 200).tolist()
        strat = np.random.normal(0.0005, 0.008, 200).tolist()
        result = compute_benchmark_metrics(strat, bench)
        assert abs(result["beta"]) < 0.5
        assert abs(result["benchmark_correlation"]) < 0.3

    def test_up_down_capture(self):
        """Test up/down capture ratios."""
        bench = [0.01, 0.02, -0.01, -0.02, 0.015, -0.005] * 5
        strat = [0.005, 0.01, -0.02, -0.03, 0.01, -0.01] * 5
        result = compute_benchmark_metrics(strat, bench)
        assert "up_capture_pct" in result
        assert "down_capture_pct" in result
        assert isinstance(result["up_capture_pct"], float)
        assert isinstance(result["down_capture_pct"], float)

    def test_insufficient_data(self):
        """Should return empty metrics with < 10 data points."""
        result = compute_benchmark_metrics([0.01] * 5, [0.01] * 5)
        assert result["beta"] == 0.0
        assert result["alpha_pct"] == 0.0

    def test_length_mismatch(self):
        """Should handle different-length inputs by truncating."""
        np.random.seed(42)
        strat = np.random.normal(0.001, 0.01, 100).tolist()
        bench = np.random.normal(0.001, 0.01, 80).tolist()
        result = compute_benchmark_metrics(strat, bench)
        assert "beta" in result
        assert result["benchmark_ticker"] == "SPY"

    def test_empty_metrics(self):
        """Empty metrics should return all expected keys."""
        result = _empty_metrics()
        assert result["benchmark_ticker"] == "SPY"
        assert result["beta"] == 0.0
        assert len(result) == 12

    def test_information_ratio(self):
        """Info ratio should be positive for consistent outperformance."""
        np.random.seed(42)
        bench = np.random.normal(0.0003, 0.01, 200).tolist()
        strat = [b + 0.002 for b in bench]
        result = compute_benchmark_metrics(strat, bench)
        assert result["information_ratio"] > 0

    def test_tracking_error(self):
        """Tracking error should be non-negative."""
        np.random.seed(42)
        bench = np.random.normal(0.0003, 0.01, 200).tolist()
        strat = [b + np.random.normal(0, 0.005) for b in bench]
        result = compute_benchmark_metrics(strat, bench)
        assert result["tracking_error_pct"] > 0
