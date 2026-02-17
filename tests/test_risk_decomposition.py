"""Tests for risk decomposition module."""

import numpy as np
from unittest.mock import MagicMock
from datetime import date, timedelta
from risk.risk_decomposition import RiskDecomposition


def _mock_engine(n_days=200, n_positions=5):
    """Create a mock engine with realistic snapshots."""
    engine = MagicMock()

    snapshots = []
    equity = 1_000_000.0
    np.random.seed(42)

    start = date(2024, 1, 1)
    for i in range(n_days):
        d = start + timedelta(days=i)
        daily_ret = np.random.normal(0.0003, 0.01)
        equity *= (1 + daily_ret)

        long_val = equity * 0.5 * np.random.uniform(0.8, 1.2)
        short_val = equity * 0.3 * np.random.uniform(0.8, 1.2)

        snap = MagicMock()
        snap.date = d
        snap.equity = equity
        snap.daily_return = daily_ret
        snap.long_value = long_val
        snap.short_value = short_val
        snap.gross_value = long_val + short_val
        snap.net_value = long_val - short_val
        snap.num_positions = n_positions
        snap.num_long = 3
        snap.num_short = 2
        snapshots.append(snap)

    engine.snapshots = snapshots
    engine.trades = []

    # Mock portfolio positions
    positions = {}
    for j in range(n_positions):
        pos = MagicMock()
        pos.ticker = f"TICK{j}"
        pos.shares = 100 if j < 3 else -50
        pos.current_price = 100.0 + j * 10
        pos.market_value = pos.shares * pos.current_price
        positions[pos.ticker] = pos

    engine.portfolio = MagicMock()
    engine.portfolio.positions = positions

    return engine


class TestRiskDecomposition:

    def test_portfolio_vol_metrics(self):
        """Should compute annualized vol and rolling metrics."""
        engine = _mock_engine()
        rd = RiskDecomposition(engine)
        result = rd.compute_summary()

        assert "risk_ann_vol_pct" in result
        assert result["risk_ann_vol_pct"] > 0
        assert "risk_downside_vol_pct" in result
        assert result["risk_downside_vol_pct"] > 0

    def test_exposure_metrics(self):
        """Should compute exposure statistics."""
        engine = _mock_engine()
        rd = RiskDecomposition(engine)
        result = rd.compute_summary()

        assert "risk_avg_gross_exposure_pct" in result
        assert result["risk_avg_gross_exposure_pct"] > 0
        assert "risk_avg_net_exposure_pct" in result
        assert "risk_final_gross_pct" in result

    def test_concentration_metrics(self):
        """Should compute HHI and top-N metrics."""
        engine = _mock_engine()
        rd = RiskDecomposition(engine)
        result = rd.compute_summary()

        assert "risk_hhi" in result
        assert result["risk_hhi"] > 0
        assert "risk_top1_pct" in result
        assert "risk_top5_pct" in result

    def test_beta_decomposition_with_benchmark(self):
        """With benchmark, should compute beta and alpha P&L."""
        engine = _mock_engine()
        np.random.seed(42)
        bench = np.random.normal(0.0003, 0.01, 200).tolist()

        rd = RiskDecomposition(engine, benchmark_returns=bench)
        result = rd.compute_summary()

        assert result["risk_portfolio_beta"] is not None
        assert "risk_beta_pnl_pct" in result
        assert "risk_alpha_pnl_pct" in result
        assert "risk_idio_ratio_pct" in result

    def test_beta_decomposition_without_benchmark(self):
        """Without benchmark, beta fields should be None."""
        engine = _mock_engine()
        rd = RiskDecomposition(engine, benchmark_returns=None)
        result = rd.compute_summary()

        assert result["risk_portfolio_beta"] is None
        assert result["risk_beta_pnl_pct"] is None

    def test_idiosyncratic_metrics(self):
        """Should compute idiosyncratic vol and Sharpe."""
        engine = _mock_engine()
        np.random.seed(42)
        bench = np.random.normal(0.0003, 0.01, 200).tolist()

        rd = RiskDecomposition(engine, benchmark_returns=bench)
        result = rd.compute_summary()

        assert result["risk_idio_vol_pct"] is not None
        assert result["risk_idio_vol_pct"] > 0
        assert result["risk_idio_sharpe"] is not None

    def test_exposure_time_series(self):
        """Should return daily exposure data for charting."""
        engine = _mock_engine(n_days=50)
        rd = RiskDecomposition(engine)
        ts = rd.exposure_time_series()

        assert len(ts) == 50
        assert "gross_pct" in ts[0]
        assert "net_pct" in ts[0]
        assert "date" in ts[0]

    def test_insufficient_data(self):
        """Should return empty summary with very few snapshots."""
        engine = _mock_engine(n_days=3)
        rd = RiskDecomposition(engine)
        result = rd.compute_summary()

        assert result["risk_ann_vol_pct"] == 0.0

    def test_no_positions(self):
        """Should handle empty portfolio gracefully."""
        engine = _mock_engine()
        engine.portfolio.positions = {}

        rd = RiskDecomposition(engine)
        result = rd.compute_summary()

        assert result["risk_hhi"] == 0.0
        assert result["risk_top1_pct"] == 0.0
