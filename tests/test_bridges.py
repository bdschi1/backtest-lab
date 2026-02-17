"""Tests for bridge modules — isolated from external repos.

These tests verify the bridge code's internal logic without requiring
the sister repos to be present. Each bridge gracefully degrades
when its target repo is not importable.
"""

import json
import tempfile
from datetime import date
from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# Committee Bridge tests
# ---------------------------------------------------------------------------

class TestCommitteeTSignal:
    def test_loads_from_json(self):
        from bridges.committee_bridge import CommitteeTSignal

        data = [
            {"date": "2024-01-15", "ticker": "AAPL", "t_signal": 0.72},
            {"date": "2024-01-15", "ticker": "TSLA", "t_signal": -0.45},
            {"date": "2024-01-16", "ticker": "AAPL", "t_signal": 0.68},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            signal = CommitteeTSignal(path)
            assert signal.name == "Committee-T-Signal"
            assert signal.lookback_days == 1
        finally:
            Path(path).unlink()

    def test_returns_signals_for_date(self):
        from bridges.committee_bridge import CommitteeTSignal

        data = [
            {"date": "2024-01-15", "ticker": "AAPL", "t_signal": 0.72},
            {"date": "2024-01-15", "ticker": "TSLA", "t_signal": -0.45},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            signal = CommitteeTSignal(path)
            dummy_prices = pl.DataFrame({"date": [date(2024, 1, 15)], "ticker": ["AAPL"]})

            result = signal.generate_signals(dummy_prices, date(2024, 1, 15))
            assert result == {"AAPL": 0.72, "TSLA": -0.45}
        finally:
            Path(path).unlink()

    def test_returns_empty_for_missing_date(self):
        from bridges.committee_bridge import CommitteeTSignal

        data = [
            {"date": "2024-01-15", "ticker": "AAPL", "t_signal": 0.72},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            signal = CommitteeTSignal(path)
            dummy_prices = pl.DataFrame({"date": [date(2024, 2, 1)], "ticker": ["AAPL"]})

            result = signal.generate_signals(dummy_prices, date(2024, 2, 1))
            assert result == {}
        finally:
            Path(path).unlink()

    def test_missing_file_no_crash(self):
        from bridges.committee_bridge import CommitteeTSignal

        signal = CommitteeTSignal("/nonexistent/path/signals.json")
        result = signal.generate_signals(pl.DataFrame(), date(2024, 1, 15))
        assert result == {}

    def test_signal_values_in_range(self):
        from bridges.committee_bridge import CommitteeTSignal

        data = [
            {"date": "2024-01-15", "ticker": "AAPL", "t_signal": 0.72},
            {"date": "2024-01-15", "ticker": "TSLA", "t_signal": -0.45},
            {"date": "2024-01-15", "ticker": "GOOG", "t_signal": 1.0},
            {"date": "2024-01-15", "ticker": "AMZN", "t_signal": -1.0},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            signal = CommitteeTSignal(path)
            result = signal.generate_signals(pl.DataFrame(), date(2024, 1, 15))
            for v in result.values():
                assert -1.0 <= v <= 1.0
        finally:
            Path(path).unlink()


# ---------------------------------------------------------------------------
# Redflag Bridge tests
# ---------------------------------------------------------------------------

class TestRedflagBridge:
    def test_run_compliance_check_graceful_fallback(self):
        """When redflag_ex1_analyst is not importable, should return PASS."""
        from bridges.redflag_bridge import run_compliance_check

        result = run_compliance_check("This is a normal research note.")
        assert isinstance(result, dict)
        assert "gate_decision" in result or "error" in result

    def test_check_document_file_graceful_fallback(self):
        """Should not crash when redflag repo is missing."""
        from bridges.redflag_bridge import check_document_file

        result = check_document_file("/nonexistent/file.txt")
        assert isinstance(result, dict)

    def test_compliance_check_returns_expected_keys(self):
        """Verify return dict has expected structure."""
        from bridges.redflag_bridge import run_compliance_check

        result = run_compliance_check("Test text")
        # Should have either the real result keys or the fallback error
        assert any(k in result for k in ["gate_decision", "error"])


# ---------------------------------------------------------------------------
# Fund Tracker Bridge tests
# ---------------------------------------------------------------------------

class TestFundTrackerBridge:
    def test_get_consensus_buys_graceful_fallback(self):
        """Should return empty list when fund-tracker-13f is not importable."""
        from bridges.fund_tracker_bridge import get_consensus_buys

        result = get_consensus_buys(min_funds=3)
        assert isinstance(result, list)

    def test_get_high_conviction_adds_graceful_fallback(self):
        from bridges.fund_tracker_bridge import get_high_conviction_adds

        result = get_high_conviction_adds()
        assert isinstance(result, list)

    def test_get_new_positions_graceful_fallback(self):
        from bridges.fund_tracker_bridge import get_new_positions

        result = get_new_positions()
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Portfolio Lab Bridge tests
# ---------------------------------------------------------------------------

class TestPortfolioLabBridge:
    def test_compute_factor_exposure_graceful_fallback(self):
        """Should return error dict when ls-portfolio-lab is not importable."""
        from bridges.portfolio_lab_bridge import compute_factor_exposure

        import numpy as np
        returns = list(np.random.normal(0.001, 0.01, 252))
        result = compute_factor_exposure(returns, "FF3")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# KB Risk Bridge tests
# ---------------------------------------------------------------------------

class TestKBRiskBridge:
    def test_chunk_risk_document_graceful_fallback(self):
        """Should return a list (possibly empty or with error status) for nonexistent file."""
        from bridges.kb_risk_bridge import chunk_risk_document

        result = chunk_risk_document("/nonexistent/file.pdf")
        assert isinstance(result, list)

    def test_format_risk_kb_graceful_fallback(self):
        from bridges.kb_risk_bridge import format_risk_kb

        result = format_risk_kb([], max_tokens=1000)
        assert isinstance(result, str)

    def test_build_risk_kb_from_directory_nonexistent(self):
        from bridges.kb_risk_bridge import build_risk_kb_from_directory

        result = build_risk_kb_from_directory("/nonexistent/dir")
        assert isinstance(result, list)
        assert len(result) == 0
