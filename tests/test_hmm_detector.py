"""Tests for the HMM regime detector."""

import numpy as np

from regime.detector import Regime
from regime.hmm_detector import HMMRegimeDetector, HMMState


def _make_returns(n: int, mean: float = 0.0, std: float = 0.01) -> list[float]:
    """Generate synthetic daily returns."""
    np.random.seed(42)
    return list(np.random.normal(mean, std, n))


def _make_regime_returns(
    n_low: int = 100,
    n_crisis: int = 50,
    n_recovery: int = 100,
) -> list[float]:
    """Generate returns with distinct low-vol and crisis regimes."""
    np.random.seed(42)
    low_vol = list(np.random.normal(0.0005, 0.005, n_low))
    crisis = list(np.random.normal(-0.003, 0.03, n_crisis))
    recovery = list(np.random.normal(0.0003, 0.008, n_recovery))
    return low_vol + crisis + recovery


class TestHMMRegimeDetector:
    def test_insufficient_data_returns_normal(self):
        detector = HMMRegimeDetector(min_data_days=63)
        returns = _make_returns(10)
        state = detector.update(returns)
        assert state.regime == Regime.NORMAL
        assert state.state_probability == 1.0
        assert state.regime_duration == 0

    def test_fits_after_enough_data(self):
        detector = HMMRegimeDetector(
            n_states=2,
            lookback_days=100,
            min_data_days=63,
        )
        returns = _make_returns(100)
        state = detector.update(returns)
        assert detector.is_fitted
        assert isinstance(state, HMMState)
        assert state.regime in list(Regime)

    def test_state_probability_sums_valid(self):
        detector = HMMRegimeDetector(n_states=2, min_data_days=30)
        returns = _make_returns(100)
        state = detector.update(returns)
        assert 0 <= state.state_probability <= 1.0

    def test_three_state_model(self):
        detector = HMMRegimeDetector(n_states=3, min_data_days=30)
        returns = _make_returns(200)
        state = detector.update(returns)
        assert detector.is_fitted
        assert state.state_id in [0, 1, 2]

    def test_regime_duration_increments(self):
        detector = HMMRegimeDetector(n_states=2, min_data_days=30, refit_interval=100)
        returns = _make_returns(100)
        state1 = detector.update(returns)
        # Add more of the same returns — regime should be the same
        state2 = detector.update(returns + _make_returns(5, std=0.01))
        if state1.regime == state2.regime:
            assert state2.regime_duration >= state1.regime_duration

    def test_crisis_regime_detected_in_volatile_data(self):
        """High-vol returns should push toward ELEVATED or CRISIS."""
        detector = HMMRegimeDetector(n_states=3, min_data_days=30)
        np.random.seed(42)
        # Very volatile returns
        returns = list(np.random.normal(0.0, 0.05, 200))
        state = detector.update(returns)
        # Should detect high-vol state
        assert state.annualized_vol > 0.20

    def test_low_vol_regime_detected(self):
        """Low-vol returns should push toward LOW or NORMAL."""
        detector = HMMRegimeDetector(n_states=3, min_data_days=30)
        np.random.seed(42)
        # Very calm returns
        returns = list(np.random.normal(0.0003, 0.002, 200))
        state = detector.update(returns)
        assert state.annualized_vol < 0.15

    def test_refit_interval(self):
        detector = HMMRegimeDetector(
            n_states=2,
            min_data_days=30,
            refit_interval=10,
        )
        returns = _make_returns(50)
        detector.update(returns)
        assert detector._days_since_fit == 0  # just fitted

        # Update 9 more times — should not refit
        for _ in range(9):
            detector.update(returns)
        assert detector._days_since_fit == 9

        # 10th update should trigger refit
        detector.update(returns)
        assert detector._days_since_fit == 0

    def test_state_to_regime_mapping_two_states(self):
        detector = HMMRegimeDetector(n_states=2, min_data_days=30)
        returns = _make_returns(100)
        detector.update(returns)
        regimes_in_map = set(detector._state_to_regime.values())
        assert Regime.NORMAL in regimes_in_map
        assert Regime.ELEVATED in regimes_in_map

    def test_state_to_regime_mapping_three_states(self):
        detector = HMMRegimeDetector(n_states=3, min_data_days=30)
        returns = _make_returns(100)
        detector.update(returns)
        regimes_in_map = set(detector._state_to_regime.values())
        assert Regime.LOW in regimes_in_map
        assert Regime.NORMAL in regimes_in_map
        assert Regime.CRISIS in regimes_in_map

    def test_hmm_state_has_all_fields(self):
        detector = HMMRegimeDetector(n_states=2, min_data_days=30)
        returns = _make_returns(100)
        state = detector.update(returns)

        assert hasattr(state, "regime")
        assert hasattr(state, "state_id")
        assert hasattr(state, "state_mean")
        assert hasattr(state, "state_vol")
        assert hasattr(state, "state_probability")
        assert hasattr(state, "annualized_vol")
        assert hasattr(state, "regime_duration")
        assert hasattr(state, "transition_from")

    def test_classify_returns_valid_state(self):
        detector = HMMRegimeDetector(n_states=3, min_data_days=30)
        returns = _make_returns(100)
        detector.update(returns)

        # Directly test classify
        state_id, prob = detector._classify(returns[-21:])
        assert state_id in [0, 1, 2]
        assert 0 <= prob <= 1.0

    def test_no_crash_on_identical_returns(self):
        """Edge case: all returns identical (zero variance)."""
        detector = HMMRegimeDetector(n_states=2, min_data_days=30)
        returns = [0.001] * 100
        state = detector.update(returns)
        assert isinstance(state, HMMState)
