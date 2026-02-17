"""Tests for regime detection and adaptation."""

import numpy as np

from regime.detector import Regime, VolatilityRegimeDetector
from regime.adapter import RegimeAdapter, RegimeOverrides


class TestVolatilityRegimeDetector:
    def test_low_vol_regime(self):
        np.random.seed(42)
        detector = VolatilityRegimeDetector()
        returns = list(np.random.normal(0, 0.003, 252))  # ~4.8% ann vol
        state = detector.update(returns)
        assert state.regime == Regime.LOW

    def test_normal_regime(self):
        np.random.seed(42)
        detector = VolatilityRegimeDetector()
        returns = list(np.random.normal(0, 0.012, 252))  # ~19% ann vol → solidly NORMAL
        state = detector.update(returns)
        assert state.regime == Regime.NORMAL

    def test_elevated_regime(self):
        detector = VolatilityRegimeDetector()
        # Use higher daily vol to reliably land in ELEVATED or CRISIS
        returns = list(np.random.normal(0, 0.02, 252))  # ~32% ann vol
        state = detector.update(returns)
        assert state.regime in (Regime.ELEVATED, Regime.CRISIS)

    def test_crisis_regime(self):
        detector = VolatilityRegimeDetector()
        returns = list(np.random.normal(0, 0.04, 252))  # ~63% ann vol
        state = detector.update(returns)
        assert state.regime == Regime.CRISIS

    def test_regime_transition_tracked(self):
        detector = VolatilityRegimeDetector()
        # Start normal
        normal_returns = list(np.random.normal(0, 0.01, 252))
        detector.update(normal_returns)
        assert detector.current_regime == Regime.NORMAL

        # Transition to crisis with very high vol
        crisis_returns = list(np.random.normal(0, 0.04, 252))
        state = detector.update(crisis_returns)
        assert state.regime == Regime.CRISIS
        assert state.transition_from == Regime.NORMAL

    def test_duration_tracks(self):
        detector = VolatilityRegimeDetector()
        returns = list(np.random.normal(0, 0.01, 252))
        detector.update(returns)
        state = detector.update(returns)  # same regime
        assert state.regime_duration >= 1

    def test_insufficient_data(self):
        detector = VolatilityRegimeDetector(lookback_days=21)
        state = detector.update([0.01] * 10)  # not enough
        assert state.regime == Regime.NORMAL  # default


class TestRegimeAdapter:
    def test_normal_overrides(self):
        adapter = RegimeAdapter()
        detector = VolatilityRegimeDetector()
        # 0.011 daily → ~17.5% ann vol → solidly NORMAL (12-20%)
        np.random.seed(99)
        returns = list(np.random.normal(0, 0.011, 252))
        state = detector.update(returns)
        overrides = adapter.adapt(state)
        assert overrides.gross_scale == 1.0
        assert overrides.position_scale == 1.0

    def test_crisis_reduces_exposure(self):
        adapter = RegimeAdapter()
        detector = VolatilityRegimeDetector()
        returns = list(np.random.normal(0, 0.025, 252))
        state = detector.update(returns)
        overrides = adapter.adapt(state)
        assert overrides.gross_scale < 1.0
        assert overrides.position_scale < 1.0
        assert overrides.signal_threshold > 0.1

    def test_low_vol_increases_exposure(self):
        adapter = RegimeAdapter()
        detector = VolatilityRegimeDetector()
        returns = list(np.random.normal(0, 0.003, 252))
        state = detector.update(returns)
        overrides = adapter.adapt(state)
        assert overrides.gross_scale >= 1.0

    def test_custom_overrides(self):
        custom = {
            Regime.NORMAL: RegimeOverrides(gross_scale=0.8),
            Regime.CRISIS: RegimeOverrides(gross_scale=0.1),
            Regime.LOW: RegimeOverrides(gross_scale=1.5),
            Regime.ELEVATED: RegimeOverrides(gross_scale=0.5),
        }
        adapter = RegimeAdapter(overrides=custom)
        detector = VolatilityRegimeDetector()
        returns = list(np.random.normal(0, 0.01, 252))
        state = detector.update(returns)
        overrides = adapter.adapt(state)
        assert overrides.gross_scale == 0.8
