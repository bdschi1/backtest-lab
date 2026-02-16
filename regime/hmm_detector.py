"""Hidden Markov Model regime detection.

Uses a Gaussian HMM fitted on rolling returns to detect 2-3
latent market states (e.g., bull / bear / crash).

Advantages over threshold-based:
    - Learns state boundaries from data (no manual thresholds)
    - Incorporates transition probabilities (regime persistence)
    - Can detect subtle shifts earlier than vol thresholds

Disadvantages:
    - Requires fitting (risk of overfitting on short histories)
    - Non-deterministic (EM initialization matters)
    - Slower than threshold-based

This implementation uses a simple Gaussian HMM with diagonal covariance.
It does NOT require hmmlearn — implements EM from scratch using numpy
to avoid adding a heavy dependency.

If you want the full hmmlearn version:
    pip install hmmlearn
    from hmmlearn import hmm
    model = hmm.GaussianHMM(n_components=3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

from regime.detector import Regime, RegimeState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HMMState:
    """State from the HMM regime detector."""

    regime: Regime
    state_id: int            # raw HMM state (0, 1, 2)
    state_mean: float        # mean return of this state
    state_vol: float         # std dev of this state
    state_probability: float  # P(current state)
    annualized_vol: float
    regime_duration: int
    transition_from: Regime | None


class HMMRegimeDetector:
    """Hidden Markov Model regime detector.

    Fits a Gaussian HMM with n_states components on rolling returns,
    then classifies the current period.

    States are automatically labeled:
        - Lowest vol state → LOW or NORMAL
        - Mid vol state → NORMAL or ELEVATED
        - Highest vol state → ELEVATED or CRISIS

    Args:
        n_states: Number of HMM states (default: 3).
        lookback_days: Fitting window in days (default: 252 = 1 year).
        refit_interval: Days between refitting (default: 21 = monthly).
        min_data_days: Minimum days needed before fitting (default: 63).
    """

    def __init__(
        self,
        n_states: int = 3,
        lookback_days: int = 252,
        refit_interval: int = 21,
        min_data_days: int = 63,
    ):
        self._n_states = n_states
        self._lookback = lookback_days
        self._refit_interval = refit_interval
        self._min_data = min_data_days

        # HMM parameters (initialized on first fit)
        self._means: np.ndarray | None = None   # (n_states,) state means
        self._stds: np.ndarray | None = None     # (n_states,) state stds
        self._trans: np.ndarray | None = None    # (n_states, n_states) transition matrix
        self._pi: np.ndarray | None = None       # (n_states,) initial state probs

        # State tracking
        self._fitted = False
        self._days_since_fit = 0
        self._current_regime = Regime.NORMAL
        self._regime_duration = 0
        self._prev_regime: Regime | None = None
        self._state_to_regime: dict[int, Regime] = {}

    @property
    def current_regime(self) -> Regime:
        return self._current_regime

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def update(self, daily_returns: list[float]) -> HMMState:
        """Update regime with latest daily returns.

        Refits the HMM periodically and classifies the current state.

        Args:
            daily_returns: List of daily returns (most recent last).

        Returns:
            HMMState with current classification.
        """
        if len(daily_returns) < self._min_data:
            return HMMState(
                regime=Regime.NORMAL,
                state_id=0,
                state_mean=0.0,
                state_vol=0.0,
                state_probability=1.0,
                annualized_vol=0.0,
                regime_duration=0,
                transition_from=None,
            )

        # Refit if needed
        self._days_since_fit += 1
        if not self._fitted or self._days_since_fit >= self._refit_interval:
            self._fit(daily_returns)

        # Classify current state using Viterbi-like approach
        window = daily_returns[-self._lookback:] if len(daily_returns) >= self._lookback else daily_returns
        recent = daily_returns[-21:]  # use last month for classification

        state_id, state_prob = self._classify(recent)

        # Map state to regime
        regime = self._state_to_regime.get(state_id, Regime.NORMAL)
        state_mean = float(self._means[state_id]) if self._means is not None else 0.0
        state_vol = float(self._stds[state_id]) if self._stds is not None else 0.0
        ann_vol = state_vol * np.sqrt(252)

        # Track regime changes
        prev = self._current_regime
        if regime != prev:
            self._prev_regime = prev
            self._regime_duration = 1
            logger.info(
                "HMM regime change: %s → %s (state=%d, vol=%.1f%%)",
                prev.value, regime.value, state_id, ann_vol * 100,
            )
        else:
            self._regime_duration += 1

        self._current_regime = regime

        return HMMState(
            regime=regime,
            state_id=state_id,
            state_mean=state_mean,
            state_vol=state_vol,
            state_probability=state_prob,
            annualized_vol=ann_vol,
            regime_duration=self._regime_duration,
            transition_from=self._prev_regime,
        )

    def _fit(self, returns: list[float]) -> None:
        """Fit the Gaussian HMM using Expectation-Maximization.

        Simple diagonal-covariance Gaussian HMM with EM.
        """
        data = np.array(returns[-self._lookback:])
        n = len(data)
        k = self._n_states

        if n < self._min_data:
            return

        # Initialize with k-means-like approach
        sorted_data = np.sort(data)
        chunk_size = n // k

        means = np.array([
            np.mean(sorted_data[i * chunk_size:(i + 1) * chunk_size])
            for i in range(k)
        ])
        stds = np.array([
            max(np.std(sorted_data[i * chunk_size:(i + 1) * chunk_size]), 1e-8)
            for i in range(k)
        ])

        # Uniform initial probs and transition matrix
        pi = np.ones(k) / k
        trans = np.ones((k, k)) / k
        # Add persistence bias — states tend to stick
        for i in range(k):
            trans[i, i] = 0.8
            others = (1.0 - 0.8) / (k - 1)
            for j in range(k):
                if j != i:
                    trans[i, j] = others

        # EM iterations
        for iteration in range(50):
            # E-step: compute responsibilities
            log_likelihood = np.zeros((n, k))
            for j in range(k):
                log_likelihood[:, j] = (
                    -0.5 * np.log(2 * np.pi * stds[j] ** 2)
                    - 0.5 * ((data - means[j]) / stds[j]) ** 2
                )

            # Forward pass (scaled)
            alpha = np.zeros((n, k))
            alpha[0] = pi * np.exp(log_likelihood[0])
            scale = np.zeros(n)
            scale[0] = np.sum(alpha[0])
            if scale[0] > 0:
                alpha[0] /= scale[0]

            for t in range(1, n):
                for j in range(k):
                    alpha[t, j] = np.sum(alpha[t - 1] * trans[:, j]) * np.exp(log_likelihood[t, j])
                scale[t] = np.sum(alpha[t])
                if scale[t] > 0:
                    alpha[t] /= scale[t]

            # Backward pass
            beta = np.zeros((n, k))
            beta[-1] = 1.0

            for t in range(n - 2, -1, -1):
                for j in range(k):
                    beta[t, j] = np.sum(
                        trans[j, :] * np.exp(log_likelihood[t + 1]) * beta[t + 1]
                    )
                if scale[t + 1] > 0:
                    beta[t] /= scale[t + 1]

            # Posterior
            gamma = alpha * beta
            gamma_sum = gamma.sum(axis=1, keepdims=True)
            gamma_sum = np.maximum(gamma_sum, 1e-10)
            gamma = gamma / gamma_sum

            # M-step
            for j in range(k):
                resp = gamma[:, j]
                total_resp = np.sum(resp)
                if total_resp > 1e-10:
                    means[j] = np.sum(resp * data) / total_resp
                    stds[j] = np.sqrt(np.sum(resp * (data - means[j]) ** 2) / total_resp)
                    stds[j] = max(stds[j], 1e-8)

            # Update transition matrix
            for i in range(k):
                for j in range(k):
                    numerator = 0.0
                    denominator = 0.0
                    for t in range(n - 1):
                        numerator += alpha[t, i] * trans[i, j] * np.exp(log_likelihood[t + 1, j]) * beta[t + 1, j]
                        denominator += gamma[t, i]
                    if denominator > 1e-10:
                        trans[i, j] = numerator / denominator
                # Normalize
                row_sum = trans[i].sum()
                if row_sum > 0:
                    trans[i] /= row_sum

            pi = gamma[0]

        # Store fitted parameters
        self._means = means
        self._stds = stds
        self._trans = trans
        self._pi = pi
        self._fitted = True
        self._days_since_fit = 0

        # Map states to regimes based on volatility
        self._map_states_to_regimes()

        logger.info(
            "HMM fit complete: %d states, means=%s, stds=%s",
            k,
            [f"{m:.4f}" for m in means],
            [f"{s:.4f}" for s in stds],
        )

    def _map_states_to_regimes(self) -> None:
        """Map HMM states to Regime enum based on volatility ordering."""
        if self._stds is None:
            return

        # Sort states by volatility (ascending)
        vol_order = np.argsort(self._stds)

        if self._n_states == 2:
            self._state_to_regime = {
                vol_order[0]: Regime.NORMAL,
                vol_order[1]: Regime.ELEVATED,
            }
        elif self._n_states == 3:
            self._state_to_regime = {
                vol_order[0]: Regime.LOW,
                vol_order[1]: Regime.NORMAL,
                vol_order[2]: Regime.CRISIS,
            }
        elif self._n_states >= 4:
            self._state_to_regime = {
                vol_order[0]: Regime.LOW,
                vol_order[1]: Regime.NORMAL,
                vol_order[2]: Regime.ELEVATED,
                vol_order[3]: Regime.CRISIS,
            }
            # Assign any remaining to CRISIS
            for i in range(4, self._n_states):
                self._state_to_regime[vol_order[i]] = Regime.CRISIS

    def _classify(self, recent_returns: list[float]) -> tuple[int, float]:
        """Classify the current regime from recent returns.

        Uses the most likely state given the recent data.

        Returns:
            (state_id, probability)
        """
        if self._means is None or self._stds is None:
            return 0, 1.0

        data = np.array(recent_returns)

        # Compute log-likelihood under each state
        log_probs = np.zeros(self._n_states)
        for j in range(self._n_states):
            log_probs[j] = np.sum(
                -0.5 * np.log(2 * np.pi * self._stds[j] ** 2)
                - 0.5 * ((data - self._means[j]) / self._stds[j]) ** 2
            )

        # Add prior from transition matrix (persistence)
        if self._pi is not None:
            log_probs += np.log(np.maximum(self._pi, 1e-10))

        # Softmax to get probabilities
        log_probs -= np.max(log_probs)  # numerical stability
        probs = np.exp(log_probs)
        probs /= np.sum(probs)

        best_state = int(np.argmax(probs))
        return best_state, float(probs[best_state])
