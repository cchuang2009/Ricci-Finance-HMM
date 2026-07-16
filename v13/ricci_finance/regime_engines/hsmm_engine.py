from __future__ import annotations

import math
import numpy as np
from scipy.stats import multivariate_normal

from .base import BaseRegimeEngine


class ExplicitDurationGaussianHSMM(BaseRegimeEngine):
    """Duration-aware Gaussian semi-Markov decoder.

    Emissions and transitions are initialized from a GaussianHMM. State-duration
    distributions are estimated from decoded runs, then the sequence is decoded
    with an explicit-duration dynamic program.
    """

    name = "hsmm"
    supports_viterbi = True
    supports_probabilities = False

    def __init__(
        self,
        n_components: int = 3,
        min_duration: int = 3,
        max_duration: int = 60,
        random_state: int = 42,
    ) -> None:
        self.n_components = int(n_components)
        self.min_duration = int(min_duration)
        self.max_duration = int(max_duration)
        self.random_state = int(random_state)

    def fit(self, X: np.ndarray) -> "ExplicitDurationGaussianHSMM":
        from hmmlearn.hmm import GaussianHMM

        X = np.asarray(X, dtype=float)
        base = GaussianHMM(
            n_components=self.n_components,
            covariance_type="diag",
            min_covar=1e-4,
            n_iter=500,
            random_state=self.random_state,
        )
        base.fit(X)
        states = base.predict(X)

        self.means_ = np.asarray(base.means_, dtype=float)
        self.covars_ = np.asarray(base.covars_, dtype=float)
        self.startprob_ = np.clip(base.startprob_, 1e-12, 1.0)
        self.transmat_ = np.clip(base.transmat_, 1e-12, 1.0)

        durations: dict[int, list[int]] = {s: [] for s in range(self.n_components)}
        start = 0
        while start < len(states):
            end = start + 1
            while end < len(states) and states[end] == states[start]:
                end += 1
            durations[int(states[start])].append(end - start)
            start = end

        self.duration_mean_ = np.asarray([
            max(
                self.min_duration,
                float(np.mean(durations[s])) if durations[s] else 8.0,
            )
            for s in range(self.n_components)
        ])
        return self

    def _emission_logprob(self, X: np.ndarray) -> np.ndarray:
        result = np.empty((len(X), self.n_components), dtype=float)
        for state in range(self.n_components):
            covariance = np.diag(np.maximum(self.covars_[state], 1e-5))
            result[:, state] = multivariate_normal.logpdf(
                X,
                mean=self.means_[state],
                cov=covariance,
                allow_singular=True,
            )
        return result

    def _duration_logprob(self, state: int, duration: int) -> float:
        if duration < self.min_duration or duration > self.max_duration:
            return -np.inf
        shifted_mean = max(self.duration_mean_[state] - self.min_duration, 1e-6)
        k = duration - self.min_duration
        return k * math.log(shifted_mean) - shifted_mean - math.lgamma(k + 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        total_steps = len(X)
        emission = self._emission_logprob(X)
        cumulative = np.vstack([
            np.zeros(self.n_components),
            np.cumsum(emission, axis=0),
        ])

        score = np.full((total_steps + 1, self.n_components), -np.inf)
        back_state = np.full((total_steps + 1, self.n_components), -1, dtype=int)
        back_duration = np.zeros((total_steps + 1, self.n_components), dtype=int)
        log_start = np.log(self.startprob_)
        log_trans = np.log(self.transmat_)

        for end in range(1, total_steps + 1):
            maximum_duration = min(self.max_duration, end)
            for state in range(self.n_components):
                for duration in range(self.min_duration, maximum_duration + 1):
                    begin = end - duration
                    segment = cumulative[end, state] - cumulative[begin, state]
                    duration_score = self._duration_logprob(state, duration)

                    if begin == 0:
                        candidate = log_start[state] + segment + duration_score
                        previous_state = -1
                    else:
                        previous_scores = score[begin] + log_trans[:, state]
                        previous_scores[state] = -np.inf
                        previous_state = int(np.argmax(previous_scores))
                        candidate = previous_scores[previous_state] + segment + duration_score

                    if candidate > score[end, state]:
                        score[end, state] = candidate
                        back_state[end, state] = previous_state
                        back_duration[end, state] = duration

        states = np.empty(total_steps, dtype=int)
        end = total_steps
        state = int(np.argmax(score[end]))

        while end > 0:
            duration = int(back_duration[end, state])
            if duration <= 0:
                duration = min(end, self.min_duration)
            begin = max(0, end - duration)
            states[begin:end] = state
            state = int(back_state[end, state])
            end = begin
            if state < 0 and end > 0:
                state = int(np.argmax(score[end]))

        return states

    def score(self, X: np.ndarray) -> float | None:
        return None
