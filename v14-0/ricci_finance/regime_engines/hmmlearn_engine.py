from __future__ import annotations

import numpy as np

from .base import BaseRegimeEngine


class HmmlearnGaussianEngine(BaseRegimeEngine):
    name = "hmmlearn"
    supports_viterbi = True
    supports_probabilities = True

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "diag",
        min_covar: float = 1e-4,
        n_iter: int = 500,
        random_state: int = 42,
        decoding: str = "viterbi",
    ) -> None:
        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.min_covar = float(min_covar)
        self.n_iter = int(n_iter)
        self.random_state = int(random_state)
        self.decoding = decoding
        self.model = None

    def fit(self, X: np.ndarray) -> "HmmlearnGaussianEngine":
        from hmmlearn.hmm import GaussianHMM

        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            min_covar=self.min_covar,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.model.fit(np.asarray(X, dtype=float))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Engine must be fitted before predict().")
        X = np.asarray(X, dtype=float)
        if self.decoding == "posterior":
            return self.model.predict_proba(X).argmax(axis=1)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Engine must be fitted before predict_proba().")
        return self.model.predict_proba(np.asarray(X, dtype=float))

    def score(self, X: np.ndarray) -> float:
        if self.model is None:
            raise RuntimeError("Engine must be fitted before score().")
        return float(self.model.score(np.asarray(X, dtype=float)))
