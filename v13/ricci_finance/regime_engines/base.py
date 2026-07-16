from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RegimeOutput:
    engine: str
    states: np.ndarray
    probabilities: np.ndarray | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRegimeEngine(ABC):
    name = "base"
    supports_viterbi = False
    supports_probabilities = False

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseRegimeEngine":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        return None

    def score(self, X: np.ndarray) -> float | None:
        return None

    def fit_predict(self, X: np.ndarray) -> RegimeOutput:
        self.fit(X)
        states = self.predict(X)
        probabilities = self.predict_proba(X)
        return RegimeOutput(
            engine=self.name,
            states=np.asarray(states, dtype=int),
            probabilities=probabilities,
            score=self.score(X),
            metadata={
                "supports_viterbi": self.supports_viterbi,
                "supports_probabilities": self.supports_probabilities,
            },
        )
