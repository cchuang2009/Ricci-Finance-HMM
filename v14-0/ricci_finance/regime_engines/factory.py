from __future__ import annotations

from .base import BaseRegimeEngine
from .hmmlearn_engine import HmmlearnGaussianEngine
from .hsmm_engine import ExplicitDurationGaussianHSMM
from .pomegranate_engine import PomegranateDenseEngine


def create_regime_engine(
    name: str,
    *,
    n_components: int = 3,
    random_state: int = 42,
    decoding: str = "viterbi",
    **kwargs,
) -> BaseRegimeEngine:
    normalized = name.strip().lower()

    if normalized in {"hmmlearn", "hmm", "gaussianhmm"}:
        return HmmlearnGaussianEngine(
            n_components=n_components,
            random_state=random_state,
            decoding=decoding,
            **kwargs,
        )

    if normalized in {"pomegranate", "densehmm"}:
        return PomegranateDenseEngine(
            n_components=n_components,
            **kwargs,
        )

    if normalized in {"hsmm", "duration", "explicit-duration"}:
        return ExplicitDurationGaussianHSMM(
            n_components=n_components,
            random_state=random_state,
            **kwargs,
        )

    raise ValueError(f"Unknown regime engine: {name}")
