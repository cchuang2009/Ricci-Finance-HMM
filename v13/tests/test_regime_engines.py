from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ricci_finance.regime_analysis import (
    compare_regime_engines,
    prepare_regime_matrix,
    run_regime_engine,
)
from ricci_finance.regime_engines import (
    HmmlearnGaussianEngine,
    ExplicitDurationGaussianHSMM,
    majority_consensus,
)


def feature_table(n: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    state = np.repeat([0, 1, 2], [30, 25, 35])[:n]
    return pd.DataFrame({
        "date": pd.bdate_range("2025-01-02", periods=n).astype(str),
        "avg_ricci": rng.normal(np.choose(state, [-0.2, -0.03, 0.15]), 0.04),
        "ricci_std": rng.normal(np.choose(state, [0.25, 0.18, 0.10]), 0.02),
        "ricci_min": rng.normal(np.choose(state, [-0.6, -0.35, -0.15]), 0.05),
        "negative_edge_ratio": rng.normal(np.choose(state, [0.7, 0.4, 0.15]), 0.04),
        "clusters": np.choose(state, [5, 3, 1]),
        "largest_component_ratio": rng.normal(np.choose(state, [0.4, 0.65, 0.9]), 0.03),
        "edges": np.choose(state, [12, 20, 30]),
        "density": rng.normal(np.choose(state, [0.2, 0.4, 0.7]), 0.03),
        "component_entropy": rng.normal(np.choose(state, [1.2, 0.7, 0.2]), 0.04),
        "edge_stability": rng.normal(np.choose(state, [0.35, 0.55, 0.82]), 0.03),
        "total_edge_capital_flow": np.exp(rng.normal(np.choose(state, [17, 18, 19]), 0.1)),
        "max_node_capital_share": rng.normal(np.choose(state, [0.55, 0.4, 0.25]), 0.03),
    })


def test_hmmlearn_viterbi_and_probabilities():
    pytest.importorskip("hmmlearn")
    _, X, _ = prepare_regime_matrix(feature_table())
    engine = HmmlearnGaussianEngine(n_components=3, random_state=42)
    output = engine.fit_predict(X)
    assert len(output.states) == len(X)
    assert output.probabilities is not None
    assert output.probabilities.shape == (len(X), 3)


def test_hsmm_returns_duration_aware_states():
    pytest.importorskip("hmmlearn")
    _, X, _ = prepare_regime_matrix(feature_table())
    engine = ExplicitDurationGaussianHSMM(n_components=3, min_duration=3, max_duration=30)
    output = engine.fit_predict(X)
    assert len(output.states) == len(X)
    assert output.probabilities is None


def test_optional_pomegranate_does_not_break_comparison():
    _, benchmark, _ = compare_regime_engines(
        feature_table(),
        engines=("hmmlearn", "hsmm", "pomegranate"),
        n_components=3,
    )
    assert set(benchmark["engine"]) == {"hmmlearn", "hsmm", "pomegranate"}
    assert "available" in benchmark.columns


def test_consensus():
    result = majority_consensus({
        "a": np.array([0, 0, 1, 2]),
        "b": np.array([0, 1, 1, 2]),
        "c": np.array([0, 0, 2, 2]),
    })
    assert result.states.tolist() == [0, 0, 1, 2]
    assert result.agreement[0] == 1.0


def test_run_regime_engine_table():
    pytest.importorskip("hmmlearn")
    table, output, labels, benchmark = run_regime_engine(
        feature_table(), "hmmlearn", n_components=3, decoding="posterior"
    )
    assert "regime_name" in table
    assert "state_probability" in table
    assert benchmark.available
    assert labels
