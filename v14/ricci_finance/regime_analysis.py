from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler

from .models import FrameData
from .rolling import rolling_feature_table
from .regime_engines import create_regime_engine, RegimeOutput, majority_consensus

REGIME_FEATURES = [
    "avg_ricci", "ricci_std", "ricci_min", "negative_edge_ratio",
    "clusters", "largest_component_ratio", "edges", "density",
    "component_entropy", "edge_stability", "total_edge_capital_flow",
    "max_node_capital_share",
]


@dataclass
class EngineBenchmark:
    engine: str
    available: bool
    runtime_seconds: float
    switch_rate: float = float("nan")
    median_run_length: float = float("nan")
    mean_run_length: float = float("nan")
    short_run_ratio_le_2: float = float("nan")
    posterior_entropy: float = float("nan")
    score: float = float("nan")
    message: str = ""


def run_lengths(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=int)
    if len(states) == 0:
        return np.asarray([], dtype=int)
    boundaries = np.flatnonzero(np.r_[True, states[1:] != states[:-1], True])
    return np.diff(boundaries)


def switch_rate(states: np.ndarray) -> float:
    states = np.asarray(states, dtype=int)
    return float(np.mean(states[1:] != states[:-1])) if len(states) > 1 else 0.0


def posterior_entropy(probabilities: np.ndarray | None) -> float:
    if probabilities is None:
        return float("nan")
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    return float(np.mean(-np.sum(p * np.log(p), axis=1)))


def align_states(reference: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    reference = np.asarray(reference, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    ref_ids = np.unique(reference)
    pred_ids = np.unique(predicted)
    cost = np.zeros((len(pred_ids), len(ref_ids)), dtype=float)
    for i, p in enumerate(pred_ids):
        for j, r in enumerate(ref_ids):
            cost[i, j] = -np.sum((predicted == p) & (reference == r))
    rows, cols = linear_sum_assignment(cost)
    mapping = {int(pred_ids[i]): int(ref_ids[j]) for i, j in zip(rows, cols)}
    aligned = np.asarray([mapping.get(int(state), int(state)) for state in predicted], dtype=int)
    return aligned, mapping


def prepare_regime_matrix(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    columns = [column for column in REGIME_FEATURES if column in feature_df.columns]
    if not columns:
        raise ValueError("No supported regime features were found.")
    clean = feature_df[columns].replace([np.inf, -np.inf], np.nan)
    clean = clean.interpolate(limit_direction="both").fillna(0.0)
    scaler = StandardScaler()
    matrix = scaler.fit_transform(clean)
    return clean, matrix, scaler


def label_states(feature_df: pd.DataFrame, states: np.ndarray) -> dict[int, str]:
    temp = feature_df.copy()
    temp["state"] = states
    summary = temp.groupby("state").agg(
        avg_ricci=("avg_ricci", "mean"),
        largest_ratio=("largest_component_ratio", "mean"),
        density=("density", "mean"),
    )
    ordered = list(summary.sort_values(["avg_ricci", "largest_ratio", "density"]).index)
    names = ["stress / fragmentation", "transition / rotation", "coherent risk-on"]
    return {
        int(state): names[index] if index < len(names) else f"regime_{index}"
        for index, state in enumerate(ordered)
    }


def run_regime_engine(
    feature_df: pd.DataFrame,
    engine_name: str,
    *,
    n_components: int = 3,
    decoding: str = "viterbi",
    random_state: int = 42,
) -> tuple[pd.DataFrame, RegimeOutput, dict[int, str], EngineBenchmark]:
    clean, matrix, _ = prepare_regime_matrix(feature_df)
    started = time.perf_counter()
    try:
        engine = create_regime_engine(
            engine_name,
            n_components=n_components,
            random_state=random_state,
            decoding=decoding,
        )
        output = engine.fit_predict(matrix)
        elapsed = time.perf_counter() - started
        labels = label_states(feature_df, output.states)
        result = feature_df.copy().reset_index(drop=True)
        result["regime_state"] = output.states
        result["regime_name"] = result["regime_state"].map(labels)
        result["regime_engine"] = output.engine
        if output.probabilities is not None:
            for state in range(output.probabilities.shape[1]):
                result[f"state_probability_{state}"] = output.probabilities[:, state]
            result["state_probability"] = output.probabilities[
                np.arange(len(output.states)), output.states
            ]
        else:
            result["state_probability"] = np.nan
        lengths = run_lengths(output.states)
        benchmark = EngineBenchmark(
            engine=output.engine,
            available=True,
            runtime_seconds=elapsed,
            switch_rate=switch_rate(output.states),
            median_run_length=float(np.median(lengths)) if len(lengths) else np.nan,
            mean_run_length=float(np.mean(lengths)) if len(lengths) else np.nan,
            short_run_ratio_le_2=float(np.mean(lengths <= 2)) if len(lengths) else np.nan,
            posterior_entropy=posterior_entropy(output.probabilities),
            score=float(output.score) if output.score is not None else np.nan,
        )
        return result, output, labels, benchmark
    except Exception as exc:
        elapsed = time.perf_counter() - started
        result = feature_df.copy().reset_index(drop=True)
        result["regime_state"] = -1
        result["regime_name"] = f"{engine_name} unavailable"
        result["regime_engine"] = engine_name
        result["state_probability"] = np.nan
        output = RegimeOutput(
            engine=engine_name,
            states=np.full(len(feature_df), -1, dtype=int),
            probabilities=None,
            metadata={"error": str(exc)},
        )
        return result, output, {-1: f"{engine_name} unavailable"}, EngineBenchmark(
            engine=engine_name,
            available=False,
            runtime_seconds=elapsed,
            message=str(exc),
        )


def compare_regime_engines(
    feature_df: pd.DataFrame,
    engines: Sequence[str] = ("hmmlearn", "hsmm", "pomegranate"),
    *,
    n_components: int = 3,
    random_state: int = 42,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame | None]:
    results: dict[str, pd.DataFrame] = {}
    outputs: dict[str, RegimeOutput] = {}
    rows = []

    for engine_name in engines:
        table, output, _, benchmark = run_regime_engine(
            feature_df,
            engine_name,
            n_components=n_components,
            random_state=random_state,
        )
        results[engine_name] = table
        outputs[engine_name] = output
        rows.append(vars(benchmark))

    valid_sequences = {
        name: output.states
        for name, output in outputs.items()
        if np.all(output.states >= 0)
    }
    consensus_df = None
    if len(valid_sequences) >= 2:
        consensus = majority_consensus(valid_sequences)
        consensus_df = feature_df.copy().reset_index(drop=True)
        consensus_df["consensus_state"] = consensus.states
        consensus_df["consensus_agreement"] = consensus.agreement
        consensus_df["consensus_disagreement"] = consensus.disagreement

    return results, pd.DataFrame(rows), consensus_df


def attach_regime_to_frames(
    frames: Sequence[FrameData],
    regime_df: pd.DataFrame,
    labels: dict[int, str] | None = None,
) -> None:
    for index, frame in enumerate(frames):
        if index >= len(regime_df):
            break
        state = int(regime_df.loc[index, "regime_state"])
        name = str(regime_df.loc[index, "regime_name"])
        probability = float(regime_df.loc[index, "state_probability"])
        frame.stats.hmm_state = state
        frame.stats.regime_name = name
        frame.metadata["regime_engine"] = str(regime_df.loc[index, "regime_engine"])
        frame.metadata["hmm_probability"] = probability
        probability_map = {
            int(column.rsplit("_", 1)[-1]): float(regime_df.loc[index, column])
            for column in regime_df.columns
            if column.startswith("state_probability_")
        }
        frame.metadata["hmm_probabilities"] = probability_map
        if labels is not None:
            frame.metadata["hmm_labels"] = labels
