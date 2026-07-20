from __future__ import annotations
import io, math, warnings
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
try:
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
except Exception:
    OllivierRicci = None
warnings.filterwarnings("ignore")
RANDOM_STATE=42
np.random.seed(RANDOM_STATE)

def financial_distance_from_corr(
    corr: pd.DataFrame,
) -> pd.DataFrame:
    clipped = corr.clip(-1.0, 1.0)
    values = np.sqrt(2.0 * (1.0 - clipped))
    return pd.DataFrame(
        values,
        index=corr.index,
        columns=corr.columns,
    )

def build_graph_from_window(
    window_returns: pd.DataFrame,
    k_neighbors: int = 3,
    min_corr: float = 0.05,
    max_bridges: int = 3,
    min_pair_obs: int = 20,
) -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    usable_columns = [
        column
        for column in window_returns.columns
        if window_returns[column].notna().sum() >= min_pair_obs
    ]

    clean = window_returns[usable_columns]
    corr = clean.corr(min_periods=min_pair_obs)
    corr = corr.replace([np.inf, -np.inf], np.nan)

    dist = financial_distance_from_corr(corr)

    graph = nx.Graph()
    graph.add_nodes_from(window_returns.columns)

    for column in window_returns.columns:
        valid_obs = int(window_returns[column].notna().sum())
        graph.nodes[column]["valid_obs"] = valid_obs
        graph.nodes[column]["status"] = (
            "active"
            if valid_obs >= min_pair_obs
            else "waiting_for_data"
        )

    candidates = {}

    for source in corr.columns:
        nearest = (
            dist.loc[source]
            .drop(index=source, errors="ignore")
            .sort_values()
            .head(max(1, int(k_neighbors)))
        )

        for target, distance in nearest.items():
            rho = float(corr.loc[source, target])

            if not np.isfinite(rho) or rho < min_corr:
                continue

            edge_key = tuple(sorted((str(source), str(target))))
            candidates[edge_key] = (
                edge_key[0],
                edge_key[1],
                float(distance),
                rho,
            )

    for source, target, distance, rho in candidates.values():
        overlap = int(
            window_returns[[source, target]]
            .dropna()
            .shape[0]
        )

        confidence = min(
            1.0,
            overlap / max(min_pair_obs * 3, 1),
        )

        graph.add_edge(
            source,
            target,
            weight=distance,
            distance=distance,
            correlation=rho,
            overlap_n=overlap,
            confidence=confidence,
            bridge=False,
        )

    existing = {
        tuple(sorted(edge))
        for edge in graph.edges()
    }

    extras = []

    for index, source in enumerate(corr.columns):
        for target in corr.columns[index + 1 :]:
            edge_key = tuple(sorted((source, target)))

            if edge_key in existing:
                continue

            rho = float(corr.loc[source, target])

            if np.isfinite(rho) and rho > 0:
                distance = float(
                    np.sqrt(
                        2.0
                        * (
                            1.0
                            - np.clip(rho, -1.0, 1.0)
                        )
                    )
                )

                extras.append(
                    (
                        source,
                        target,
                        distance,
                        rho,
                    )
                )

    for source, target, distance, rho in sorted(
        extras,
        key=lambda item: item[2],
    )[: int(max_bridges)]:
        overlap = int(
            window_returns[[source, target]]
            .dropna()
            .shape[0]
        )

        graph.add_edge(
            source,
            target,
            weight=distance,
            distance=distance,
            correlation=rho,
            overlap_n=overlap,
            confidence=min(1.0, overlap / 60.0),
            bridge=True,
        )

    return graph, corr, dist

def attach_capital_attributes(
    graph: nx.Graph,
    dollar_volume_window: pd.DataFrame,
    capital_alpha: float = 0.35,
    use_capital_weighting: bool = True,
) -> nx.Graph:
    result = graph.copy()

    mass = (
        dollar_volume_window
        .reindex(columns=list(result.nodes()))
        .replace([np.inf, -np.inf], np.nan)
        .median(axis=0, skipna=True)
        .fillna(0.0)
        .clip(lower=0.0)
    )

    total_mass = float(mass.sum())

    for node in result.nodes():
        node_mass = float(mass.get(node, 0.0))

        result.nodes[node]["capital_mass"] = node_mass
        result.nodes[node]["capital_share"] = (
            node_mass / total_mass
            if total_mass > 0
            else 0.0
        )

    positive = mass[mass > 0]
    scale = (
        float(np.median(positive))
        if len(positive)
        else 1.0
    )
    log_mass = np.log1p(
        mass / max(scale, 1e-12)
    )

    for source, target, edge_data in result.edges(data=True):
        raw_distance = float(
            edge_data.get(
                "distance",
                edge_data.get("weight", 1.0),
            )
        )

        source_mass = float(mass.get(source, 0.0))
        target_mass = float(mass.get(target, 0.0))

        source_log_mass = float(
            log_mass.get(source, 0.0)
        )
        target_log_mass = float(
            log_mass.get(target, 0.0)
        )

        capital_similarity = float(
            np.exp(
                -abs(
                    source_log_mass
                    - target_log_mass
                )
            )
        )

        correlation = float(
            edge_data.get("correlation", 0.0)
        )

        edge_capital_flow = float(
            np.sqrt(
                max(source_mass, 0.0)
                * max(target_mass, 0.0)
            )
            * max(correlation, 0.0)
        )

        effective_distance = raw_distance

        if use_capital_weighting:
            effective_distance = (
                raw_distance
                / (
                    1.0
                    + float(capital_alpha)
                    * capital_similarity
                )
            )

        edge_data["raw_distance"] = raw_distance
        edge_data["capital_similarity"] = capital_similarity
        edge_data["edge_capital_flow"] = edge_capital_flow
        edge_data["distance"] = effective_distance
        edge_data["weight"] = effective_distance

    return result

def compute_ricci_curvature(
    graph: nx.Graph,
    alpha: float = 0.5,
    method: str = "OTD",
    proc: int = 1,
) -> nx.Graph:
    result = graph.copy()

    if result.number_of_edges() == 0:
        return result

    if OllivierRicci is not None:
        try:
            engine = OllivierRicci(
                result,
                alpha=alpha,
                method=method,
                weight="weight",
                proc=proc,
                verbose="ERROR",
            )

            engine.compute_ricci_curvature()
            return engine.G.copy()

        except Exception as exc:
            print(
                "GraphRicciCurvature failed; "
                "using fallback curvature:",
                exc,
            )

    for source, target in result.edges():
        common_neighbors = len(
            list(
                nx.common_neighbors(
                    result,
                    source,
                    target,
                )
            )
        )

        degree_sum = max(
            1,
            result.degree(source)
            + result.degree(target)
            - 2,
        )

        result[source][target]["ricciCurvature"] = float(
            2.0
            * common_neighbors
            / degree_sum
            - 0.5
        )

    return result

def component_entropy(
    component_sizes: list[int],
) -> float:
    total = sum(component_sizes)

    if total <= 0:
        return 0.0

    probabilities = (
        np.asarray(component_sizes, dtype=float)
        / total
    )

    return float(
        -np.sum(
            probabilities[probabilities > 0]
            * np.log(
                probabilities[probabilities > 0]
            )
        )
    )

def edge_jaccard(
    graph_a: nx.Graph,
    graph_b: nx.Graph,
) -> float:
    edges_a = {
        frozenset(edge)
        for edge in graph_a.edges()
    }

    edges_b = {
        frozenset(edge)
        for edge in graph_b.edges()
    }

    union = edges_a | edges_b

    return (
        1.0
        if not union
        else len(edges_a & edges_b)
        / len(union)
    )

def graph_feature_row(
    graph: nx.Graph,
    date: pd.Timestamp,
) -> dict[str, float | int | str]:
    curvatures = np.asarray(
        [
            float(
                edge_data.get(
                    "ricciCurvature",
                    0.0,
                )
            )
            for _, _, edge_data
            in graph.edges(data=True)
        ],
        dtype=float,
    )

    components = list(
        nx.connected_components(graph)
    )
    component_sizes = [
        len(component)
        for component in components
    ]

    node_capital = np.asarray(
        [
            float(
                node_data.get(
                    "capital_mass",
                    0.0,
                )
            )
            for _, node_data
            in graph.nodes(data=True)
        ],
        dtype=float,
    )

    edge_capital = np.asarray(
        [
            float(
                edge_data.get(
                    "edge_capital_flow",
                    0.0,
                )
            )
            for _, _, edge_data
            in graph.edges(data=True)
        ],
        dtype=float,
    )

    total_node_capital = float(
        node_capital.sum()
    )

    return {
        "date": str(pd.Timestamp(date).date()),
        "avg_ricci": (
            float(curvatures.mean())
            if len(curvatures)
            else 0.0
        ),
        "ricci_std": (
            float(curvatures.std())
            if len(curvatures)
            else 0.0
        ),
        "ricci_min": (
            float(curvatures.min())
            if len(curvatures)
            else 0.0
        ),
        "ricci_max": (
            float(curvatures.max())
            if len(curvatures)
            else 0.0
        ),
        "negative_edge_ratio": (
            float(
                np.mean(curvatures < 0)
            )
            if len(curvatures)
            else 0.0
        ),
        "clusters": len(components),
        "largest_component": max(
            component_sizes,
            default=0,
        ),
        "largest_component_ratio": (
            max(component_sizes, default=0)
            / max(graph.number_of_nodes(), 1)
        ),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": (
            float(nx.density(graph))
            if graph.number_of_nodes() > 1
            else 0.0
        ),
        "component_entropy": component_entropy(
            component_sizes
        ),
        "total_node_capital": (
            total_node_capital
        ),
        "total_edge_capital_flow": (
            float(edge_capital.sum())
            if len(edge_capital)
            else 0.0
        ),
        "avg_edge_capital_flow": (
            float(edge_capital.mean())
            if len(edge_capital)
            else 0.0
        ),
        "max_node_capital_share": (
            float(
                node_capital.max()
                / total_node_capital
            )
            if total_node_capital > 0
            else 0.0
        ),
    }

class ExplicitDurationGaussianHSMM:
    def __init__(
        self,
        n_states: int = 3,
        min_duration: int = 3,
        max_duration: int = 40,
        random_state: int = 42,
    ):
        self.n_states = int(n_states)
        self.min_duration = int(min_duration)
        self.max_duration = int(max_duration)
        self.random_state = int(random_state)

    @staticmethod
    def _run_lengths(
        states: np.ndarray,
    ) -> dict[int, list[int]]:
        durations = {}

        if len(states) == 0:
            return durations

        start = 0

        while start < len(states):
            end = start + 1

            while (
                end < len(states)
                and states[end] == states[start]
            ):
                end += 1

            state = int(states[start])
            durations.setdefault(
                state,
                [],
            ).append(end - start)

            start = end

        return durations

    def fit(
        self,
        X: np.ndarray,
    ):
        base_model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            min_covar=1e-4,
            n_iter=500,
            random_state=self.random_state,
        )

        base_model.fit(X)
        base_states = base_model.predict(X)

        self.means_ = np.asarray(
            base_model.means_,
            dtype=float,
        )

        self.covars_ = np.asarray(
            base_model.covars_,
            dtype=float,
        )

        self.startprob_ = np.clip(
            base_model.startprob_,
            1e-12,
            1.0,
        )

        self.transmat_ = np.clip(
            base_model.transmat_,
            1e-12,
            1.0,
        )

        durations = self._run_lengths(
            base_states
        )

        self.duration_lambda_ = np.asarray(
            [
                max(
                    self.min_duration,
                    np.mean(
                        durations.get(
                            state,
                            [8],
                        )
                    ),
                )
                for state in range(
                    self.n_states
                )
            ],
            dtype=float,
        )

        return self

    def _emission_log_probability(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        output = np.empty(
            (
                len(X),
                self.n_states,
            ),
            dtype=float,
        )

        for state in range(
            self.n_states
        ):
            covariance = np.diag(
                np.maximum(
                    self.covars_[state],
                    1e-5,
                )
            )

            output[:, state] = (
                multivariate_normal.logpdf(
                    X,
                    mean=self.means_[state],
                    cov=covariance,
                    allow_singular=True,
                )
            )

        return output

    def _duration_log_probability(
        self,
        state: int,
        duration: int,
    ) -> float:
        if (
            duration < self.min_duration
            or duration > self.max_duration
        ):
            return -np.inf

        shifted_lambda = max(
            self.duration_lambda_[state]
            - self.min_duration,
            1e-6,
        )

        shifted_duration = (
            duration
            - self.min_duration
        )

        return (
            shifted_duration
            * math.log(
                shifted_lambda
            )
            - shifted_lambda
            - math.lgamma(
                shifted_duration + 1
            )
        )

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        X = np.asarray(
            X,
            dtype=float,
        )

        sequence_length = len(X)

        emission = (
            self._emission_log_probability(
                X
            )
        )

        cumulative = np.vstack(
            [
                np.zeros(
                    self.n_states
                ),
                np.cumsum(
                    emission,
                    axis=0,
                ),
            ]
        )

        score = np.full(
            (
                sequence_length + 1,
                self.n_states,
            ),
            -np.inf,
        )

        back_state = np.full(
            (
                sequence_length + 1,
                self.n_states,
            ),
            -1,
            dtype=int,
        )

        back_duration = np.zeros(
            (
                sequence_length + 1,
                self.n_states,
            ),
            dtype=int,
        )

        log_start = np.log(
            self.startprob_
        )

        log_transition = np.log(
            self.transmat_
        )

        for end in range(
            1,
            sequence_length + 1,
        ):
            maximum_duration = min(
                self.max_duration,
                end,
            )

            for state in range(
                self.n_states
            ):
                for duration in range(
                    self.min_duration,
                    maximum_duration + 1,
                ):
                    begin = end - duration

                    segment_score = (
                        cumulative[end, state]
                        - cumulative[begin, state]
                    )

                    duration_score = (
                        self._duration_log_probability(
                            state,
                            duration,
                        )
                    )

                    if begin == 0:
                        candidate = (
                            log_start[state]
                            + segment_score
                            + duration_score
                        )

                        previous_state = -1

                    else:
                        previous_scores = (
                            score[begin]
                            + log_transition[:, state]
                        )

                        previous_scores[state] = (
                            -np.inf
                        )

                        previous_state = int(
                            np.argmax(
                                previous_scores
                            )
                        )

                        candidate = (
                            previous_scores[
                                previous_state
                            ]
                            + segment_score
                            + duration_score
                        )

                    if (
                        candidate
                        > score[end, state]
                    ):
                        score[end, state] = (
                            candidate
                        )

                        back_state[
                            end,
                            state,
                        ] = previous_state

                        back_duration[
                            end,
                            state,
                        ] = duration

        states = np.empty(
            sequence_length,
            dtype=int,
        )

        end = sequence_length
        state = int(
            np.argmax(
                score[
                    sequence_length
                ]
            )
        )

        while end > 0:
            duration = int(
                back_duration[
                    end,
                    state,
                ]
            )

            if duration <= 0:
                duration = min(
                    end,
                    self.min_duration,
                )

            begin = max(
                0,
                end - duration,
            )

            states[begin:end] = state

            state = int(
                back_state[
                    end,
                    state,
                ]
            )

            end = begin

            if state < 0 and end > 0:
                state = int(
                    np.argmax(
                        score[end]
                    )
                )

        return states

def run_lengths(
    states: np.ndarray,
) -> np.ndarray:
    states = np.asarray(
        states,
        dtype=int,
    )

    if len(states) == 0:
        return np.asarray(
            [],
            dtype=int,
        )

    boundaries = np.flatnonzero(
        np.r_[
            True,
            states[1:]
            != states[:-1],
            True,
        ]
    )

    return np.diff(boundaries)

def switch_rate(
    states: np.ndarray,
) -> float:
    states = np.asarray(
        states,
        dtype=int,
    )

    return (
        float(
            np.mean(
                states[1:]
                != states[:-1]
            )
        )
        if len(states) > 1
        else 0.0
    )

def engine_metrics(
    name: str,
    states: np.ndarray,
) -> dict[str, float | str]:
    lengths = run_lengths(states)

    return {
        "engine": name,
        "switch_rate": switch_rate(states),
        "median_run_length": (
            float(np.median(lengths))
            if len(lengths)
            else np.nan
        ),
        "mean_run_length": (
            float(np.mean(lengths))
            if len(lengths)
            else np.nan
        ),
        "short_run_ratio_le_2": (
            float(np.mean(lengths <= 2))
            if len(lengths)
            else np.nan
        ),
    }

def build_regime_labels(
    features: pd.DataFrame,
    states: np.ndarray,
) -> dict[int, str]:
    table = features.copy()
    table["state"] = states

    summary = (
        table
        .groupby("state")
        .agg(
            avg_ricci=(
                "avg_ricci",
                "mean",
            ),
            density=(
                "density",
                "mean",
            ),
            largest_component_ratio=(
                "largest_component_ratio",
                "mean",
            ),
            negative_edge_ratio=(
                "negative_edge_ratio",
                "mean",
            ),
        )
    )

    ordered = list(
        summary.sort_values(
            [
                "avg_ricci",
                "largest_component_ratio",
                "density",
            ]
        ).index
    )

    standard_names = [
        "stress / fragmentation",
        "transition / rotation",
        "coherent risk-on",
    ]

    labels = {}

    for index, state in enumerate(
        ordered
    ):
        labels[int(state)] = (
            standard_names[index]
            if index
            < len(standard_names)
            else f"regime_{index}"
        )

    return labels

def build_union_graph(
    frames: list[dict],
) -> nx.Graph:
    union = nx.Graph()

    for frame in frames:
        graph = frame["graph"]

        union.add_nodes_from(
            graph.nodes()
        )

        for (
            source,
            target,
            edge_data,
        ) in graph.edges(
            data=True
        ):
            weight = float(
                edge_data.get(
                    "weight",
                    1.0,
                )
            )

            if union.has_edge(
                source,
                target,
            ):
                union[source][target][
                    "weight"
                ] = min(
                    union[source][target][
                        "weight"
                    ],
                    weight,
                )
            else:
                union.add_edge(
                    source,
                    target,
                    weight=weight,
                )

    return union

def node_z_value(
    graph: nx.Graph,
    node: str,
) -> float:
    curvatures = [
        float(
            graph[node][neighbor]
            .get(
                "ricciCurvature",
                0.0,
            )
        )
        for neighbor in graph.neighbors(
            node
        )
    ]

    return (
        float(
            -np.mean(
                curvatures
            )
        )
        if curvatures
        else 0.0
    )

def ricci_color(
    curvature: float,
) -> str:
    curvature = float(
        np.clip(
            curvature,
            -0.6,
            0.6,
        )
    )

    normalized = (
        curvature + 0.6
    ) / 1.2

    red = int(
        255
        * (
            1.0
            - normalized
        )
    )

    blue = int(
        255
        * normalized
    )

    return f"rgb({red},80,{blue})"

def graph_frame_traces(
    frame_index: int,
    show_edge_labels: bool = True,
    top_edge_labels: int = 20,
):
    graph = frames[
        frame_index
    ]["graph"]

    traces = []

    edge_records = []

    for (
        source,
        target,
        edge_data,
    ) in graph.edges(
        data=True
    ):
        source_x, source_y, _ = (
            positions[source]
        )
        target_x, target_y, _ = (
            positions[target]
        )

        source_z = node_z_value(
            graph,
            source,
        )
        target_z = node_z_value(
            graph,
            target,
        )

        curvature = float(
            edge_data.get(
                "ricciCurvature",
                0.0,
            )
        )

        flow = float(
            edge_data.get(
                "edge_capital_flow",
                0.0,
            )
        )

        correlation = float(
            edge_data.get(
                "correlation",
                np.nan,
            )
        )

        distance = float(
            edge_data.get(
                "distance",
                np.nan,
            )
        )

        traces.append(
            go.Scatter3d(
                x=[
                    source_x,
                    target_x,
                ],
                y=[
                    source_y,
                    target_y,
                ],
                z=[
                    source_z,
                    target_z,
                ],
                mode="lines",
                line={
                    "width": (
                        1.0
                        + 4.0
                        * np.sqrt(
                            max(
                                flow,
                                0.0,
                            )
                            / max(
                                1.0,
                                flow,
                            )
                        )
                    ),
                    "color": ricci_color(
                        curvature
                    ),
                },
                hovertext=(
                    f"{source}–{target}<br>"
                    f"ρ={correlation:.3f}<br>"
                    f"d={distance:.3f}<br>"
                    f"κ={curvature:.3f}<br>"
                    f"flow={flow:,.0f}"
                ),
                hoverinfo="text",
                showlegend=False,
            )
        )

        edge_records.append(
            {
                "source": source,
                "target": target,
                "x": (
                    source_x
                    + target_x
                )
                / 2.0,
                "y": (
                    source_y
                    + target_y
                )
                / 2.0,
                "z": (
                    source_z
                    + target_z
                )
                / 2.0,
                "flow": flow,
                "label": (
                    f"ρ={correlation:.2f}<br>"
                    f"d={distance:.2f}<br>"
                    f"κ={curvature:.2f}"
                ),
            }
        )

    if show_edge_labels:
        selected_edges = sorted(
            edge_records,
            key=lambda item: item[
                "flow"
            ],
            reverse=True,
        )[: int(top_edge_labels)]

        traces.append(
            go.Scatter3d(
                x=[
                    edge["x"]
                    for edge in selected_edges
                ],
                y=[
                    edge["y"]
                    for edge in selected_edges
                ],
                z=[
                    edge["z"]
                    for edge in selected_edges
                ],
                mode="text",
                text=[
                    edge["label"]
                    for edge in selected_edges
                ],
                textfont={
                    "size": 9,
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

    node_x = []
    node_y = []
    node_z = []
    node_text = []
    node_size = []
    node_hover = []

    capital_values = np.asarray(
        [
            float(
                graph.nodes[node]
                .get(
                    "capital_mass",
                    0.0,
                )
            )
            for node in graph.nodes()
        ],
        dtype=float,
    )

    capital_scale = (
        np.percentile(
            capital_values[
                capital_values > 0
            ],
            90,
        )
        if np.any(
            capital_values > 0
        )
        else 1.0
    )

    for node in sorted(
        graph.nodes()
    ):
        x, y, _ = positions[node]
        z = node_z_value(
            graph,
            node,
        )

        capital_mass = float(
            graph.nodes[node]
            .get(
                "capital_mass",
                0.0,
            )
        )

        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_text.append(node)
        node_size.append(
            9.0
            + 20.0
            * np.sqrt(
                max(
                    capital_mass,
                    0.0,
                )
                / max(
                    capital_scale,
                    1e-12,
                )
            )
        )

        node_hover.append(
            f"Ticker: {node}<br>"
            f"capital={capital_mass:,.0f}<br>"
            f"Ricci stress z={z:.3f}"
        )

    traces.append(
        go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            marker={
                "size": node_size,
                "line": {
                    "width": 1,
                    "color": "#222",
                },
            },
            hovertext=node_hover,
            hoverinfo="text",
            showlegend=False,
        )
    )

    return traces

def download_data(tickers, period, interval):
    raw=yf.download(tickers=tickers, period=period, interval=interval, auto_adjust=False, group_by="column", progress=False, threads=True)
    if raw.empty: raise ValueError("yfinance returned no data")
    def field(name):
        if isinstance(raw.columns,pd.MultiIndex):
            x=raw[name].copy()
        else:
            x=raw[[name]].copy(); x.columns=[tickers[0]]
        return x.sort_index().dropna(axis=1,how="all")
    close,volume=field("Close"),field("Volume")
    common=close.columns.intersection(volume.columns)
    return close[common], volume[common], close[common]*volume[common]

def build_frames(returns,dollar_volume,window,step,max_frames,k,min_corr,max_bridges,min_obs,alpha,method,proc,capital_alpha,use_capital):
    starts=list(range(0,max(0,len(returns)-window+1),step))
    if max_frames and len(starts)>max_frames: starts=starts[-max_frames:]
    frames=[]; rows=[]; prev=None
    prog=st.progress(0,text="Building Ricci frames…")
    for j,start in enumerate(starts):
        end=start+window
        wr=returns.iloc[start:end]
        dv=dollar_volume.reindex(wr.index).iloc[:,:]
        g,_,_=build_graph_from_window(wr,k,min_corr,max_bridges,min_obs)
        g=attach_capital_attributes(g,dv,capital_alpha,use_capital)
        g=compute_ricci_curvature(g,alpha,method,proc)
        row=graph_feature_row(g,wr.index[-1])
        row["edge_stability"]=edge_jaccard(prev,g) if prev is not None else 1.0
        frames.append({"graph":g,"date":wr.index[-1]}); rows.append(row); prev=g
        prog.progress((j+1)/max(1,len(starts)),text=f"Frame {j+1}/{len(starts)}")
    prog.empty()
    return frames,pd.DataFrame(rows)

def make_positions(frames):
    u=build_union_graph(frames)
    return nx.spring_layout(u,dim=3,seed=RANDOM_STATE,weight="weight") if u.number_of_nodes() else {}

def rgba(curv,alpha):
    curv=float(np.clip(curv,-.6,.6)); n=(curv+.6)/1.2
    return f"rgba({int(255*(1-n))},80,{int(255*n)},{float(np.clip(alpha,.08,1)):.3f})"

def snapshot(frames,feature_df,positions,index):
    g=frames[index]["graph"]; reg=str(feature_df.loc[index,"regime_name"]); prob=float(feature_df.loc[index,"hmm_state_probability"])
    colors={"stress / fragmentation":"#d62728","transition / rotation":"#ffbf00","coherent risk-on":"#2ca02c"}
    fig=go.Figure(); flows=np.array([float(d.get("edge_capital_flow",0)) for *_,d in g.edges(data=True)])
    pos=flows[np.isfinite(flows)&(flows>0)]; ref=float(np.percentile(pos,90)) if len(pos) else 1.
    for a,b,d in g.edges(data=True):
        x1,y1,_=positions[a]; x2,y2,_=positions[b]; z1=node_z_value(g,a); z2=node_z_value(g,b)
        c=float(d.get("ricciCurvature",0)); f=float(d.get("edge_capital_flow",0)); fs=float(np.clip(np.sqrt(max(f,0)/max(ref,1e-12)),0,1)); al=.10+.58*fs+.32*min(abs(c)/.6,1)
        fig.add_trace(go.Scatter3d(x=[x1,x2],y=[y1,y2],z=[z1,z2],mode="lines",line={"width":.8+4.2*fs,"color":rgba(c,al)},hovertext=f"{a}–{b}<br>ρ={d.get('correlation',np.nan):.3f}<br>κ={c:.3f}<br>flow={f:,.0f}",hoverinfo="text",showlegend=False))
    caps=np.array([float(g.nodes[n].get("capital_mass",0)) for n in g]); p=caps[caps>0]; cref=float(np.percentile(p,90)) if len(p) else 1.
    xs=[];ys=[];zs=[];labs=[];sizes=[];hov=[]
    for n in sorted(g):
        x,y,_=positions[n]; z=node_z_value(g,n); cap=float(g.nodes[n].get("capital_mass",0)); xs.append(x);ys.append(y);zs.append(z);labs.append(n);sizes.append(10+22*np.sqrt(max(cap,0)/max(cref,1e-12)));hov.append(f"{n}<br>capital={cap:,.0f}<br>Ricci stress={z:.3f}")
    fig.add_trace(go.Scatter3d(x=xs,y=ys,z=zs,mode="markers+text",text=labs,textposition="top center",marker={"size":sizes,"color":colors.get(reg,"#777"),"opacity":.92,"line":{"width":1,"color":"#222"}},hovertext=hov,hoverinfo="text",name=reg))
    fig.update_layout(title=f"Final frame {index+1} · {frames[index]['date'].date()} · {reg} · posterior {prob:.1%}",height=780,scene={"xaxis":{"visible":False},"yaxis":{"visible":False},"zaxis":{"title":"Ricci stress"},"aspectmode":"cube"})
    return fig

st.set_page_config(page_title="Ricci Finance v14",layout="wide")
st.title("Ricci Finance v14 — HMM Regime Research")
st.caption("雙語 Streamlit app · yfinance · Ollivier–Ricci curvature · HMM/Viterbi · pomegranate 1.1.2 validation")
with st.sidebar:
    st.header("Configuration / 設定")
    tickers_txt=st.text_area("Tickers", "NVDA,AMD,AVGO,TSM,MU,MRVL,AMAT,LRCX,KLAC,ANET,AAOI,COHR,LITE,SMCI,PLTR,IONQ,QBTS,QUBT,RGTI,NBIS",height=120)
    period=st.selectbox("Period",["1y","2y","5y","10y"],index=2); interval=st.selectbox("Interval",["1d","1wk"],index=0)
    window=st.slider("Rolling window",30,126,63); step=st.slider("Rolling step",1,21,5); max_frames=st.slider("Maximum frames",20,180,120)
    k=st.slider("K neighbors",1,8,3); min_corr=st.slider("Minimum correlation",-0.2,0.8,0.05,.05); max_bridges=st.slider("Bridge edges",0,10,3); n_states=st.slider("HMM states",2,5,3)
    run=st.button("Run analysis / 執行分析",type="primary",use_container_width=True)

with st.expander("HMM and Viterbi / HMM 與 Viterbi",expanded=False):
    st.markdown("""**English.** The Gaussian HMM assumes that each rolling graph-feature vector is emitted by an unobserved market regime. Viterbi finds the single globally most likely state path, while forward–backward posterior probabilities quantify frame-level uncertainty.

**中文。** 高斯 HMM 假設每個滾動網路特徵向量由不可直接觀察的市場狀態產生。Viterbi 尋找整條序列的全域最佳狀態路徑；forward–backward posterior 則衡量每一幀的不確定性。""")

if run:
  try:
    tickers=list(dict.fromkeys(x.strip().upper() for x in tickers_txt.replace("\n",",").split(",") if x.strip()))
    close,volume,dv=download_data(tickers,period,interval)
    returns=np.log(close/close.shift(1)).replace([np.inf,-np.inf],np.nan).dropna(how="all")
    dv=dv.reindex(index=returns.index,columns=returns.columns)
    frames,feature_df=build_frames(returns,dv,window,step,max_frames,k,min_corr,max_bridges,20,.5,"OTD",1,.35,True)
    if len(feature_df)<max(10,n_states*3): raise ValueError("Too few rolling frames for HMM. Increase period or reduce window/step.")
    features=["avg_ricci","ricci_std","ricci_min","negative_edge_ratio","density","largest_component_ratio","component_entropy","capital_concentration","edge_capital_concentration","edge_stability"]
    X=feature_df[features].replace([np.inf,-np.inf],np.nan).interpolate(limit_direction="both").fillna(0)
    scaler=StandardScaler(); Xs=scaler.fit_transform(X)
    model=GaussianHMM(n_components=n_states,covariance_type="diag",n_iter=300,tol=1e-4,min_covar=1e-4,random_state=RANDOM_STATE)
    model.fit(Xs); viterbi=model.predict(Xs); post=model.predict_proba(Xs); posterior=post.argmax(1)
    feature_df["hmm_viterbi_state"]=viterbi; feature_df["hmm_posterior_state"]=posterior; feature_df["hmm_state_probability"]=post.max(1)
    labels=build_regime_labels(feature_df,viterbi); feature_df["regime_name"]=[labels[int(s)] for s in viterbi]
    st.success(f"Completed {len(frames)} frames with {len(close.columns)} tickers.")
    a,b,c,d=st.columns(4); a.metric("Final regime",feature_df.iloc[-1]["regime_name"]); b.metric("Posterior confidence",f"{feature_df.iloc[-1]['hmm_state_probability']:.1%}"); c.metric("Viterbi switches",int(np.sum(viterbi[1:]!=viterbi[:-1]))); d.metric("Edges, final",frames[-1]["graph"].number_of_edges())
    tab1,tab2,tab3,tab4=st.tabs(["Regime timeline","3-D snapshot","Comparison","Data"])
    with tab1:
        plot_df=feature_df.copy(); plot_df["frame"]=np.arange(len(plot_df));
        fig=go.Figure(); fig.add_trace(go.Scatter(x=plot_df["date"],y=plot_df["hmm_viterbi_state"],mode="lines+markers",name="Viterbi")); fig.add_trace(go.Scatter(x=plot_df["date"],y=plot_df["hmm_posterior_state"],mode="lines",name="Posterior")); fig.update_layout(height=430,yaxis_title="State",xaxis_title="Date"); st.plotly_chart(fig,use_container_width=True)
        st.dataframe(feature_df[["date","regime_name","hmm_viterbi_state","hmm_posterior_state","hmm_state_probability"]].tail(30),use_container_width=True)
    with tab2:
        idx=st.slider("Frame / 幀",0,len(frames)-1,len(frames)-1); positions=make_positions(frames); st.plotly_chart(snapshot(frames,feature_df,positions,idx),use_container_width=True)
    with tab3:
        agree=float(np.mean(viterbi==posterior)); st.metric("Viterbi–posterior agreement",f"{agree:.1%}")
        st.dataframe(pd.DataFrame({"metric":["Viterbi switch rate","Posterior switch rate","Median Viterbi run"],"value":[switch_rate(viterbi),switch_rate(posterior),float(np.median(run_lengths(viterbi)))]}),use_container_width=True)
        st.code(model.transmat_,language=None)
    with tab4:
        st.dataframe(feature_df,use_container_width=True,height=500)
        st.download_button("Download feature CSV",feature_df.to_csv(index=False).encode(),"v14_streamlit_features.csv","text/csv")
  except Exception as exc:
    st.exception(exc)
else:
    st.info("Choose parameters in the sidebar and press Run analysis. / 在側欄設定參數後按下執行分析。")
