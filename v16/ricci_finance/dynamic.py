from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from .sector_objects import normalize_memberships


def ricci_communities(graph: nx.Graph) -> dict[str, int]:
    """Detect graph communities with weighted greedy modularity.

    Correlation magnitude is used as the community weight. Isolated nodes are
    assigned their own cluster so every ticker receives a label.
    """
    if graph.number_of_nodes() == 0:
        return {}
    work = graph.copy()
    for u, v, data in work.edges(data=True):
        data["community_weight"] = max(abs(float(data.get("correlation", 0.0))), 1e-9)
    communities = list(nx.community.greedy_modularity_communities(work, weight="community_weight"))
    output: dict[str, int] = {}
    for cluster_id, members in enumerate(communities):
        for node in members:
            output[str(node)] = cluster_id
    next_id = len(communities)
    for node in work.nodes:
        if str(node) not in output:
            output[str(node)] = next_id
            next_id += 1
    return output


def dynamic_memberships_for_graph(
    graph: nx.Graph,
    static_memberships: Mapping[str, Mapping[str, float]],
    *,
    static_weight: float = 0.65,
    neighbor_weight: float = 0.25,
    signal_weight: float = 0.10,
) -> dict[str, dict[str, float]]:
    """Create per-frame sector weights from static profiles and graph context.

    Neighbor affinity is accumulated from adjacent tickers using absolute
    correlation. A small capital/curvature signal increases the node's current
    primary theme, producing gradual rolling-window evolution while preserving
    a stable fundamental anchor.
    """
    output: dict[str, dict[str, float]] = {}
    for node in graph.nodes:
        ticker = str(node)
        base = normalize_memberships(static_memberships.get(ticker, {"Other": 1.0}))
        neighbor_scores: defaultdict[str, float] = defaultdict(float)
        total_edge = 0.0
        for neighbor in graph.neighbors(node):
            edge = graph.edges[node, neighbor]
            strength = abs(float(edge.get("correlation", 0.0)))
            total_edge += strength
            for sector, weight in static_memberships.get(str(neighbor), {"Other": 1.0}).items():
                neighbor_scores[str(sector)] += strength * float(weight)
        neighbor_profile = normalize_memberships(neighbor_scores if total_edge > 0 else base)

        primary = max(base, key=base.get)
        capital = max(float(graph.nodes[node].get("capital_share", 0.0)), 0.0)
        curvature = float(graph.nodes[node].get("ricciCurvature", 0.0))
        signal = max(0.05, 1.0 + capital + 0.20 * np.tanh(curvature))

        raw: defaultdict[str, float] = defaultdict(float)
        for sector, value in base.items():
            raw[sector] += static_weight * float(value)
        for sector, value in neighbor_profile.items():
            raw[sector] += neighbor_weight * float(value)
        raw[primary] += signal_weight * signal
        output[ticker] = normalize_memberships(raw)
    return output


def build_dynamic_sector_history(
    frames: Sequence[dict],
    static_memberships: Mapping[str, Mapping[str, float]],
    **kwargs,
) -> list[dict[str, dict[str, float]]]:
    return [
        dynamic_memberships_for_graph(frame["graph"], static_memberships, **kwargs)
        for frame in frames
    ]


def dynamic_primary_history(history: Sequence[Mapping[str, Mapping[str, float]]]) -> list[dict[str, str]]:
    return [
        {ticker: max(weights, key=weights.get) for ticker, weights in frame.items()}
        for frame in history
    ]


def sector_evolution_table(
    frames: Sequence[dict],
    history: Sequence[Mapping[str, Mapping[str, float]]],
) -> pd.DataFrame:
    rows = []
    for frame, memberships in zip(frames, history):
        date = pd.Timestamp(frame["date"])
        for ticker, weights in memberships.items():
            for sector, weight in weights.items():
                rows.append({"date": date, "ticker": ticker, "sector": sector, "weight": float(weight)})
    return pd.DataFrame(rows)


def compare_cluster_labels(
    profiles,
    themes: Mapping[str, Mapping[str, float]],
    ricci_labels: Mapping[str, int],
    gnn_labels: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    rows = []
    for ticker in sorted(themes):
        profile = profiles.get(ticker) if profiles else None
        theme = max(themes[ticker], key=themes[ticker].get)
        rows.append({
            "ticker": ticker,
            "yahoo_sector": getattr(profile, "official_sector", "Manual/Unknown"),
            "detected_theme": theme,
            "ricci_community": ricci_labels.get(ticker, -1),
            "gnn_latent_cluster": (gnn_labels or {}).get(ticker, -1),
        })
    return pd.DataFrame(rows)
