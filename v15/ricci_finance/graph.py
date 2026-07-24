from __future__ import annotations

import math
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linprog

CurvatureEngine = Literal["ollivier_lp", "forman"]


def build_graph_from_window(
    window_returns: pd.DataFrame,
    k: int = 4,
    min_corr: float = 0.2,
    max_bridges: int = 3,
    min_obs: int = 25,
):
    """Build a sparse correlation-distance graph from one rolling window."""
    cols = [
        c for c in window_returns.columns
        if window_returns[c].notna().sum() >= min_obs
    ]
    if len(cols) < 2:
        raise ValueError("Fewer than two tickers have enough observations")

    r = window_returns[cols]
    corr = r.corr(min_periods=min_obs).clip(-1, 1)
    dist = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - corr)))

    G = nx.Graph()
    G.add_nodes_from(cols)

    for u in cols:
        ranked = (
            corr.loc[u]
            .drop(index=u)
            .abs()
            .sort_values(ascending=False)
            .head(k)
        )
        for v in ranked.index:
            rho = float(corr.loc[u, v])
            if np.isfinite(rho) and abs(rho) >= min_corr:
                distance = max(1e-6, float(dist.loc[u, v]))
                G.add_edge(
                    u,
                    v,
                    correlation=rho,
                    distance=distance,
                    weight=distance,
                )

    # Connect components with the strongest available cross-component links.
    for _ in range(max_bridges):
        components = list(nx.connected_components(G))
        if len(components) <= 1:
            break
        best = None
        for i, left in enumerate(components):
            for right in components[i + 1:]:
                for u in left:
                    for v in right:
                        rho = corr.loc[u, v]
                        if np.isfinite(rho) and (
                            best is None or abs(rho) > best[0]
                        ):
                            best = (abs(rho), u, v, float(rho))
        if best is None:
            break
        _, u, v, rho = best
        distance = max(1e-6, float(dist.loc[u, v]))
        G.add_edge(
            u,
            v,
            correlation=rho,
            distance=distance,
            weight=distance,
            bridge=True,
        )

    return G, corr, dist


def attach_capital_attributes(
    G: nx.Graph,
    dollar_volume: pd.DataFrame,
    capital_alpha: float = 0.35,
    use_capital: bool = True,
) -> nx.Graph:
    """Attach node capital shares and edge capital-flow proxies."""
    G = G.copy()
    means = (
        dollar_volume
        .reindex(columns=list(G.nodes))
        .mean(skipna=True)
        .fillna(0)
        .clip(lower=0)
    )
    total = float(means.sum())
    shares = (
        means / total
        if total > 0
        else pd.Series(1 / max(len(means), 1), index=means.index)
    )

    for node in G.nodes:
        share = float(shares.get(node, 0.0))
        G.nodes[node]["capital_share"] = share
        G.nodes[node]["capital_mass"] = share

    for u, v, data in G.edges(data=True):
        flow = math.sqrt(
            max(float(shares.get(u, 0.0)), 0.0)
            * max(float(shares.get(v, 0.0)), 0.0)
        ) * abs(float(data.get("correlation", 0.0)))
        data["edge_capital_flow"] = float(flow)
        if use_capital:
            data["weight"] = max(
                1e-6,
                float(data.get("distance", 1.0))
                / (1.0 + capital_alpha * flow),
            )

    return G


def _neighbor_measure(
    G: nx.Graph,
    node,
    alpha: float,
    weight: str,
) -> tuple[list, np.ndarray]:
    """Return the idleness measure at a node and its one-hop neighbors."""
    neighbors = list(G.neighbors(node))
    support = [node, *neighbors]
    mass = np.zeros(len(support), dtype=float)
    mass[0] = alpha

    if not neighbors:
        mass[0] = 1.0
        return support, mass

    # Closer neighbors receive more of the non-idle mass.
    affinities = np.array(
        [1.0 / max(float(G[node][nbr].get(weight, 1.0)), 1e-12)
         for nbr in neighbors],
        dtype=float,
    )
    affinity_sum = float(affinities.sum())
    if not np.isfinite(affinity_sum) or affinity_sum <= 0:
        affinities = np.ones(len(neighbors), dtype=float)
        affinity_sum = float(len(neighbors))
    mass[1:] = (1.0 - alpha) * affinities / affinity_sum
    return support, mass


def _transport_distance_lp(
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    cost: np.ndarray,
) -> float:
    """Solve the discrete 1-Wasserstein transport problem with SciPy HiGHS."""
    n_source, n_target = cost.shape
    objective = np.asarray(cost, dtype=float).reshape(-1)

    A_eq = np.zeros((n_source + n_target, n_source * n_target), dtype=float)
    b_eq = np.concatenate([source_mass, target_mass]).astype(float)

    # Source marginal constraints.
    for i in range(n_source):
        A_eq[i, i * n_target:(i + 1) * n_target] = 1.0

    # Target marginal constraints.
    for j in range(n_target):
        A_eq[n_source + j, j::n_target] = 1.0

    result = linprog(
        objective,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Optimal-transport LP failed: {result.message}")
    return float(result.fun)


def compute_ollivier_ricci_lp(
    G: nx.Graph,
    alpha: float = 0.5,
    weight: str = "weight",
) -> nx.Graph:
    """
    Compute edge Ollivier-Ricci curvature without POT.

    For each edge (x, y), this solves the discrete optimal-transport problem
    between idleness measures m_x and m_y using scipy.optimize.linprog:

        kappa(x,y) = 1 - W_1(m_x,m_y) / d(x,y)

    Shortest-path distances are computed once per graph and reused by all LPs.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    G = G.copy()
    if G.number_of_edges() == 0:
        for node in G.nodes:
            G.nodes[node]["ricciCurvature"] = 0.0
        G.graph["curvature_engine"] = "ollivier_lp"
        return G

    # Validate positive finite edge lengths.
    for _, _, data in G.edges(data=True):
        value = float(data.get(weight, 1.0))
        if not np.isfinite(value) or value <= 0:
            data[weight] = 1e-6

    all_distances = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))
    measures = {
        node: _neighbor_measure(G, node, alpha, weight)
        for node in G.nodes
    }

    for x, y, data in G.edges(data=True):
        source_support, source_mass = measures[x]
        target_support, target_mass = measures[y]
        cost = np.empty((len(source_support), len(target_support)), dtype=float)

        for i, u in enumerate(source_support):
            for j, v in enumerate(target_support):
                try:
                    cost[i, j] = float(all_distances[u][v])
                except KeyError as exc:
                    raise ValueError(
                        "Ollivier-Ricci requires connected support nodes; "
                        f"no path between {u!r} and {v!r}"
                    ) from exc

        wasserstein = _transport_distance_lp(source_mass, target_mass, cost)
        edge_distance = float(all_distances[x][y])
        curvature = 1.0 - wasserstein / max(edge_distance, 1e-12)

        data["ricciCurvature"] = float(curvature)
        data["wassersteinDistance"] = float(wasserstein)
        data["ricci_engine"] = "ollivier_lp"

    _attach_node_curvature(G)
    G.graph["curvature_engine"] = "ollivier_lp"
    G.graph["ricci_alpha"] = float(alpha)
    return G


def compute_forman_ricci(
    G: nx.Graph,
    weight: str = "weight",
) -> nx.Graph:
    """
    Compute weighted Forman-Ricci edge curvature in pure NetworkX/NumPy.

    Node weights default to 1. Edge weights are interpreted as positive edge
    lengths. This mode is much faster than per-edge optimal-transport LPs.
    """
    G = G.copy()

    for u, v, data in G.edges(data=True):
        w_e = max(float(data.get(weight, 1.0)), 1e-12)
        w_u = max(float(G.nodes[u].get("forman_weight", 1.0)), 1e-12)
        w_v = max(float(G.nodes[v].get("forman_weight", 1.0)), 1e-12)

        adjacent_u = 0.0
        for nbr in G.neighbors(u):
            if nbr == v:
                continue
            w_adj = max(float(G[u][nbr].get(weight, 1.0)), 1e-12)
            adjacent_u += w_u / math.sqrt(w_e * w_adj)

        adjacent_v = 0.0
        for nbr in G.neighbors(v):
            if nbr == u:
                continue
            w_adj = max(float(G[v][nbr].get(weight, 1.0)), 1e-12)
            adjacent_v += w_v / math.sqrt(w_e * w_adj)

        curvature = w_e * (
            w_u / w_e
            + w_v / w_e
            - adjacent_u
            - adjacent_v
        )
        data["ricciCurvature"] = float(curvature)
        data["ricci_engine"] = "forman"

    _attach_node_curvature(G)
    G.graph["curvature_engine"] = "forman"
    return G


def _attach_node_curvature(G: nx.Graph) -> None:
    """Attach mean incident-edge curvature to each node."""
    for node in G.nodes:
        values = [
            float(G[node][nbr].get("ricciCurvature", 0.0))
            for nbr in G.neighbors(node)
        ]
        G.nodes[node]["ricciCurvature"] = (
            float(np.mean(values)) if values else 0.0
        )


def compute_ricci_curvature(
    G: nx.Graph,
    alpha: float = 0.5,
    engine: CurvatureEngine = "ollivier_lp",
    weight: str = "weight",
) -> nx.Graph:
    """Dispatch to SciPy-LP Ollivier-Ricci or fast Forman-Ricci."""
    normalized = str(engine).strip().lower()
    aliases = {
        "ollivier": "ollivier_lp",
        "ollivier-lp": "ollivier_lp",
        "lp": "ollivier_lp",
        "forman-ricci": "forman",
    }
    normalized = aliases.get(normalized, normalized)

    if normalized == "ollivier_lp":
        return compute_ollivier_ricci_lp(G, alpha=alpha, weight=weight)
    if normalized == "forman":
        return compute_forman_ricci(G, weight=weight)
    raise ValueError(
        f"Unknown curvature engine {engine!r}; use 'ollivier_lp' or 'forman'."
    )
