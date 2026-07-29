from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import networkx as nx


def financial_distance_from_corr(corr: pd.DataFrame) -> pd.DataFrame:
    clipped = corr.clip(-1.0, 1.0)
    values = np.sqrt(2.0 * (1.0 - clipped))
    return pd.DataFrame(values, index=corr.index, columns=corr.columns)


def _knn_candidates(corr: pd.DataFrame, k: int, min_corr: float):
    dist = financial_distance_from_corr(corr)
    found = {}
    for u in corr.columns:
        nearest = dist.loc[u].drop(index=u, errors="ignore").sort_values().head(max(1, k))
        for v, d in nearest.items():
            rho = float(corr.loc[u, v])
            if np.isfinite(rho) and rho >= min_corr:
                key = tuple(sorted((str(u), str(v))))
                found[key] = (key[0], key[1], float(d), rho)
    return list(found.values())


def build_graph_from_window(
    window_returns: pd.DataFrame,
    max_distance: float = 1.35,
    min_abs_corr: float = 0.10,
    keep_top_edges: Optional[int] = None,
    min_node_obs: int = 1,
    min_pair_obs: int = 4,
    graph_mode: str = "knn+bridges",
    k_neighbors: int = 3,
    min_corr: float = 0.05,
    max_bridges: int = 3,
) -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    min_pair_obs = max(3, int(min_pair_obs))
    node_cols = list(window_returns.columns)
    pair_cols = [c for c in node_cols if window_returns[c].notna().sum() >= min_pair_obs]
    clean = window_returns[pair_cols]
    corr = clean.corr(min_periods=min_pair_obs) if len(pair_cols) >= 2 else pd.DataFrame()
    corr = corr.replace([np.inf, -np.inf], np.nan)
    dist = financial_distance_from_corr(corr) if not corr.empty else pd.DataFrame()

    G = nx.Graph()
    G.add_nodes_from(node_cols)
    for c in node_cols:
        n = int(window_returns[c].notna().sum())
        G.nodes[c].update(valid_obs=n, is_active=n >= min_node_obs,
                          status="active" if n >= min_pair_obs else "waiting_for_data")

    mode = graph_mode.lower().replace(" ", "")
    candidates = []
    if mode in {"knn", "knn+bridges", "knnbridges"}:
        candidates = _knn_candidates(corr.fillna(0.0), k_neighbors, min_corr) if not corr.empty else []
    else:
        cols = list(corr.columns)
        for i, u in enumerate(cols):
            for v in cols[i+1:]:
                rho = float(corr.loc[u, v])
                d = float(dist.loc[u, v])
                if np.isfinite(rho) and np.isfinite(d) and abs(rho) >= min_abs_corr and d <= max_distance:
                    candidates.append((u, v, d, rho))

    if keep_top_edges:
        candidates = sorted(candidates, key=lambda x: x[2])[:int(keep_top_edges)]
    for u, v, d, rho in candidates:
        overlap = int(window_returns[[u, v]].dropna().shape[0])
        confidence = min(1.0, overlap / max(min_pair_obs * 3, 1))
        G.add_edge(u, v, weight=d, distance=d, correlation=rho,
                   overlap_n=overlap, confidence=confidence, bridge=False)

    if mode in {"knn+bridges", "knnbridges"} and max_bridges > 0 and not corr.empty:
        existing = {tuple(sorted(e)) for e in G.edges()}
        extras = []
        cols = list(corr.columns)
        for i, u in enumerate(cols):
            for v in cols[i+1:]:
                if tuple(sorted((u, v))) in existing:
                    continue
                rho = float(corr.loc[u, v])
                if np.isfinite(rho) and rho > 0:
                    d = float(np.sqrt(2 * (1 - np.clip(rho, -1, 1))))
                    extras.append((u, v, d, rho))
        for u, v, d, rho in sorted(extras, key=lambda x: x[2])[:max_bridges]:
            overlap = int(window_returns[[u, v]].dropna().shape[0])
            G.add_edge(u, v, weight=d, distance=d, correlation=rho,
                       overlap_n=overlap, confidence=min(1.0, overlap / 20), bridge=True)
    return G, corr, dist


def compute_components(G: nx.Graph) -> dict[str, int]:
    result = {}
    for cid, comp in enumerate(sorted(nx.connected_components(G), key=len, reverse=True)):
        for node in comp:
            result[node] = cid
    return result


def edge_jaccard(a: nx.Graph, b: nx.Graph) -> float:
    ea = {frozenset(e) for e in a.edges()}
    eb = {frozenset(e) for e in b.edges()}
    union = ea | eb
    return 1.0 if not union else len(ea & eb) / len(union)
