from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
from .models import FrameData, WindowStats
from .graph import build_graph_from_window, compute_components, edge_jaccard
from .ricci import compute_ricci_curvature
from .capital import attach_capital_attributes


def _entropy(component_sizes: list[int]) -> float:
    total = sum(component_sizes)
    if total <= 0:
        return 0.0
    p = np.asarray(component_sizes, dtype=float) / total
    return float(-(p[p > 0] * np.log(p[p > 0])).sum())


def compute_window_stats(G: nx.Graph, end_date: str) -> WindowStats:
    curv = np.asarray([float(d.get("ricciCurvature", 0.0)) for *_, d in G.edges(data=True)])
    comps = list(nx.connected_components(G))
    sizes = [len(c) for c in comps]
    node_cap = np.asarray([float(d.get("capital_mass", 0.0)) for _, d in G.nodes(data=True)])
    edge_cap = np.asarray([float(d.get("edge_capital_flow", 0.0)) for *_, d in G.edges(data=True)])
    total_node = float(node_cap.sum()) if len(node_cap) else 0.0
    return WindowStats(
        end_date=str(end_date),
        avg_ricci=float(curv.mean()) if len(curv) else 0.0,
        ricci_std=float(curv.std()) if len(curv) else 0.0,
        ricci_min=float(curv.min()) if len(curv) else 0.0,
        ricci_max=float(curv.max()) if len(curv) else 0.0,
        negative_edge_ratio=float((curv < 0).mean()) if len(curv) else 0.0,
        num_clusters=len(comps),
        largest_component=max(sizes, default=0),
        num_nodes=G.number_of_nodes(),
        num_edges=G.number_of_edges(),
        density=float(nx.density(G)) if G.number_of_nodes() > 1 else 0.0,
        graph_entropy=_entropy(sizes),
        largest_component_ratio=max(sizes, default=0) / max(G.number_of_nodes(), 1),
        total_node_capital=total_node,
        total_edge_capital_flow=float(edge_cap.sum()) if len(edge_cap) else 0.0,
        avg_edge_capital_flow=float(edge_cap.mean()) if len(edge_cap) else 0.0,
        max_node_capital_share=float(node_cap.max() / total_node) if total_node > 0 else 0.0,
    )


def build_frame(
    returns: pd.DataFrame,
    start: int,
    window_size: int = 60,
    dollar_volume: pd.DataFrame | None = None,
    use_capital_weighting: bool = True,
    capital_alpha: float = 0.35,
    **kwargs,
) -> FrameData:
    window = returns.iloc[start:start + window_size]
    graph_keys = {
        "max_distance", "min_abs_corr", "keep_top_edges", "min_node_obs",
        "min_pair_obs", "graph_mode", "k_neighbors", "min_corr", "max_bridges"
    }
    ricci_keys = {"alpha", "method", "proc"}
    graph_kwargs = {k: v for k, v in kwargs.items() if k in graph_keys}
    ricci_kwargs = {k: v for k, v in kwargs.items() if k in ricci_keys}
    G0, corr, dist = build_graph_from_window(window, **graph_kwargs)
    dv_window = None
    if dollar_volume is not None:
        dv_window = dollar_volume.reindex(index=window.index, columns=window.columns)
    G0 = attach_capital_attributes(
        G0, dv_window,
        capital_alpha=capital_alpha,
        use_capital_weighting=use_capital_weighting,
    )
    G = compute_ricci_curvature(G0, **ricci_kwargs)
    stats = compute_window_stats(G, str(window.index[-1])[:19])
    return FrameData(
        G=G,
        node_cluster=compute_components(G),
        stats=stats,
        corr=corr,
        dist=dist,
        metadata={"start": start, "window_size": window_size},
    )


def build_rolling_frames(
    returns: pd.DataFrame,
    window_size: int = 60,
    step: int = 5,
    max_frames: int = 40,
    **kwargs,
):
    if len(returns) < window_size:
        raise ValueError(f"Need at least {window_size} return rows, got {len(returns)}.")
    starts = list(range(0, len(returns) - window_size + 1, step))
    if len(starts) > max_frames:
        idx = np.linspace(0, len(starts) - 1, max_frames).astype(int)
        starts = [starts[i] for i in idx]
    frames = [build_frame(returns, s, window_size, **kwargs) for s in starts]
    for i in range(1, len(frames)):
        frames[i].stats.edge_stability = edge_jaccard(frames[i-1].G, frames[i].G)
    return frames, starts


def rolling_feature_table(frames: list[FrameData]) -> pd.DataFrame:
    rows = []
    for fd in frames:
        s = fd.stats
        rows.append({
            "date": s.end_date,
            "avg_ricci": s.avg_ricci,
            "ricci_std": s.ricci_std,
            "ricci_min": s.ricci_min,
            "ricci_max": s.ricci_max,
            "negative_edge_ratio": s.negative_edge_ratio,
            "clusters": s.num_clusters,
            "largest_component": s.largest_component,
            "largest_component_ratio": s.largest_component_ratio,
            "nodes": s.num_nodes,
            "edges": s.num_edges,
            "density": s.density,
            "component_entropy": s.graph_entropy,
            "edge_stability": s.edge_stability,
            "total_node_capital": s.total_node_capital,
            "total_edge_capital_flow": s.total_edge_capital_flow,
            "avg_edge_capital_flow": s.avg_edge_capital_flow,
            "max_node_capital_share": s.max_node_capital_share,
            "hmm_state": s.hmm_state,
            "regime_name": s.regime_name,
        })
    return pd.DataFrame(rows)
