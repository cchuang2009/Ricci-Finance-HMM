from __future__ import annotations
from typing import Iterable
import networkx as nx
from .models import FrameData


def build_base_graph_for_layout(frames: list[FrameData], all_nodes: Iterable[str] | None = None) -> nx.Graph:
    base = nx.Graph()
    if all_nodes is not None:
        base.add_nodes_from(all_nodes)
    for fd in frames:
        base.add_nodes_from(fd.G.nodes())
        for u, v, d in fd.G.edges(data=True):
            w = float(d.get("weight", 1.0))
            if base.has_edge(u, v):
                base[u][v]["weight"] = min(base[u][v]["weight"], w)
            else:
                base.add_edge(u, v, weight=w)
    return base


def compute_stable_layout(
    base_graph: nx.Graph,
    seed: int = 42,
    scale: float = 1.0,
    layout_k: float = 0.45,
    iterations: int = 600,
):
    if base_graph.number_of_nodes() == 0:
        return {}
    pos = (
        nx.circular_layout(base_graph, scale=scale)
        if base_graph.number_of_edges() == 0
        else nx.spring_layout(base_graph, seed=seed, weight="weight",
                              k=layout_k, iterations=iterations, scale=scale)
    )
    return {n: tuple(map(float, xy)) for n, xy in pos.items()}
