import networkx as nx
import numpy as np

from ricci_finance.dynamic import (
    build_dynamic_sector_history,
    ricci_communities,
)
from ricci_finance.gnn import train_graph_regime


def make_graph(scale=1.0):
    graph = nx.Graph()
    graph.add_node("A", capital_share=.6, ricciCurvature=.2)
    graph.add_node("B", capital_share=.3, ricciCurvature=-.1)
    graph.add_node("C", capital_share=.1, ricciCurvature=.05)
    graph.add_edge("A", "B", correlation=.8 * scale)
    graph.add_edge("B", "C", correlation=.5 * scale)
    return graph


def test_dynamic_memberships_normalize_and_evolve():
    frames = [{"graph": make_graph(.5), "date": "2026-01-01"}, {"graph": make_graph(1.0), "date": "2026-01-02"}]
    static = {"A": {"AI": 1}, "B": {"Cloud": 1}, "C": {"Photonics": 1}}
    history = build_dynamic_sector_history(frames, static)
    assert len(history) == 2
    assert all(np.isclose(sum(weights.values()), 1.0) for frame in history for weights in frame.values())
    assert history[0]["A"] == history[1]["A"]  # correlation scaling preserves relative affinity


def test_communities_cover_nodes():
    graph = make_graph()
    labels = ricci_communities(graph)
    assert set(labels) == {"A", "B", "C"}


def test_gat_returns_attention_and_embeddings():
    graphs = [make_graph(0.5 + i / 20) for i in range(12)]
    labels = [i % 2 for i in range(12)]
    memberships = {"A": {"AI": 1}, "B": {"Cloud": 1}, "C": {"Photonics": 1}}
    result = train_graph_regime(graphs, labels, memberships, model_type="GAT", epochs=2, hidden=8)
    assert result.graph_embeddings.shape[0] == 12
    assert result.node_embeddings.shape[:2] == (12, 3)
    assert result.attention.shape == (12, 3, 3)
