import networkx as nx
import numpy as np

from ricci_finance.graph import (
    compute_forman_ricci,
    compute_ollivier_ricci_lp,
)


def weighted_path():
    graph = nx.path_graph(["A", "B", "C", "D"])
    for u, v in graph.edges:
        graph[u][v]["weight"] = 1.0
        graph[u][v]["distance"] = 1.0
        graph[u][v]["correlation"] = 0.8
    return graph


def test_ollivier_lp_is_finite():
    graph = compute_ollivier_ricci_lp(weighted_path(), alpha=0.5)
    values = [data["ricciCurvature"] for _, _, data in graph.edges(data=True)]
    assert len(values) == 3
    assert np.isfinite(values).all()
    assert graph.graph["curvature_engine"] == "ollivier_lp"


def test_forman_is_finite():
    graph = compute_forman_ricci(weighted_path())
    values = [data["ricciCurvature"] for _, _, data in graph.edges(data=True)]
    assert np.isfinite(values).all()
    assert graph.graph["curvature_engine"] == "forman"
