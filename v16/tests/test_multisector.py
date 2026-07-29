import networkx as nx
import numpy as np

from ricci_finance.sector_objects import build_profile, normalize_memberships
from ricci_finance.sectors import parse_sector_map, sector_flow_matrix
from ricci_finance.gnn import graph_to_dense


def test_normalize_memberships():
    result = normalize_memberships({"Semiconductors": 6, "AI": 3, "DataCenter": 1})
    assert np.isclose(sum(result.values()), 1.0)
    assert np.isclose(result["Semiconductors"], 0.6)


def test_parse_weighted_sector_map():
    result = parse_sector_map("NVDA=Semiconductors:6|AI:3|DataCenter:1", ["NVDA"])
    assert np.isclose(sum(result["NVDA"].values()), 1.0)
    assert result["NVDA"]["AI"] == 0.3


def test_automatic_profile_is_normalized():
    profile = build_profile(
        "NVDA",
        official_sector="Technology",
        official_industry="Semiconductors",
        description="AI data center GPU networking and autonomous driving products",
    )
    profile.validate()
    assert np.isclose(sum(profile.memberships.values()), 1.0)
    assert len(profile.memberships) >= 2


def test_sector_flow_is_conserved():
    graph = nx.Graph()
    graph.add_edge("A", "B", edge_capital_flow=100.0)
    sectors = {
        "A": {"AI": 0.6, "Semiconductors": 0.4},
        "B": {"Networking": 0.75, "AI": 0.25},
    }
    flow = sector_flow_matrix(graph, sectors)
    # Symmetric matrix records each undirected edge in both directions.
    assert np.isclose(flow.to_numpy().sum(), 200.0)


def test_gnn_uses_weighted_multihot():
    graph = nx.Graph()
    graph.add_node("NVDA", capital_share=0.5, ricciCurvature=0.1)
    graph.add_node("ANET", capital_share=0.5, ricciCurvature=0.2)
    graph.add_edge("NVDA", "ANET", correlation=0.8)
    memberships = {
        "NVDA": {"AI": 0.3, "Semiconductors": 0.7},
        "ANET": {"AI": 0.2, "Networking": 0.8},
    }
    vocab = {"AI": 0, "Networking": 1, "Semiconductors": 2}
    x, _ = graph_to_dense(graph, ["NVDA", "ANET"], memberships, vocab)
    assert np.isclose(float(x[0, 6 + vocab["AI"]]), 0.3)
    assert np.isclose(float(x[0, 6 + vocab["Semiconductors"]]), 0.7)
