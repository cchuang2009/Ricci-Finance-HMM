from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import networkx as nx
import pandas as pd


@dataclass
class WindowStats:
    end_date: str
    avg_ricci: float
    ricci_std: float
    ricci_min: float
    ricci_max: float
    negative_edge_ratio: float
    num_clusters: int
    largest_component: int
    num_nodes: int
    num_edges: int
    density: float
    graph_entropy: float = 0.0
    edge_stability: float = float("nan")
    largest_component_ratio: float = 0.0
    total_node_capital: float = 0.0
    total_edge_capital_flow: float = 0.0
    avg_edge_capital_flow: float = 0.0
    max_node_capital_share: float = 0.0
    hmm_state: int = -1
    regime_name: str = "not computed"


@dataclass
class FrameData:
    G: nx.Graph
    node_cluster: Dict[str, int]
    stats: WindowStats
    corr: pd.DataFrame
    dist: pd.DataFrame
    metadata: dict = field(default_factory=dict)
