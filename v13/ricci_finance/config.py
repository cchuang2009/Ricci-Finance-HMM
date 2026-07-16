from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class DataConfig:
    period: str = "1y"
    interval: str = "1d"
    auto_adjust: bool = False


@dataclass(frozen=True)
class GraphConfig:
    window_size: int = 60
    step: int = 5
    max_frames: int = 40
    graph_mode: str = "knn+bridges"
    k_neighbors: int = 3
    min_corr: float = 0.05
    min_abs_corr: float = 0.10
    max_distance: float = 1.35
    max_bridges: int = 3
    keep_top_edges: Optional[int] = None
    min_node_obs: int = 1
    min_pair_obs: int = 4


@dataclass(frozen=True)
class RicciConfig:
    alpha: float = 0.5
    method: str = "OTD"
    proc: int = 1
    flow_iterations: int = 8
    flow_step: float = 0.25
    normalize_flow: bool = True


@dataclass(frozen=True)
class HMMConfig:
    enabled: bool = True
    n_components: int = 3
    forward_days: int = 5
    random_state: int = 42
    min_covar: float = 1e-4


@dataclass(frozen=True)
class ModelConfig:
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    ricci: RicciConfig = field(default_factory=RicciConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
