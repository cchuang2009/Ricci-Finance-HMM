from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import networkx as nx
import numpy as np


@dataclass
class GNNResult:
    labels: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    accuracy: float
    balanced_accuracy: float
    losses: list[float]
    epochs: int
    device: str
    class_weights: np.ndarray
    note: str
    graph_embeddings: np.ndarray | None = None
    node_embeddings: np.ndarray | None = None
    attention: np.ndarray | None = None
    nodes: tuple[str, ...] = ()
    model_type: str = "GCN"


def _graph_to_dense_full(G, nodes, sectors, sector_vocab):
    import torch

    idx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)
    adjacency = np.zeros((n_nodes, n_nodes), np.float32)
    features = np.zeros((n_nodes, 6 + len(sector_vocab)), np.float32)

    for node, i in idx.items():
        if node in G:
            attrs = G.nodes[node]
            features[i, :6] = [
                G.degree(node),
                float(attrs.get("capital_share", 0)),
                float(attrs.get("ricciCurvature", 0)),
                1.0,
                0.0,
                0.0,
            ]
            memberships = sectors.get(str(node), {"Other": 1.0})
            if isinstance(memberships, str):
                memberships = {memberships: 1.0}
            total = sum(max(float(value), 0.0) for value in memberships.values())
            if total <= 0:
                memberships = {"Other": 1.0}
                total = 1.0
            for sector, value in memberships.items():
                if sector in sector_vocab:
                    features[i, 6 + sector_vocab[sector]] = max(float(value), 0.0) / total

    raw_adjacency = np.zeros_like(adjacency)
    for u, v, attrs in G.edges(data=True):
        if u in idx and v in idx:
            weight = abs(float(attrs.get("correlation", 1)))
            adjacency[idx[u], idx[v]] = weight
            adjacency[idx[v], idx[u]] = weight
            raw_adjacency[idx[u], idx[v]] = weight
            raw_adjacency[idx[v], idx[u]] = weight

    adjacency += np.eye(n_nodes, dtype=np.float32)
    degree = np.maximum(adjacency.sum(1), 1e-8)
    normalizer = np.diag(1 / np.sqrt(degree))
    normalized = normalizer @ adjacency @ normalizer
    return torch.tensor(features), torch.tensor(normalized), torch.tensor(raw_adjacency)


def graph_to_dense(G, nodes, sectors, sector_vocab):
    """Backward-compatible V15/V16 API returning features and normalized adjacency."""
    features, adjacency, _ = _graph_to_dense_full(G, nodes, sectors, sector_vocab)
    return features, adjacency


def _sector_vocab(nodes, sectors):
    names = set()
    for node in nodes:
        memberships = sectors.get(str(node), {"Other": 1.0})
        if isinstance(memberships, str):
            names.add(memberships)
        else:
            names.update(str(sector) for sector in memberships)
    return {sector: i for i, sector in enumerate(sorted(names or {"Other"}))}


def train_graph_regime(
    graphs: Sequence[nx.Graph],
    labels: Sequence[int],
    sectors: Mapping[str, object] | Sequence[Mapping[str, object]],
    *,
    model_type: str = "GCN",
    epochs: int = 120,
    hidden: int = 24,
    random_state: int = 42,
) -> GNNResult:
    """Train a compact dense GCN or GAT and return latent representations.

    ``sectors`` may be one static membership map or one map per graph frame.
    GAT returns the latest attention matrix for visualization.
    """
    import torch
    from torch import nn

    graphs = list(graphs)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(graphs) != len(y):
        raise ValueError(f"Graph/label length mismatch: graphs={len(graphs)}, labels={len(y)}")
    if len(graphs) < 10:
        raise ValueError("At least 10 graph snapshots are required")

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    nodes = sorted({node for graph in graphs for node in graph.nodes}, key=str)
    dynamic = isinstance(sectors, Sequence) and not isinstance(sectors, Mapping)
    sector_maps = list(sectors) if dynamic else [sectors] * len(graphs)
    if len(sector_maps) != len(graphs):
        raise ValueError("Dynamic sector membership history must match graph count")

    combined = {}
    for mapping in sector_maps:
        for ticker, memberships in mapping.items():
            combined.setdefault(str(ticker), {}).update(memberships if not isinstance(memberships, str) else {memberships: 1.0})
    vocabulary = _sector_vocab(nodes, combined)
    pairs = [_graph_to_dense_full(graph, nodes, membership, vocabulary) for graph, membership in zip(graphs, sector_maps)]

    classes = np.unique(y)
    remap = {label: i for i, label in enumerate(classes)}
    remapped = np.array([remap[label] for label in y])
    split = max(2, min(len(y) - 2, int(len(y) * 0.7)))
    train_idx = np.arange(split)
    test_idx = np.arange(split, len(y))
    counts = np.bincount(remapped[train_idx], minlength=len(classes))
    weights = len(train_idx) / (len(classes) * np.maximum(counts, 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class DenseGCN(nn.Module):
        def __init__(self, input_width, hidden_width, classes_count):
            super().__init__()
            self.w1 = nn.Linear(input_width, hidden_width)
            self.w2 = nn.Linear(hidden_width, hidden_width)
            self.out = nn.Linear(hidden_width, classes_count)

        def forward(self, x, adjacency, raw_adjacency):
            h = torch.relu(self.w1(adjacency @ x))
            h = torch.relu(self.w2(adjacency @ h))
            graph_embedding = h.mean(0)
            return self.out(graph_embedding), graph_embedding, h, None

    class DenseGAT(nn.Module):
        def __init__(self, input_width, hidden_width, classes_count):
            super().__init__()
            self.proj = nn.Linear(input_width, hidden_width, bias=False)
            self.query = nn.Linear(hidden_width, 1, bias=False)
            self.key = nn.Linear(hidden_width, 1, bias=False)
            self.out = nn.Linear(hidden_width, classes_count)

        def forward(self, x, adjacency, raw_adjacency):
            h = torch.tanh(self.proj(x))
            scores = self.query(h) + self.key(h).T
            mask = (raw_adjacency > 0) | torch.eye(raw_adjacency.shape[0], dtype=torch.bool, device=raw_adjacency.device)
            scores = scores.masked_fill(~mask, -1e9)
            attention = torch.softmax(scores, dim=1)
            node_embedding = torch.relu(attention @ h)
            graph_embedding = node_embedding.mean(0)
            return self.out(graph_embedding), graph_embedding, node_embedding, attention

    model_name = model_type.upper()
    model_class = DenseGAT if model_name == "GAT" else DenseGCN
    model = model_class(pairs[0][0].shape[1], hidden, len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    losses = []

    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        logits = torch.stack([
            model(x.to(device), adjacency.to(device), raw.to(device))[0]
            for x, adjacency, raw in pairs[:split]
        ])
        loss = loss_function(logits, torch.tensor(remapped[:split], device=device))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    logits_all = []
    graph_embeddings = []
    node_embeddings = []
    attentions = []
    model.eval()
    with torch.no_grad():
        for x, adjacency, raw in pairs:
            logits, graph_embedding, node_embedding, attention = model(
                x.to(device), adjacency.to(device), raw.to(device)
            )
            logits_all.append(logits)
            graph_embeddings.append(graph_embedding.cpu().numpy())
            node_embeddings.append(node_embedding.cpu().numpy())
            if attention is not None:
                attentions.append(attention.cpu().numpy())

        probabilities = torch.softmax(torch.stack(logits_all), 1).cpu().numpy()
        prediction_index = probabilities.argmax(1)

    predictions = np.array([classes[index] for index in prediction_index])
    accuracy = float((predictions[test_idx] == y[test_idx]).mean())
    recalls = []
    for label in np.unique(y[test_idx]):
        mask = y[test_idx] == label
        recalls.append(float((predictions[test_idx][mask] == label).mean()))
    balanced = float(np.mean(recalls)) if recalls else float("nan")

    return GNNResult(
        labels=y,
        predictions=predictions,
        probabilities=probabilities,
        train_indices=train_idx,
        test_indices=test_idx,
        accuracy=accuracy,
        balanced_accuracy=balanced,
        losses=losses,
        epochs=len(losses),
        device=device,
        class_weights=weights,
        note=(
            f"V16 dense {model_name} with normalized dynamic multi-sector features, "
            "chronological 70/30 split, class-weighted loss, graph/node embeddings"
            + (", and inspectable attention" if model_name == "GAT" else "")
        ),
        graph_embeddings=np.asarray(graph_embeddings),
        node_embeddings=np.asarray(node_embeddings),
        attention=np.asarray(attentions) if attentions else None,
        nodes=tuple(str(node) for node in nodes),
        model_type=model_name,
    )


def train_gcn_regime(
    graphs: Sequence[nx.Graph],
    labels: Sequence[int],
    sectors: Mapping[str, object] | Sequence[Mapping[str, object]],
    epochs: int = 120,
    hidden: int = 24,
    random_state: int = 42,
) -> GNNResult:
    return train_graph_regime(
        graphs,
        labels,
        sectors,
        model_type="GCN",
        epochs=epochs,
        hidden=hidden,
        random_state=random_state,
    )
