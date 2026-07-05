"""
helper.py - Ricci Finance v9

Graduate/research helpers for Perelman-style Ricci theory in financial networks.

v9 layers
---------
1. Correlation-distance financial graph.
2. Ollivier-Ricci curvature, with a didactic fallback when GraphRicciCurvature is absent.
3. Discrete Ricci flow on graph edge distances.
4. Perelman-inspired financial surgery: singular-edge detection, cut, and topology diagnostics.
5. Rolling-frame animation helpers and HMM hidden-regime diagnostics.

Install:
    pip install streamlit yfinance pandas numpy networkx plotly matplotlib scikit-learn hmmlearn
Optional true Ollivier-Ricci:
    pip install GraphRicciCurvature pot networkit
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import matplotlib as mpl

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

try:
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
except Exception:  # pragma: no cover
    OllivierRicci = None

try:
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    GaussianHMM = None
    StandardScaler = None


DEFAULT_TICKERS = [
    "NVDA", "AMD", "AVGO", "TSM", "MU", "MRVL", "AMAT", "LRCX", "KLAC",
    "ANET", "AAOI", "COHR", "LITE", "SMCI", "PLTR", "IONQ", "QBTS", "QUBT",
    "RGTI", "NBIS", "QNT",
]

COMPONENT_COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
]


@dataclass
class WindowStats:
    end_date: str
    avg_ricci: float
    ricci_std: float
    ricci_min: float
    ricci_max: float
    num_clusters: int
    largest_component: int
    num_nodes: int
    num_edges: int
    density: float
    graph_entropy: float
    negative_edge_ratio: float
    hmm_state: int = -1
    regime_name: str = "not computed"


@dataclass
class FrameData:
    G: nx.Graph
    node_cluster: Dict[str, int]
    stats: WindowStats
    corr: pd.DataFrame
    dist: pd.DataFrame


@dataclass
class SurgeryResult:
    before: nx.Graph
    after: nx.Graph
    removed_edges: List[Tuple[str, str, float, str]]
    report: pd.DataFrame
    before_stats: Dict[str, float]
    after_stats: Dict[str, float]


def parse_tickers(text_or_list: str | Sequence[str]) -> List[str]:
    if isinstance(text_or_list, str):
        raw_items = text_or_list.replace("\n", ",").split(",")
    else:
        raw_items = list(text_or_list)
    out: List[str] = []
    for item in raw_items:
        ticker = str(item).strip().upper()
        if ticker:
            out.append(ticker)
    return list(dict.fromkeys(out))


def download_prices(tickers: Sequence[str], period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")
    tickers = parse_tickers(tickers)
    data = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=True,
    )
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        elif "Adj Close" in data.columns.get_level_values(0):
            prices = data["Adj Close"].copy()
        else:
            prices = data[data.columns.get_level_values(0)[0]].copy()
    else:
        close_col = "Close" if "Close" in data.columns else data.columns[0]
        prices = data[[close_col]].copy()
        prices.columns = [tickers[0]]
    return prices.dropna(axis=1, how="all").ffill().dropna(how="all")


def make_demo_prices(
    tickers,
    n_days=300,
    seed=42,
    ipo_tickers=("QNT",),
    ipo_start_frac=0.60,
):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)

    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days)

    market = rng.normal(0, 0.01, n_days)

    prices = {}

    ai_factor = rng.normal(0, 0.012, n_days)
    quantum_factor = rng.normal(0, 0.018, n_days)
    optical_factor = rng.normal(0, 0.014, n_days)

    ai_names = {
        "NVDA", "AMD", "AVGO", "MU", "MRVL",
        "AMAT", "LRCX", "KLAC", "SMCI"
    }

    quantum_names = {
        "IONQ", "QBTS", "QUBT", "RGTI", "QNT", "BNT"
    }

    optical_names = {
        "AAOI", "LITE", "COHR", "CIEN"
    }

    ipo_start_idx = int(n_days * ipo_start_frac)

    for t in tickers:

        noise = rng.normal(0, 0.01, n_days)

        if t in ai_names:
            ret = 0.55 * market + 0.45 * ai_factor + noise

        elif t in quantum_names:
            ret = 0.35 * market + 0.65 * quantum_factor + noise

        elif t in optical_names:
            ret = 0.45 * market + 0.55 * optical_factor + noise

        else:
            ret = 0.70 * market + noise

        px = 100 * np.exp(np.cumsum(ret))

        s = pd.Series(px, index=dates)

        # IPO-aware late appearance
        if t in ipo_tickers:
            s.iloc[:ipo_start_idx] = np.nan

        prices[t] = s

    return pd.DataFrame(prices)

def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.replace([np.inf, -np.inf], np.nan)
    returns = np.log(prices / prices.shift(1))
    return returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")


def financial_distance_from_corr(corr: pd.DataFrame) -> pd.DataFrame:
    clipped = corr.clip(lower=-1.0, upper=1.0)
    dist = np.sqrt(2.0 * (1.0 - clipped))
    return pd.DataFrame(dist, index=corr.index, columns=corr.columns)


def build_graph_from_window(
    window_returns: pd.DataFrame,
    max_distance: float = 1.05,
    min_abs_corr: float = 0.30,
    keep_top_edges: Optional[int] = None,
    min_node_obs: int = 1,
    min_pair_obs: int = 4,
    edge_mode: str = "threshold",
    knn_k: int = 3,
    positive_only: bool = False,
) -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    """Build correlation-distance graph.

    edge_mode="threshold": keep edges passing filters.
    edge_mode="knn": each node keeps k nearest neighbors, useful for sensible market maps.
    edge_mode="hybrid": threshold edges plus kNN anti-isolation edges.
    """
    min_pair_obs = max(3, int(min_pair_obs))
    min_node_obs = max(1, int(min_node_obs))
    node_cols = [c for c in window_returns.columns if window_returns[c].notna().sum() >= min_node_obs]
    pair_cols = [c for c in window_returns.columns if window_returns[c].notna().sum() >= min_pair_obs]
    clean = window_returns[pair_cols].copy()
    if len(pair_cols) >= 2:
        corr = clean.corr(min_periods=min_pair_obs).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        corr = pd.DataFrame(index=pair_cols, columns=pair_cols, dtype=float)
    dist = financial_distance_from_corr(corr)
    G = nx.Graph()
    G.add_nodes_from(node_cols)

    def add_edge(u: str, v: str, mode_label: str) -> None:
        rho = float(corr.loc[u, v])
        d = float(dist.loc[u, v])
        if not np.isfinite(rho) or not np.isfinite(d):
            return
        if positive_only and rho < 0:
            return
        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=d, distance=d, correlation=rho, edge_source=mode_label)

    candidates: List[Tuple[str, str, float, float]] = []
    tickers = list(corr.columns)
    for i, u in enumerate(tickers):
        for j in range(i + 1, len(tickers)):
            v = tickers[j]
            rho = float(corr.iloc[i, j])
            d = float(dist.iloc[i, j])
            pass_corr = (rho >= min_abs_corr) if positive_only else (abs(rho) >= min_abs_corr)
            if np.isfinite(rho) and np.isfinite(d) and d <= max_distance and pass_corr:
                candidates.append((u, v, d, rho))
    if keep_top_edges is not None and keep_top_edges > 0:
        candidates = sorted(candidates, key=lambda x: x[2])[: int(keep_top_edges)]

    if edge_mode in {"threshold", "hybrid"}:
        for u, v, _, _ in candidates:
            add_edge(u, v, "threshold")

    if edge_mode in {"knn", "hybrid"} and len(corr.columns) >= 2:
        for u in corr.columns:
            neighbors = dist.loc[u].drop(index=u).sort_values().head(max(1, int(knn_k)))
            for v in neighbors.index:
                add_edge(u, str(v), "knn" if edge_mode == "knn" else "hybrid_knn")

    return G, corr, dist


def compute_ricci_curvature(
    G: nx.Graph,
    alpha: float = 0.5,
    method: str = "OTD",
    proc: int = 1,
    fallback: str = "forman_like",
) -> nx.Graph:
    H = G.copy()
    if H.number_of_edges() == 0:
        return H
    if OllivierRicci is not None:
        try:
            orc = OllivierRicci(H, alpha=alpha, method=method, weight="weight", proc=proc, verbose="ERROR")
            orc.compute_ricci_curvature()
            R = orc.G.copy()
            for u, v in R.edges():
                R[u][v]["ricciCurvature"] = float(R[u][v].get("ricciCurvature", 0.0))
            return R
        except Exception:
            pass
    # fallback: triangle redundancy minus bridge-likeness
    for u, v in H.edges():
        cn = len(list(nx.common_neighbors(H, u, v)))
        deg_sum = max(1, H.degree(u) + H.degree(v) - 2)
        H[u][v]["ricciCurvature"] = float(2.0 * cn / deg_sum - 0.5)
    return H


def compute_components(G: nx.Graph) -> Dict[str, int]:
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    out: Dict[str, int] = {}
    for cid, comp in enumerate(components):
        for node in comp:
            out[node] = cid
    return out


def component_entropy(G: nx.Graph) -> float:
    sizes = np.array([len(c) for c in nx.connected_components(G)], dtype=float)
    if sizes.size == 0:
        return 0.0
    p = sizes / sizes.sum()
    return max(0.0, float(-(p * np.log(p + 1e-12)).sum()))


def compute_window_stats(G: nx.Graph, end_date: str) -> WindowStats:
    vals = np.array([float(data.get("ricciCurvature", 0.0)) for _, _, data in G.edges(data=True)], dtype=float)
    comps = list(nx.connected_components(G))
    return WindowStats(
        end_date=str(end_date),
        avg_ricci=float(vals.mean()) if vals.size else 0.0,
        ricci_std=float(vals.std()) if vals.size else 0.0,
        ricci_min=float(vals.min()) if vals.size else 0.0,
        ricci_max=float(vals.max()) if vals.size else 0.0,
        num_clusters=len(comps),
        largest_component=max((len(c) for c in comps), default=0),
        num_nodes=G.number_of_nodes(),
        num_edges=G.number_of_edges(),
        density=float(nx.density(G)) if G.number_of_nodes() > 1 else 0.0,
        graph_entropy=component_entropy(G),
        negative_edge_ratio=float(np.mean(vals < 0)) if vals.size else 0.0,
    )


def build_frame(
    returns: pd.DataFrame,
    start: int,
    window_size: int = 60,
    max_distance: float = 1.05,
    min_abs_corr: float = 0.30,
    keep_top_edges: Optional[int] = None,
    alpha: float = 0.5,
    method: str = "OTD",
    proc: int = 1,
    min_node_obs: int = 1,
    min_pair_obs: int = 4,
    edge_mode: str = "threshold",
    knn_k: int = 3,
    positive_only: bool = False,
) -> FrameData:
    window_returns = returns.iloc[start : start + window_size]
    end_date = str(window_returns.index[-1])[:19]
    G0, corr, dist = build_graph_from_window(
        window_returns, max_distance=max_distance, min_abs_corr=min_abs_corr,
        keep_top_edges=keep_top_edges, min_node_obs=min_node_obs, min_pair_obs=min_pair_obs,
        edge_mode=edge_mode, knn_k=knn_k, positive_only=positive_only,
    )
    G = compute_ricci_curvature(G0, alpha=alpha, method=method, proc=proc)
    return FrameData(G=G, node_cluster=compute_components(G), stats=compute_window_stats(G, end_date), corr=corr, dist=dist)


def build_rolling_frames(
    returns: pd.DataFrame,
    window_size: int = 60,
    step: int = 5,
    max_frames: int = 40,
    **kwargs,
) -> Tuple[List[FrameData], List[int]]:
    if len(returns) < window_size:
        raise ValueError(f"Need at least {window_size} return rows, got {len(returns)}")
    starts = list(range(0, len(returns) - window_size + 1, step))
    if len(starts) > max_frames:
        chosen = np.linspace(0, len(starts) - 1, int(max_frames)).astype(int)
        starts = [starts[i] for i in chosen]
    return [build_frame(returns, start=s, window_size=window_size, **kwargs) for s in starts], starts


def build_base_graph_for_layout(frames: Sequence[FrameData], all_nodes: Optional[Iterable[str]] = None) -> nx.Graph:
    base = nx.Graph()
    if all_nodes is not None:
        base.add_nodes_from(list(all_nodes))
    for fd in frames:
        base.add_nodes_from(fd.G.nodes())
        for u, v, data in fd.G.edges(data=True):
            d = float(data.get("weight", 1.0))
            if base.has_edge(u, v):
                base[u][v]["weight"] = min(float(base[u][v].get("weight", d)), d)
            else:
                base.add_edge(u, v, weight=d)
    return base


def compute_stable_layout(base_graph: nx.Graph, seed: int = 42, scale: float = 1.0) -> Dict[str, Tuple[float, float]]:
    if base_graph.number_of_nodes() == 0:
        return {}
    if base_graph.number_of_edges() == 0:
        pos = nx.circular_layout(base_graph, scale=scale)
    else:
        pos = nx.spring_layout(base_graph, seed=seed, weight="weight", iterations=400, scale=scale)
    return {n: (float(x), float(y)) for n, (x, y) in pos.items()}


def ricci_to_hex(kappa: float, vmin: float = -0.6, vmax: float = 0.6) -> str:
    cmap = mpl.colormaps["coolwarm_r"]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    return mpl.colors.to_hex(cmap(norm(float(kappa))))


def visualize_network(
    G: nx.Graph,
    positions: Optional[Dict[str, Tuple[float, float]]] = None,
    title: str = "Ricci financial network",
    node_cluster: Optional[Dict[str, int]] = None,
    node_size_base: int = 24,
    highlight_edges: Optional[Sequence[Tuple[str, str]]] = None,
    show_edge_weight_labels: bool = True,
    edge_label_top_n: int = 30,
) -> go.Figure:
    if positions is None:
        positions = compute_stable_layout(G)
    if node_cluster is None:
        node_cluster = compute_components(G)
    highlight = {tuple(sorted(e)) for e in (highlight_edges or [])}
    edge_traces: List[go.Scatter] = []
    label_x: List[float] = []
    label_y: List[float] = []
    label_text: List[str] = []
    label_hover: List[str] = []
    labeled_edges = sorted(
        G.edges(data=True),
        key=lambda e: float(e[2].get("distance", e[2].get("weight", np.inf))),
    )[: max(0, int(edge_label_top_n))]
    labeled_pairs = {tuple(sorted((u, v))) for u, v, _ in labeled_edges} if show_edge_weight_labels else set()

    for u, v, data in G.edges(data=True):
        x0, y0 = positions.get(u, (0.0, 0.0)); x1, y1 = positions.get(v, (0.0, 0.0))
        kappa = float(data.get("ricciCurvature", 0.0))
        rho = float(data.get("correlation", np.nan))
        d = float(data.get("distance", data.get("weight", np.nan)))
        is_highlight = tuple(sorted((u, v))) in highlight
        edge_traces.append(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line={"width": 5.5 if is_highlight else 2.0, "color": "black" if is_highlight else ricci_to_hex(kappa)},
            hoverinfo="text",
            text=f"{u}-{v}<br>weight / distance={d:.4f}<br>correlation={rho:.4f}<br>Ricci={kappa:.4f}<br>source={data.get('edge_source','')}",
            showlegend=False,
        ))
        if tuple(sorted((u, v))) in labeled_pairs:
            label_x.append((x0 + x1) / 2.0)
            label_y.append((y0 + y1) / 2.0)
            label_text.append(f"w={d:.2f}")
            label_hover.append(f"{u}-{v}<br>weight / distance={d:.4f}<br>correlation={rho:.4f}<br>Ricci={kappa:.4f}")

    if show_edge_weight_labels:
        edge_traces.append(go.Scatter(
            x=label_x, y=label_y, mode="text", text=label_text,
            textfont={"size": 11, "color": "#111111"},
            hoverinfo="text", hovertext=label_hover, showlegend=False,
        ))
    degrees = dict(G.degree())
    node_x: List[float] = []; node_y: List[float] = []; labels: List[str] = []; hover: List[str] = []; colors: List[str] = []; sizes: List[int] = []
    for node in sorted(G.nodes()):
        x, y = positions.get(node, (0.0, 0.0)); cid = node_cluster.get(node, 0)
        node_x.append(x); node_y.append(y); labels.append(node)
        hover.append(f"Ticker: {node}<br>Component: {cid}<br>Degree: {degrees.get(node, 0)}")
        colors.append(COMPONENT_COLORS[cid % len(COMPONENT_COLORS)])
        sizes.append(node_size_base + 2 * degrees.get(node, 0))
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=labels, textposition="top center",
        marker={"size": sizes, "color": colors, "line": {"width": 1.2, "color": "#111"}},
        hoverinfo="text", hovertext=hover, showlegend=False,
    )
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=title, height=650, hovermode="closest", showlegend=False,
        xaxis={"showgrid": False, "zeroline": False, "visible": False},
        yaxis={"showgrid": False, "zeroline": False, "visible": False, "scaleanchor": "x", "scaleratio": 1},
        margin={"l": 20, "r": 20, "t": 50, "b": 20}, plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def edge_universe(frames: Sequence[FrameData]) -> List[Tuple[str, str]]:
    edges = set()
    for fd in frames:
        for u, v in fd.G.edges():
            edges.add(tuple(sorted((u, v))))
    return sorted(edges)


def stats_annotation(stats: WindowStats) -> dict:
    return {
        "text": (
            f"<b>Date:</b> {stats.end_date}<br>"
            f"<b>Avg Ricci:</b> {stats.avg_ricci:.4f}<br>"
            f"<b>Ricci min:</b> {stats.ricci_min:.4f}<br>"
            f"<b>Clusters:</b> {stats.num_clusters}<br>"
            f"<b>Nodes:</b> {stats.num_nodes} &nbsp; <b>Edges:</b> {stats.num_edges}<br>"
            f"<b>Density:</b> {stats.density:.3f}<br>"
            f"<b>Entropy:</b> {stats.graph_entropy:.3f}<br>"
            f"<b>Negative-edge ratio:</b> {stats.negative_edge_ratio:.2f}<br>"
            f"<b>HMM:</b> {stats.hmm_state} - {stats.regime_name}"
        ),
        "xref": "paper", "yref": "paper", "x": 1.02, "y": 0.98,
        "showarrow": False, "align": "left", "bordercolor": "#999", "borderwidth": 1,
        "bgcolor": "rgba(255,255,255,0.92)", "font": {"size": 13, "color": "#111"},
    }


def _frame_traces(
    fd: FrameData,
    positions: Dict[str, Tuple[float, float]],
    all_edges: List[Tuple[str, str]],
    node_label_size: int = 14,
    node_size_base: int = 24,
    edge_width_scale: float = 5.0,
    show_edge_weight_labels: bool = True,
    edge_label_top_n: int = 30,
) -> List[go.Scatter]:
    traces: List[go.Scatter] = []
    G = fd.G
    label_x: List[float] = []
    label_y: List[float] = []
    label_text: List[str] = []
    label_hover: List[str] = []
    current_edges_by_distance = sorted(
        G.edges(data=True),
        key=lambda e: float(e[2].get("distance", e[2].get("weight", np.inf))),
    )[: max(0, int(edge_label_top_n))]
    labeled_pairs = {tuple(sorted((u, v))) for u, v, _ in current_edges_by_distance} if show_edge_weight_labels else set()

    for u, v in all_edges:
        if G.has_edge(u, v):
            data = G[u][v]
            x0, y0 = positions.get(u, (0.0, 0.0)); x1, y1 = positions.get(v, (0.0, 0.0))
            kappa = float(data.get("ricciCurvature", 0.0)); rho = float(data.get("correlation", np.nan)); d = float(data.get("distance", data.get("weight", np.nan)))
            traces.append(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line={"width": 0.8 + edge_width_scale * min(1.0, abs(kappa)), "color": ricci_to_hex(kappa)}, hoverinfo="text", text=f"{u}-{v}<br>weight / distance={d:.4f}<br>correlation={rho:.4f}<br>Ricci={kappa:.4f}", showlegend=False))
            if tuple(sorted((u, v))) in labeled_pairs:
                label_x.append((x0 + x1) / 2.0)
                label_y.append((y0 + y1) / 2.0)
                label_text.append(f"w={d:.2f}")
                label_hover.append(f"{u}-{v}<br>weight / distance={d:.4f}<br>correlation={rho:.4f}<br>Ricci={kappa:.4f}")
        else:
            traces.append(go.Scatter(x=[None, None], y=[None, None], mode="lines", line={"width": 0, "color": "rgba(0,0,0,0)"}, hoverinfo="skip", showlegend=False))

    traces.append(go.Scatter(
        x=label_x, y=label_y, mode="text", text=label_text,
        textfont={"size": 11, "color": "#111111"},
        hoverinfo="text", hovertext=label_hover, showlegend=False,
    ))
    degrees = dict(G.degree())
    node_x=[]; node_y=[]; labels=[]; hover=[]; colors=[]; sizes=[]
    for node in sorted(G.nodes()):
        x, y = positions.get(node, (0.0, 0.0)); cid = fd.node_cluster.get(node, 0)
        node_x.append(x); node_y.append(y); labels.append(node); hover.append(f"Ticker: {node}<br>Component: {cid}<br>Degree: {degrees.get(node,0)}")
        colors.append(COMPONENT_COLORS[cid % len(COMPONENT_COLORS)]); sizes.append(node_size_base + 2 * degrees.get(node, 0))
    traces.append(go.Scatter(x=node_x, y=node_y, mode="markers+text", text=labels, textposition="top center", textfont={"size": int(node_label_size), "color": "#111"}, hoverinfo="text", hovertext=hover, marker={"size": sizes, "color": colors, "line": {"width": 1.5, "color": "#111"}}, showlegend=False))
    return traces


def build_plotly_animation(
    frames: Sequence[FrameData],
    positions: Dict[str, Tuple[float, float]],
    frame_duration_ms: int = 700,
    node_label_size: int = 14,
    node_size_base: int = 24,
    edge_width_scale: float = 5.0,
    title: str = "Rolling Ricci Finance v9",
    show_edge_weight_labels: bool = True,
    edge_label_top_n: int = 30,
) -> go.Figure:
    if not frames:
        return go.Figure()
    all_edges = edge_universe(frames)
    fig = go.Figure(data=_frame_traces(frames[0], positions, all_edges, node_label_size, node_size_base, edge_width_scale, show_edge_weight_labels, edge_label_top_n))
    fig.frames = [go.Frame(name=str(i), data=_frame_traces(fd, positions, all_edges, node_label_size, node_size_base, edge_width_scale, show_edge_weight_labels, edge_label_top_n), layout=go.Layout(annotations=[stats_annotation(fd.stats)])) for i, fd in enumerate(frames)]
    steps = [{"method": "animate", "label": str(i+1), "args": [[str(i)], {"mode": "immediate", "frame": {"duration": frame_duration_ms, "redraw": True}, "transition": {"duration": 0}}]} for i in range(len(frames))]
    fig.update_layout(
        title=title, height=780, showlegend=False, hovermode="closest", annotations=[stats_annotation(frames[0].stats)],
        xaxis={"showgrid": False, "zeroline": False, "visible": False}, yaxis={"showgrid": False, "zeroline": False, "visible": False, "scaleanchor": "x", "scaleratio": 1},
        margin={"l": 20, "r": 230, "t": 60, "b": 70}, plot_bgcolor="white", paper_bgcolor="white",
        updatemenus=[{"type": "buttons", "showactive": False, "x": 0.02, "y": -0.05, "xanchor": "left", "yanchor": "top", "buttons": [
            {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": frame_duration_ms, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True}]},
            {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
        ]}], sliders=[{"active": 0, "currentvalue": {"prefix": "Frame ", "font": {"size": 14}}, "pad": {"t": 40}, "steps": steps}],
    )
    return fig


def graph_topology_stats(G: nx.Graph) -> Dict[str, float]:
    comps = list(nx.connected_components(G))
    return {
        "nodes": float(G.number_of_nodes()),
        "edges": float(G.number_of_edges()),
        "clusters": float(len(comps)),
        "largest_component": float(max((len(c) for c in comps), default=0)),
        "density": float(nx.density(G)) if G.number_of_nodes() > 1 else 0.0,
        "component_entropy": component_entropy(G),
    }


def run_ricci_flow(
    G: nx.Graph,
    iterations: int = 8,
    step_size: float = 0.25,
    alpha: float = 0.5,
    method: str = "OTD",
    proc: int = 1,
    normalize_mean_weight: bool = True,
    min_weight: float = 1e-4,
    max_weight: float = 10.0,
) -> Tuple[nx.Graph, pd.DataFrame]:
    """Discrete graph Ricci flow. Recomputes curvature after each edge-distance update."""
    H = compute_ricci_curvature(G, alpha=alpha, method=method, proc=proc)
    initial_mean = np.mean([float(d.get("weight", 1.0)) for _, _, d in H.edges(data=True)]) if H.number_of_edges() else 1.0
    rows = []
    for it in range(int(iterations) + 1):
        vals = [float(d.get("ricciCurvature", 0.0)) for _, _, d in H.edges(data=True)]
        weights = [float(d.get("weight", d.get("distance", 1.0))) for _, _, d in H.edges(data=True)]
        rows.append({
            "iteration": it,
            "avg_ricci": float(np.mean(vals)) if vals else 0.0,
            "min_ricci": float(np.min(vals)) if vals else 0.0,
            "max_ricci": float(np.max(vals)) if vals else 0.0,
            "ricci_std": float(np.std(vals)) if vals else 0.0,
            "mean_distance": float(np.mean(weights)) if weights else 0.0,
            "clusters": len(list(nx.connected_components(H))),
            "component_entropy": component_entropy(H),
        })
        if it == int(iterations):
            break
        for u, v, data in H.edges(data=True):
            w = float(data.get("weight", data.get("distance", 1.0)))
            kappa = float(data.get("ricciCurvature", 0.0))
            new_w = np.clip(w * (1.0 - float(step_size) * kappa), min_weight, max_weight)
            H[u][v]["weight"] = float(new_w)
            H[u][v]["distance"] = float(new_w)
        if normalize_mean_weight and H.number_of_edges() > 0:
            cur_mean = np.mean([float(d.get("weight", 1.0)) for _, _, d in H.edges(data=True)])
            if cur_mean > 0:
                scale = initial_mean / cur_mean
                for _, _, data in H.edges(data=True):
                    data["weight"] = float(np.clip(data["weight"] * scale, min_weight, max_weight))
                    data["distance"] = data["weight"]
        H = compute_ricci_curvature(H, alpha=alpha, method=method, proc=proc)
    return H, pd.DataFrame(rows)


def detect_singular_edges(
    G: nx.Graph,
    curvature_threshold: float = -0.35,
    distance_quantile: float = 0.80,
    use_bridge_test: bool = True,
) -> pd.DataFrame:
    """Detect edges that behave like financial neck-pinches.

    An edge is singular if curvature is very negative; optionally it is also long or a bridge.
    """
    if G.number_of_edges() == 0:
        return pd.DataFrame(columns=["u", "v", "distance", "correlation", "ricciCurvature", "is_bridge", "reason"])
    distances = np.array([float(d.get("distance", d.get("weight", np.nan))) for _, _, d in G.edges(data=True)], dtype=float)
    dist_cut = float(np.nanquantile(distances, distance_quantile)) if np.isfinite(distances).any() else np.inf
    bridges = {tuple(sorted(e)) for e in nx.bridges(G)} if use_bridge_test else set()
    rows = []
    for u, v, data in G.edges(data=True):
        k = float(data.get("ricciCurvature", 0.0)); d = float(data.get("distance", data.get("weight", np.nan))); rho = float(data.get("correlation", np.nan))
        is_bridge = tuple(sorted((u, v))) in bridges
        reasons = []
        if k <= curvature_threshold:
            reasons.append("negative_curvature")
        if np.isfinite(d) and d >= dist_cut:
            reasons.append("long_distance")
        if is_bridge:
            reasons.append("topological_bridge")
        singular = (k <= curvature_threshold) and ((np.isfinite(d) and d >= dist_cut) or is_bridge or not use_bridge_test)
        if singular:
            rows.append({"u": u, "v": v, "distance": d, "correlation": rho, "ricciCurvature": k, "is_bridge": is_bridge, "reason": "+".join(reasons)})
    return pd.DataFrame(rows).sort_values("ricciCurvature") if rows else pd.DataFrame(columns=["u", "v", "distance", "correlation", "ricciCurvature", "is_bridge", "reason"])


def perform_financial_surgery(
    G: nx.Graph,
    curvature_threshold: float = -0.35,
    distance_quantile: float = 0.80,
    use_bridge_test: bool = True,
    remove_isolated_nodes: bool = False,
) -> SurgeryResult:
    """Cut singular edges; this is the finance analogue of Ricci-flow surgery."""
    before = G.copy()
    report = detect_singular_edges(before, curvature_threshold, distance_quantile, use_bridge_test)
    after = before.copy()
    removed: List[Tuple[str, str, float, str]] = []
    for _, row in report.iterrows():
        u, v = str(row["u"]), str(row["v"])
        if after.has_edge(u, v):
            after.remove_edge(u, v)
            removed.append((u, v, float(row["ricciCurvature"]), str(row["reason"])))
    if remove_isolated_nodes:
        after.remove_nodes_from(list(nx.isolates(after)))
    return SurgeryResult(
        before=before,
        after=after,
        removed_edges=removed,
        report=report,
        before_stats=graph_topology_stats(before),
        after_stats=graph_topology_stats(after),
    )


def compare_before_after_flow(G0: nx.Graph, G1: nx.Graph) -> pd.DataFrame:
    rows = []
    for u, v, data0 in G0.edges(data=True):
        if G1.has_edge(u, v):
            data1 = G1[u][v]
            d0 = float(data0.get("distance", data0.get("weight", np.nan)))
            d1 = float(data1.get("distance", data1.get("weight", np.nan)))
            rows.append({"u": u, "v": v, "distance_before": d0, "distance_after": d1, "distance_change": d1 - d0, "ricci_before": float(data0.get("ricciCurvature", 0.0)), "ricci_after": float(data1.get("ricciCurvature", 0.0)), "correlation": float(data0.get("correlation", np.nan))})
    return pd.DataFrame(rows).sort_values("distance_change", ascending=False) if rows else pd.DataFrame()


def rolling_feature_table(frames: Sequence[FrameData]) -> pd.DataFrame:
    return pd.DataFrame([{
        "date": fd.stats.end_date,
        "avg_ricci": fd.stats.avg_ricci,
        "ricci_std": fd.stats.ricci_std,
        "ricci_min": fd.stats.ricci_min,
        "ricci_max": fd.stats.ricci_max,
        "clusters": fd.stats.num_clusters,
        "largest_component_ratio": fd.stats.largest_component / max(fd.stats.num_nodes, 1),
        "nodes": fd.stats.num_nodes,
        "edges": fd.stats.num_edges,
        "density": fd.stats.density,
        "component_entropy": fd.stats.graph_entropy,
        "negative_edge_ratio": fd.stats.negative_edge_ratio,
    } for fd in frames])


def compute_hmm_regimes(frames: Sequence[FrameData], returns: pd.DataFrame, starts: Sequence[int], window_size: int, n_components: int = 3, forward_days: int = 5, random_state: int = 42) -> Tuple[pd.DataFrame, Dict[int, str]]:
    feature_df = rolling_feature_table(frames)
    market_ret = returns.mean(axis=1, skipna=True)
    next_returns = []
    for start in starts:
        end_pos = min(start + window_size - 1, len(market_ret) - 1)
        next_end = min(end_pos + forward_days, len(market_ret) - 1)
        if end_pos + 1 <= next_end:
            next_returns.append(float(market_ret.iloc[end_pos + 1: next_end + 1].sum()))
        else:
            next_returns.append(np.nan)
    fwd_col = f"next_{forward_days}d_market_return"
    feature_df[fwd_col] = next_returns
    feature_cols = ["avg_ricci", "ricci_std", "ricci_min", "clusters", "largest_component_ratio", "edges", "density", "component_entropy", "negative_edge_ratio"]
    if GaussianHMM is None or StandardScaler is None or len(feature_df) < max(8, n_components * 4):
        feature_df["hmm_state"] = -1
        feature_df["regime_name"] = "HMM unavailable or too few frames"
        for fd in frames:
            fd.stats.hmm_state = -1; fd.stats.regime_name = "HMM unavailable or too few frames"
        return feature_df, {-1: "HMM unavailable or too few frames"}
    X = feature_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xs = StandardScaler().fit_transform(X)
    model = GaussianHMM(n_components=int(n_components), covariance_type="full", n_iter=1000, random_state=int(random_state))
    model.fit(Xs)
    states = model.predict(Xs)
    feature_df["hmm_state"] = states
    summary = feature_df.groupby("hmm_state").agg(avg_ricci=("avg_ricci", "mean"), ricci_min=("ricci_min", "mean"), density=("density", "mean"), largest_ratio=("largest_component_ratio", "mean"), entropy=("component_entropy", "mean"), next_return=(fwd_col, "mean"), count=("date", "count"))
    regime_names: Dict[int, str] = {}
    for state, row in summary.iterrows():
        if row["ricci_min"] < summary["ricci_min"].median() and row["entropy"] >= summary["entropy"].median():
            name = "stress / fragmentation"
        elif row["avg_ricci"] >= summary["avg_ricci"].median() and row["density"] >= summary["density"].median():
            name = "coherent risk-on"
        else:
            name = "transition / rotation"
        regime_names[int(state)] = name
    feature_df["regime_name"] = feature_df["hmm_state"].map(regime_names)
    for i, fd in enumerate(frames):
        fd.stats.hmm_state = int(feature_df.loc[i, "hmm_state"])
        fd.stats.regime_name = str(feature_df.loc[i, "regime_name"])
    return feature_df, regime_names


def plot_rolling_features(feature_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=feature_df["date"], y=feature_df["avg_ricci"], mode="lines+markers", name="Average Ricci"))
    fig.add_trace(go.Scatter(x=feature_df["date"], y=feature_df["ricci_min"], mode="lines+markers", name="Minimum Ricci"))
    fig.add_trace(go.Scatter(x=feature_df["date"], y=feature_df["component_entropy"], mode="lines+markers", name="Component entropy"))
    fig.add_trace(go.Scatter(x=feature_df["date"], y=feature_df["clusters"], mode="lines+markers", name="Clusters", yaxis="y2"))
    fig.update_layout(title="Rolling Ricci, entropy, and cluster observables", height=520, xaxis_title="Window end date", yaxis_title="Curvature / entropy", yaxis2={"title": "Cluster count", "overlaying": "y", "side": "right"}, hovermode="x unified")
    return fig


def plot_ricci_flow_history(flow_history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if flow_history.empty:
        return fig
    fig.add_trace(go.Scatter(x=flow_history["iteration"], y=flow_history["avg_ricci"], mode="lines+markers", name="Average Ricci"))
    fig.add_trace(go.Scatter(x=flow_history["iteration"], y=flow_history["min_ricci"], mode="lines+markers", name="Minimum Ricci"))
    fig.add_trace(go.Scatter(x=flow_history["iteration"], y=flow_history["component_entropy"], mode="lines+markers", name="Entropy"))
    fig.add_trace(go.Scatter(x=flow_history["iteration"], y=flow_history["mean_distance"], mode="lines+markers", name="Mean distance", yaxis="y2"))
    fig.update_layout(title="Ricci flow diagnostics", height=460, xaxis_title="Flow iteration", yaxis_title="Curvature / entropy", yaxis2={"title": "Mean distance", "overlaying": "y", "side": "right"}, hovermode="x unified")
    return fig



SECTOR_MAP = {
    "NVDA": "AI", "AMD": "AI", "AVGO": "AI", "TSM": "AI", "SMCI": "AI", "PLTR": "AI", "ANET": "AI",
    "MU": "Memory", "MRVL": "AI", "AMAT": "Semicap", "LRCX": "Semicap", "KLAC": "Semicap",
    "AAOI": "Optical", "COHR": "Optical", "LITE": "Optical",
    "IONQ": "Quantum", "QBTS": "Quantum", "QUBT": "Quantum", "RGTI": "Quantum", "QNT": "Quantum", "BNT": "Quantum",
    "NBIS": "AI", "SPCX": "Space/IPO", "CBRS": "AI", "DXYZ": "AI",
}

def sector_flow_table(returns: pd.DataFrame, lookback: int = 20, sector_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Simple capital-flow proxy by sector.

    Ricci networks use correlation geometry, not dollars traded.  This table adds a
    separate momentum/participation proxy: recent cumulative return and breadth by sector.
    It helps distinguish "money flowing into AI/Quantum" from "AI/Quantum moving as one correlated cluster".
    """
    if sector_map is None:
        sector_map = SECTOR_MAP
    if returns.empty:
        return pd.DataFrame(columns=["sector", "tickers", "lookback", "cum_return", "mean_daily_return", "volatility", "positive_breadth", "active_tickers"])
    lb = min(max(1, int(lookback)), len(returns))
    recent = returns.tail(lb)
    rows = []
    for sector in sorted(set(sector_map.get(t, "Other") for t in recent.columns)):
        cols = [t for t in recent.columns if sector_map.get(t, "Other") == sector and recent[t].notna().sum() > 0]
        if not cols:
            continue
        sector_ret = recent[cols].mean(axis=1, skipna=True).dropna()
        if sector_ret.empty:
            continue
        indiv_cum = np.exp(recent[cols].sum(skipna=True)) - 1.0
        rows.append({
            "sector": sector,
            "tickers": ", ".join(cols),
            "lookback": lb,
            "cum_return": float(np.exp(sector_ret.sum()) - 1.0),
            "mean_daily_return": float(sector_ret.mean()),
            "volatility": float(sector_ret.std()) if len(sector_ret) > 1 else 0.0,
            "positive_breadth": float((indiv_cum > 0).mean()) if len(indiv_cum) else 0.0,
            "active_tickers": int(len(cols)),
        })
    return pd.DataFrame(rows).sort_values("cum_return", ascending=False) if rows else pd.DataFrame()

def plot_sector_flow(flow_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if flow_df.empty:
        fig.update_layout(title="Sector capital-flow proxy", height=420)
        return fig
    fig.add_trace(go.Bar(
        x=flow_df["sector"],
        y=100.0 * flow_df["cum_return"],
        text=[f"{v:.1f}%" for v in 100.0 * flow_df["cum_return"]],
        textposition="outside",
        hovertext=flow_df["tickers"],
        hoverinfo="x+y+text",
        name="Cumulative return",
    ))
    fig.update_layout(
        title="Sector capital-flow proxy: recent cumulative return",
        height=430,
        yaxis_title="Recent cumulative return (%)",
        xaxis_title="Theme / sector",
        margin={"l": 40, "r": 20, "t": 60, "b": 70},
    )
    return fig

def summarize_edges(G: nx.Graph) -> pd.DataFrame:
    rows = []
    for u, v, data in G.edges(data=True):
        rows.append({"u": u, "v": v, "distance": float(data.get("distance", data.get("weight", np.nan))), "correlation": float(data.get("correlation", np.nan)), "ricciCurvature": float(data.get("ricciCurvature", 0.0)), "edge_source": data.get("edge_source", "")})
    return pd.DataFrame(rows).sort_values("ricciCurvature") if rows else pd.DataFrame(columns=["u", "v", "distance", "correlation", "ricciCurvature", "edge_source"])
