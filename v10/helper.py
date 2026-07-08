"""
helper.py
Graduate lecture helpers for: Perelman / Ricci ideas in financial networks.

The module keeps the notebook readable by moving data download, graph construction,
Ollivier-Ricci curvature, rolling-frame construction, HMM regime detection, and
visualization into reusable functions.

Install core packages:
    pip install yfinance pandas numpy networkx plotly matplotlib scikit-learn hmmlearn

Optional, for true Ollivier-Ricci curvature:
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
    "RGTI", "NBIS","QNT","SPCX",
]

COMPONENT_COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
]


@dataclass
class WindowStats:
    end_date: str
    avg_ricci: float
    ricci_std: float
    num_clusters: int
    largest_component: int
    num_nodes: int
    num_edges: int
    density: float
    hmm_state: int = -1
    regime_name: str = "not computed"


@dataclass
class FrameData:
    G: nx.Graph
    node_cluster: Dict[str, int]
    stats: WindowStats
    corr: pd.DataFrame
    dist: pd.DataFrame


def parse_tickers(text_or_list: str | Sequence[str]) -> List[str]:
    """Return unique upper-case ticker symbols from comma/newline text or a list."""
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
    """Download adjusted close prices with yfinance. Falls back cleanly if data are unavailable."""
    if yf is None:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")
    # Trick for yfinance to avoid caching issues on some systems. See
    yf.set_tz_cache_location(".")

    tickers = parse_tickers(tickers)
    data = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=False,
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
    tickers: Sequence[str] = DEFAULT_TICKERS[:8],
    n_days: int = 260,
    seed: int = 7,
    ipo_aware: bool = True,
) -> pd.DataFrame:
    """Synthetic prices for offline lectures or when yfinance is unavailable.

    IPO-aware mode keeps late-start tickers such as QNT visible in the ticker
    universe but with NaN prices before their synthetic start date. The graph
    builder will still draw their node before IPO, but no correlation edge is
    created until enough observations exist.
    """
    rng = np.random.default_rng(seed)
    tickers = parse_tickers(tickers)
    # Three latent market factors: broad AI, memory/equipment, speculative beta.
    factors = rng.normal(0, [0.010, 0.014, 0.020], size=(n_days, 3))
    loadings = rng.uniform(-0.3, 1.2, size=(len(tickers), 3))
    noise = rng.normal(0, 0.012, size=(n_days, len(tickers)))
    returns = factors @ loadings.T + noise
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    df = pd.DataFrame(prices, index=idx, columns=tickers)

    if ipo_aware:
        late_start_fraction = {
            "QNT": 0.70,
            "BNT": 0.70,
            "SPCX": 0.76,
            "CBRS": 0.62,
            "DXYZ": 0.55,
        }
        for symbol, frac in late_start_fraction.items():
            if symbol in df.columns:
                start_idx = min(max(1, int(float(frac) * n_days)), n_days - 2)
                df.loc[df.index[:start_idx], symbol] = np.nan
                base = df.loc[df.index[start_idx], symbol]
                if pd.notna(base) and base != 0:
                    df.loc[df.index[start_idx]:, symbol] = 100.0 * df.loc[df.index[start_idx]:, symbol] / base
    return df


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns from price levels."""
    prices = prices.replace([np.inf, -np.inf], np.nan)
    returns = np.log(prices / prices.shift(1))
    return returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")


def financial_distance_from_corr(corr: pd.DataFrame) -> pd.DataFrame:
    """Mantegna-style distance d_ij = sqrt(2(1-rho_ij))."""
    clipped = corr.clip(lower=-1.0, upper=1.0)
    dist = np.sqrt(2.0 * (1.0 - clipped))
    return pd.DataFrame(dist, index=corr.index, columns=corr.columns)


def build_knn_edges_from_corr(
    corr: pd.DataFrame,
    k: int = 3,
    min_corr: float = 0.05,
) -> List[Tuple[str, str, float, float]]:
    """Return k-nearest-neighbor edges from a correlation matrix.

    This is usually better for a *market map* than a pure threshold graph because
    each ticker keeps its closest positive-correlation neighbors, so isolated
    components such as a single PLTR node are less likely to appear only because
    a threshold was too strict.
    """
    if corr.empty or len(corr.columns) < 2:
        return []
    dist = financial_distance_from_corr(corr)
    edges: Dict[Tuple[str, str], Tuple[str, str, float, float]] = {}
    k = max(1, int(k))
    for u in corr.columns:
        nearest = dist.loc[u].drop(index=u, errors="ignore").sort_values().head(k)
        for v, d in nearest.items():
            rho = float(corr.loc[u, v])
            d = float(d)
            if np.isfinite(rho) and np.isfinite(d) and rho >= float(min_corr):
                a, b = sorted((str(u), str(v)))
                # If both directions nominate the pair, keep the shorter distance.
                if (a, b) not in edges or d < edges[(a, b)][2]:
                    edges[(a, b)] = (a, b, d, rho)
    return list(edges.values())


def add_bridge_edges(
    G: nx.Graph,
    corr: pd.DataFrame,
    max_bridges: int = 3,
    min_corr: float = 0.0,
) -> nx.Graph:
    """Add a few strongest positive-correlation edges not already in G.

    These weak bridge edges keep separated themes in one readable map. Edges are
    tagged with ``bridge=True`` so they can be inspected separately.
    """
    H = G.copy()
    if corr.empty or max_bridges <= 0:
        return H
    existing = {tuple(sorted((u, v))) for u, v in H.edges()}
    candidates: List[Tuple[str, str, float, float]] = []
    cols = list(corr.columns)
    for i, u in enumerate(cols):
        for v in cols[i + 1:]:
            key = tuple(sorted((u, v)))
            if key in existing:
                continue
            rho = float(corr.loc[u, v])
            if not np.isfinite(rho) or rho <= float(min_corr):
                continue
            d = float(np.sqrt(2.0 * (1.0 - np.clip(rho, -1.0, 1.0))))
            candidates.append((u, v, d, rho))
    for u, v, d, rho in sorted(candidates, key=lambda x: x[2])[: int(max_bridges)]:
        H.add_edge(u, v, weight=d, distance=d, correlation=float(rho), bridge=True)
    return H


def build_graph_from_window(
    window_returns: pd.DataFrame,
    max_distance: float = 1.05,
    min_abs_corr: float = 0.30,
    keep_top_edges: Optional[int] = None,
    min_node_obs: int = 1,
    min_pair_obs: int = 4,
    graph_mode: str = "threshold",
    k_neighbors: int = 3,
    min_corr: float = 0.05,
    max_bridges: int = 0,
) -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    """Build an IPO-aware correlation-distance graph from one rolling return window.

    graph_mode:
        ``threshold``: only edges satisfying distance/correlation filters.
        ``knn``: each node keeps k nearest positive-correlation neighbors.
        ``knn+bridges``: kNN plus a few positive bridge edges for a more readable
        market-map layout.
    """
    min_pair_obs = max(3, int(min_pair_obs))
    min_node_obs = max(1, int(min_node_obs))

    # IMPORTANT v10 correction:
    # Keep every requested ticker as a node in every rolling frame.
    # Example: before QNT IPO/first valid price, QNT should still be visible
    # as an inactive isolated node. It simply has no edges until enough
    # observations exist for pairwise correlation.
    node_cols = list(window_returns.columns)

    # Edges still require enough valid observations. This preserves IPO-aware
    # mathematics while keeping the visual market universe stable.
    pair_cols = [c for c in window_returns.columns if window_returns[c].notna().sum() >= min_pair_obs]
    clean = window_returns[pair_cols].copy()
    if len(pair_cols) >= 2:
        corr = clean.corr(min_periods=min_pair_obs).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        corr = pd.DataFrame(index=pair_cols, columns=pair_cols, dtype=float)
    dist = financial_distance_from_corr(corr)
    G = nx.Graph()
    G.add_nodes_from(node_cols)
    for c in node_cols:
        valid_obs = int(window_returns[c].notna().sum()) if c in window_returns.columns else 0
        G.nodes[c]["valid_obs"] = valid_obs
        G.nodes[c]["is_active"] = bool(valid_obs >= min_pair_obs)
        G.nodes[c]["status"] = "active" if valid_obs >= min_pair_obs else "waiting_for_data"

    mode = str(graph_mode).lower().replace(" ", "")
    if mode in {"knn", "k-nearest", "nearest", "knn+bridges", "knnbridges"}:
        candidates = build_knn_edges_from_corr(corr, k=int(k_neighbors), min_corr=float(min_corr))
    else:
        candidates: List[Tuple[str, str, float, float]] = []
        tickers = list(corr.columns)
        for i, u in enumerate(tickers):
            for j in range(i + 1, len(tickers)):
                v = tickers[j]
                rho = float(corr.iloc[i, j])
                d = float(dist.iloc[i, j])
                if np.isfinite(rho) and np.isfinite(d) and d <= max_distance and abs(rho) >= min_abs_corr:
                    candidates.append((u, v, d, rho))

    if keep_top_edges is not None and keep_top_edges > 0:
        candidates = sorted(candidates, key=lambda x: x[2])[: int(keep_top_edges)]
    for u, v, d, rho in candidates:
        G.add_edge(u, v, weight=float(d), distance=float(d), correlation=float(rho))

    if mode in {"knn+bridges", "knnbridges"} and max_bridges > 0:
        G = add_bridge_edges(G, corr, max_bridges=int(max_bridges), min_corr=0.0)
    return G, corr, dist


def compute_ricci_curvature(
    G: nx.Graph,
    alpha: float = 0.5,
    method: str = "OTD",
    proc: int = 1,
    fallback: str = "forman_like",
) -> nx.Graph:
    """Compute Ollivier-Ricci curvature when available; otherwise use a simple didactic proxy.

    The fallback is not a replacement for Ollivier-Ricci. It only keeps the lecture notebook
    executable on systems where GraphRicciCurvature is not installed.
    """
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
    # Didactic fallback: high triangle support and low bridge-likeness => positive curvature proxy.
    for u, v in H.edges():
        cn = len(list(nx.common_neighbors(H, u, v)))
        deg_sum = max(1, H.degree(u) + H.degree(v) - 2)
        H[u][v]["ricciCurvature"] = float(2 * cn / deg_sum - 0.5)
    return H


def compute_components(G: nx.Graph) -> Dict[str, int]:
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    node_cluster: Dict[str, int] = {}
    for cid, comp in enumerate(components):
        for node in comp:
            node_cluster[node] = cid
    return node_cluster


def compute_window_stats(G: nx.Graph, end_date: str) -> WindowStats:
    vals = [float(data.get("ricciCurvature", 0.0)) for _, _, data in G.edges(data=True)]
    comps = list(nx.connected_components(G))
    return WindowStats(
        end_date=str(end_date),
        avg_ricci=float(np.mean(vals)) if vals else 0.0,
        ricci_std=float(np.std(vals)) if vals else 0.0,
        num_clusters=len(comps),
        largest_component=max((len(c) for c in comps), default=0),
        num_nodes=G.number_of_nodes(),
        num_edges=G.number_of_edges(),
        density=float(nx.density(G)) if G.number_of_nodes() > 1 else 0.0,
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
    graph_mode: str = "threshold",
    k_neighbors: int = 3,
    min_corr: float = 0.05,
    max_bridges: int = 0,
) -> FrameData:
    window_returns = returns.iloc[start : start + window_size]
    end_date = str(window_returns.index[-1])[:19]
    G0, corr, dist = build_graph_from_window(
        window_returns,
        max_distance=max_distance,
        min_abs_corr=min_abs_corr,
        keep_top_edges=keep_top_edges,
        min_node_obs=min_node_obs,
        min_pair_obs=min_pair_obs,
        graph_mode=graph_mode,
        k_neighbors=k_neighbors,
        min_corr=min_corr,
        max_bridges=max_bridges,
    )
    G = compute_ricci_curvature(G0, alpha=alpha, method=method, proc=proc)
    node_cluster = compute_components(G)
    stats = compute_window_stats(G, end_date)
    return FrameData(G=G, node_cluster=node_cluster, stats=stats, corr=corr, dist=dist)


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
    frames = [build_frame(returns, start=s, window_size=window_size, **kwargs) for s in starts]
    return frames, starts


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


def compute_stable_layout(
    base_graph: nx.Graph,
    seed: int = 42,
    scale: float = 1.0,
    layout_k: float = 0.45,
    iterations: int = 600,
) -> Dict[str, Tuple[float, float]]:
    """Stable spring layout with tunable spacing. Lower layout_k pulls clusters closer."""
    if base_graph.number_of_nodes() == 0:
        return {}
    if base_graph.number_of_edges() == 0:
        pos = nx.circular_layout(base_graph, scale=scale)
    else:
        pos = nx.spring_layout(
            base_graph,
            seed=seed,
            weight="weight",
            k=float(layout_k),
            iterations=int(iterations),
            scale=scale,
        )
    return {n: (float(x), float(y)) for n, (x, y) in pos.items()}


def ricci_to_hex(kappa: float, vmin: float = -0.6, vmax: float = 0.6) -> str:
    cmap = mpl.colormaps["coolwarm_r"]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    return mpl.colors.to_hex(cmap(norm(float(kappa))))


def layout_axis_ranges(positions: Dict[str, Tuple[float, float]], pad_ratio: float = 0.15) -> Tuple[List[float], List[float]]:
    """Return safe x/y ranges for Plotly from a stable layout.

    This prevents NameError from undefined x_range/y_range and keeps isolated
    IPO-waiting nodes visible at the edge of the animation canvas.
    """
    if not positions:
        return [-1.2, 1.2], [-1.2, 1.2]
    xs = np.array([float(x) for x, _ in positions.values()], dtype=float)
    ys = np.array([float(y) for _, y in positions.values()], dtype=float)
    if xs.size == 0 or ys.size == 0 or not np.isfinite(xs).any() or not np.isfinite(ys).any():
        return [-1.2, 1.2], [-1.2, 1.2]
    xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
    ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
    xspan = max(xmax - xmin, 1e-6)
    yspan = max(ymax - ymin, 1e-6)
    xpad = max(0.15, xspan * float(pad_ratio))
    ypad = max(0.15, yspan * float(pad_ratio))
    return [xmin - xpad, xmax + xpad], [ymin - ypad, ymax + ypad]


def visualize_network(
    G: nx.Graph,
    positions: Optional[Dict[str, Tuple[float, float]]] = None,
    title: str = "Ricci financial network",
    node_cluster: Optional[Dict[str, int]] = None,
    node_size_base: int = 24,
    show_edge_weights: bool = True,
) -> go.Figure:
    """Visualize one financial network.

    Edge color = Ricci curvature.
    Edge text label = current edge weight / financial distance.
    """
    if positions is None:
        positions = compute_stable_layout(G)
    if node_cluster is None:
        node_cluster = compute_components(G)

    x_range, y_range = layout_axis_ranges(positions)

    edge_traces = []
    edge_label_x, edge_label_y, edge_label_text = [], [], []

    for u, v, data in G.edges(data=True):
        x0, y0 = positions.get(u, (0.0, 0.0))
        x1, y1 = positions.get(v, (0.0, 0.0))
        kappa = float(data.get("ricciCurvature", 0.0))
        rho = float(data.get("correlation", np.nan))
        d = float(data.get("distance", data.get("weight", np.nan)))
        w = float(data.get("weight", d))

        edge_traces.append(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line={"width": 2.0, "color": ricci_to_hex(kappa)},
            hoverinfo="text",
            text=(
                f"{u}-{v}<br>"
                f"weight={w:.3f}<br>"
                f"distance={d:.3f}<br>"
                f"correlation={rho:.3f}<br>"
                f"Ricci={kappa:.3f}"
            ),
            showlegend=False,
        ))

        edge_label_x.append((x0 + x1) / 2.0)
        edge_label_y.append((y0 + y1) / 2.0)
        edge_label_text.append(f"{w:.2f}")

    edge_label_trace = go.Scatter(
        x=edge_label_x,
        y=edge_label_y,
        mode="text",
        text=edge_label_text if show_edge_weights else [],
        textposition="middle center",
        textfont={"size": 10},
        hoverinfo="skip",
        showlegend=False,
    )

    degrees = dict(G.degree())
    node_x, node_y, labels, hover, colors, sizes = [], [], [], [], [], []
    for node in sorted(G.nodes()):
        x, y = positions.get(node, (0.0, 0.0))
        cid = node_cluster.get(node, 0)
        node_x.append(x)
        node_y.append(y)
        labels.append(node)
        status = G.nodes[node].get("status", "active")
        valid_obs = G.nodes[node].get("valid_obs", "")
        hover.append(f"Ticker: {node}<br>Status: {status}<br>Valid obs: {valid_obs}<br>Component: {cid}<br>Degree: {degrees.get(node, 0)}")
        colors.append("#D0D0D0" if status != "active" else COMPONENT_COLORS[cid % len(COMPONENT_COLORS)])
        sizes.append(node_size_base + 2 * degrees.get(node, 0))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker={"size": sizes, "color": colors, "line": {"width": 1.2, "color": "#111"}},
        hoverinfo="text",
        hovertext=hover,
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [edge_label_trace, node_trace])
    fig.update_layout(
        title=title,
        height=650,
        hovermode="closest",
        showlegend=False,
        xaxis={"showgrid": False, "zeroline": False, "visible": False, "range": x_range},
        yaxis={"showgrid": False, "zeroline": False, "visible": False, "range": y_range, "scaleanchor": "x", "scaleratio": 1},
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig



def edge_universe(frames: Sequence[FrameData]) -> List[Tuple[str, str]]:
    """Return a stable sorted list of every edge that appears in any animation frame."""
    edges = set()
    for fd in frames:
        for u, v in fd.G.edges():
            edges.add(tuple(sorted((str(u), str(v)))))
    return sorted(edges)


def stats_annotation(stats: WindowStats) -> dict:
    """Right-side annotation used by the v9-style Plotly animation."""
    ricci_min = getattr(stats, "ricci_min", None)
    ricci_max = getattr(stats, "ricci_max", None)
    entropy = getattr(stats, "graph_entropy", None)
    negative_ratio = getattr(stats, "negative_edge_ratio", None)

    extra = ""
    if ricci_min is not None:
        extra += f"<br><b>Ricci min:</b> {float(ricci_min):.4f}"
    if ricci_max is not None:
        extra += f"<br><b>Ricci max:</b> {float(ricci_max):.4f}"
    if entropy is not None:
        extra += f"<br><b>Entropy:</b> {float(entropy):.3f}"
    if negative_ratio is not None:
        extra += f"<br><b>Negative-edge ratio:</b> {float(negative_ratio):.2f}"

    return {
        "text": (
            f"<b>Date:</b> {stats.end_date}<br>"
            f"<b>Avg Ricci:</b> {stats.avg_ricci:.4f}<br>"
            f"<b>Ricci σ:</b> {stats.ricci_std:.4f}<br>"
            f"<b>Clusters:</b> {stats.num_clusters}<br>"
            f"<b>Nodes:</b> {stats.num_nodes} &nbsp; <b>Edges:</b> {stats.num_edges}<br>"
            f"<b>Density:</b> {stats.density:.3f}"
            f"{extra}<br>"
            f"<b>HMM:</b> {stats.hmm_state} - {stats.regime_name}"
        ),
        "xref": "paper",
        "yref": "paper",
        "x": 1.02,
        "y": 0.98,
        "showarrow": False,
        "align": "left",
        "bordercolor": "#999",
        "borderwidth": 1,
        "bgcolor": "rgba(255,255,255,0.92)",
        "font": {"size": 13, "color": "#111"},
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
    """v9-style animation frame: one trace per edge in the edge universe, then labels, then nodes.

    This avoids the Plotly issue where a changing number/order of traces can make
    the final node trace disappear or be overwritten by line traces.
    """
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
    labeled_pairs = {
        tuple(sorted((str(u), str(v)))) for u, v, _ in current_edges_by_distance
    } if show_edge_weight_labels else set()

    for u, v in all_edges:
        if G.has_edge(u, v):
            data = G[u][v]
            x0, y0 = positions.get(u, (0.0, 0.0))
            x1, y1 = positions.get(v, (0.0, 0.0))
            kappa = float(data.get("ricciCurvature", 0.0))
            rho = float(data.get("correlation", np.nan))
            d = float(data.get("distance", data.get("weight", np.nan)))
            width = 0.8 + float(edge_width_scale) * min(1.0, abs(kappa))
            traces.append(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line={"width": width, "color": ricci_to_hex(kappa)},
                hoverinfo="text",
                text=f"{u}-{v}<br>weight / distance={d:.4f}<br>correlation={rho:.4f}<br>Ricci={kappa:.4f}",
                showlegend=False,
            ))
            if tuple(sorted((str(u), str(v)))) in labeled_pairs:
                label_x.append((x0 + x1) / 2.0)
                label_y.append((y0 + y1) / 2.0)
                label_text.append(f"w={d:.2f}")
                label_hover.append(f"{u}-{v}<br>weight / distance={d:.4f}<br>correlation={rho:.4f}<br>Ricci={kappa:.4f}")
        else:
            traces.append(go.Scatter(
                x=[None, None],
                y=[None, None],
                mode="lines",
                line={"width": 0, "color": "rgba(0,0,0,0)"},
                hoverinfo="skip",
                showlegend=False,
            ))

    traces.append(go.Scatter(
        x=label_x,
        y=label_y,
        mode="text",
        text=label_text,
        textfont={"size": 11, "color": "#111111"},
        hoverinfo="text",
        hovertext=label_hover,
        showlegend=False,
    ))

    degrees = dict(G.degree())
    node_x: List[float] = []
    node_y: List[float] = []
    labels: List[str] = []
    hover: List[str] = []
    colors: List[str] = []
    sizes: List[int] = []

    for node in sorted(G.nodes()):
        x, y = positions.get(node, (0.0, 0.0))
        cid = fd.node_cluster.get(node, 0)
        node_x.append(x)
        node_y.append(y)
        labels.append(str(node))
        status = G.nodes[node].get("status", "active")
        valid_obs = G.nodes[node].get("valid_obs", "")
        hover.append(f"Ticker: {node}<br>Status: {status}<br>Valid obs: {valid_obs}<br>Component: {cid}<br>Degree: {degrees.get(node, 0)}")
        colors.append("#D0D0D0" if status != "active" else COMPONENT_COLORS[cid % len(COMPONENT_COLORS)])
        sizes.append(node_size_base + 2 * degrees.get(node, 0))

    traces.append(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        textfont={"size": int(node_label_size), "color": "#111"},
        hoverinfo="text",
        hovertext=hover,
        marker={"size": sizes, "color": colors, "line": {"width": 1.5, "color": "#111"}},
        showlegend=False,
    ))
    return traces


def build_plotly_animation(
    frames: Sequence[FrameData],
    positions: Dict[str, Tuple[float, float]],
    frame_duration_ms: int = 700,
    node_label_size: int = 14,
    node_size_base: int = 24,
    edge_width_scale: float = 5.0,
    title: str = "Rolling Ricci Finance v10",
    show_edge_weight_labels: bool = True,
    edge_label_top_n: int = 30,
) -> go.Figure:
    """v9 animation implementation, ported to v10.

    It keeps a stable trace universe: each possible edge always has the same trace
    index across all frames, followed by one edge-label trace and one node trace.
    This is more robust in Streamlit/Plotly than a single variable-length edge trace.
    """
    if not frames:
        fig = go.Figure()
        fig.update_layout(title=f"{title}: no frames", height=780)
        return fig

    all_edges = edge_universe(frames)
    base_data = _frame_traces(
        frames[0], positions, all_edges, node_label_size, node_size_base,
        edge_width_scale, show_edge_weight_labels, edge_label_top_n,
    )
    trace_indices = list(range(len(base_data)))
    x_range, y_range = layout_axis_ranges(positions)

    fig = go.Figure(data=base_data)
    fig.frames = [
        go.Frame(
            name=str(i),
            data=_frame_traces(
                fd, positions, all_edges, node_label_size, node_size_base,
                edge_width_scale, show_edge_weight_labels, edge_label_top_n,
            ),
            traces=trace_indices,
            layout=go.Layout(annotations=[stats_annotation(fd.stats)]),
        )
        for i, fd in enumerate(frames)
    ]

    steps = [
        {
            "method": "animate",
            "label": str(i + 1),
            "args": [[str(i)], {
                "mode": "immediate",
                "frame": {"duration": frame_duration_ms, "redraw": True},
                "transition": {"duration": 0},
            }],
        }
        for i in range(len(frames))
    ]

    fig.update_layout(
        title=title,
        height=780,
        showlegend=False,
        hovermode="closest",
        annotations=[stats_annotation(frames[0].stats)],
        xaxis={"showgrid": False, "zeroline": False, "visible": False, "range": x_range},
        yaxis={"showgrid": False, "zeroline": False, "visible": False, "range": y_range, "scaleanchor": "x", "scaleratio": 1},
        margin={"l": 20, "r": 230, "t": 60, "b": 70},
        plot_bgcolor="white",
        paper_bgcolor="white",
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.02,
            "y": -0.05,
            "xanchor": "left",
            "yanchor": "top",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": frame_duration_ms, "redraw": True},
                        "transition": {"duration": 0},
                        "fromcurrent": True,
                    }],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                },
            ],
        }],
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "Frame ", "font": {"size": 14}},
            "pad": {"t": 40},
            "steps": steps,
        }],
    )
    return fig


def animate_rolling_networks(
    frames: Sequence[FrameData],
    positions: Dict[str, Tuple[float, float]],
    title: str = "Rolling Ricci network animation",
    show_edge_weights: bool = True,
) -> go.Figure:
    """Backward-compatible v10 name that now uses the v9 robust animation engine."""
    return build_plotly_animation(
        frames=frames,
        positions=positions,
        title=title,
        show_edge_weight_labels=show_edge_weights,
    )

def rolling_feature_table(frames: Sequence[FrameData]) -> pd.DataFrame:
    """Return rolling features for Ricci geometry + capital-flow diagnostics.

    These columns are used both for plots and for the HMM hidden-regime model.
    """
    rows = []
    for fd in frames:
        G = fd.G
        node_masses = [float(data.get("capital_mass", 0.0)) for _, data in G.nodes(data=True)]
        edge_flows = [float(data.get("edge_capital_flow", 0.0)) for _, _, data in G.edges(data=True)]
        total_node_capital = float(np.sum(node_masses)) if node_masses else 0.0
        total_edge_capital_flow = float(np.sum(edge_flows)) if edge_flows else 0.0
        max_node_share = float(np.max(node_masses) / total_node_capital) if total_node_capital > 0 and node_masses else 0.0
        avg_edge_capital_flow = float(np.mean(edge_flows)) if edge_flows else 0.0
        rows.append({
            "date": fd.stats.end_date,
            "avg_ricci": fd.stats.avg_ricci,
            "ricci_std": fd.stats.ricci_std,
            "clusters": fd.stats.num_clusters,
            "largest_component_ratio": fd.stats.largest_component / max(fd.stats.num_nodes, 1),
            "nodes": fd.stats.num_nodes,
            "edges": fd.stats.num_edges,
            "density": fd.stats.density,
            "total_node_capital": total_node_capital,
            "total_edge_capital_flow": total_edge_capital_flow,
            "avg_edge_capital_flow": avg_edge_capital_flow,
            "max_node_capital_share": max_node_share,
        })
    return pd.DataFrame(rows)


def compute_hmm_regimes(
    frames: Sequence[FrameData],
    returns: pd.DataFrame,
    starts: Sequence[int],
    window_size: int,
    n_components: int = 3,
    forward_days: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """Fit a robust Gaussian HMM to rolling Ricci + capital-flow features."""
    feature_df = rolling_feature_table(frames)
    market_ret = returns.mean(axis=1, skipna=True)
    next_returns = []
    for start in starts:
        end_pos = min(start + window_size - 1, len(market_ret) - 1)
        next_end = min(end_pos + forward_days, len(market_ret) - 1)
        if end_pos + 1 <= next_end:
            next_returns.append(float(market_ret.iloc[end_pos + 1 : next_end + 1].sum()))
        else:
            next_returns.append(np.nan)
    fwd_col = f"next_{forward_days}d_market_return"
    feature_df[fwd_col] = next_returns

    candidate_cols = [
        "avg_ricci", "ricci_std", "clusters", "largest_component_ratio",
        "edges", "density", "total_node_capital", "total_edge_capital_flow",
        "avg_edge_capital_flow", "max_node_capital_share",
    ]
    feature_cols = [c for c in candidate_cols if c in feature_df.columns]

    if GaussianHMM is None or StandardScaler is None or len(feature_df) < max(8, int(n_components) * 4):
        feature_df["hmm_state"] = -1
        feature_df["regime_name"] = "HMM unavailable or too few frames"
        for fd in frames:
            fd.stats.hmm_state = -1
            fd.stats.regime_name = "HMM unavailable or too few frames"
        return feature_df, {-1: "HMM unavailable or too few frames"}

    X = feature_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Log-transform large capital scale columns so the HMM is not dominated by NVDA-sized dollar volume.
    for col in ["total_node_capital", "total_edge_capital_flow", "avg_edge_capital_flow"]:
        if col in X.columns:
            X[col] = np.log1p(np.maximum(X[col].astype(float), 0.0))

    Xs = StandardScaler().fit_transform(X)
    model = GaussianHMM(
        n_components=int(n_components),
        covariance_type="diag",
        min_covar=1e-3,
        n_iter=1000,
        random_state=int(random_state),
    )
    model.fit(Xs)
    states = model.predict(Xs)
    feature_df["hmm_state"] = states

    summary = feature_df.groupby("hmm_state").agg(
        avg_ricci=("avg_ricci", "mean"),
        density=("density", "mean"),
        largest_ratio=("largest_component_ratio", "mean"),
        capital_flow=("total_edge_capital_flow", "mean"),
        next_return=(fwd_col, "mean"),
        count=("date", "count"),
    )

    regime_names: Dict[int, str] = {}
    ricci_med = summary["avg_ricci"].median()
    density_med = summary["density"].median()
    flow_med = summary["capital_flow"].median() if "capital_flow" in summary.columns else 0.0
    largest_med = summary["largest_ratio"].median()

    for state, row in summary.iterrows():
        if row["avg_ricci"] < ricci_med and row["largest_ratio"] < largest_med:
            name = "stress / fragmentation"
        elif row["avg_ricci"] >= ricci_med and row["density"] >= density_med and row["capital_flow"] >= flow_med:
            name = "coherent capital risk-on"
        elif row["capital_flow"] >= flow_med and row["density"] < density_med:
            name = "capital rotation / dispersion"
        else:
            name = "transition / rotation"
        regime_names[int(state)] = name

    feature_df["regime_name"] = feature_df["hmm_state"].map(regime_names)
    for i, fd in enumerate(frames):
        fd.stats.hmm_state = int(feature_df.loc[i, "hmm_state"])
        fd.stats.regime_name = str(feature_df.loc[i, "regime_name"])
    return feature_df, regime_names


def plot_hmm_regimes(hmm_df: pd.DataFrame) -> go.Figure:
    """Plot hidden regimes over time with Ricci and capital-flow context."""
    fig = go.Figure()
    if hmm_df is None or hmm_df.empty:
        fig.update_layout(title="HMM regimes: no data", height=420)
        return fig
    fig.add_trace(go.Scatter(x=hmm_df["date"], y=hmm_df["hmm_state"], mode="lines+markers", name="HMM state", yaxis="y"))
    if "avg_ricci" in hmm_df.columns:
        fig.add_trace(go.Scatter(x=hmm_df["date"], y=hmm_df["avg_ricci"], mode="lines", name="Average Ricci", yaxis="y2"))
    if "total_edge_capital_flow" in hmm_df.columns:
        y = np.log1p(hmm_df["total_edge_capital_flow"].fillna(0.0))
        fig.add_trace(go.Scatter(x=hmm_df["date"], y=y, mode="lines", name="log(1 + edge capital flow)", yaxis="y3"))
    fig.update_layout(
        title="HMM hidden regimes from Ricci + capital-flow features",
        height=520,
        xaxis_title="Window end date",
        yaxis={"title": "Hidden state"},
        yaxis2={"title": "Avg Ricci", "overlaying": "y", "side": "right"},
        yaxis3={"title": "log capital flow", "overlaying": "y", "side": "right", "position": 0.95, "anchor": "free"},
        hovermode="x unified",
    )
    return fig


def plot_rolling_features(feature_df: pd.DataFrame) -> go.Figure:
    """Line chart for Ricci curvature, density, and cluster count through time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=feature_df["date"], y=feature_df["avg_ricci"], mode="lines+markers", name="Average Ricci"))
    fig.add_trace(go.Scatter(x=feature_df["date"], y=feature_df["density"], mode="lines+markers", name="Density"))
    fig.add_trace(go.Scatter(x=feature_df["date"], y=feature_df["clusters"], mode="lines+markers", name="Clusters", yaxis="y2"))
    fig.update_layout(
        title="Rolling Ricci-network observables",
        height=520,
        xaxis_title="Window end date",
        yaxis_title="Curvature / density",
        yaxis2={"title": "Cluster count", "overlaying": "y", "side": "right"},
        hovermode="x unified",
    )
    return fig


def summarize_edges(G: nx.Graph) -> pd.DataFrame:
    rows = []
    for u, v, data in G.edges(data=True):
        rows.append({
            "u": u,
            "v": v,
            "distance": float(data.get("distance", data.get("weight", np.nan))),
            "correlation": float(data.get("correlation", np.nan)),
            "ricciCurvature": float(data.get("ricciCurvature", 0.0)),
        })
    return pd.DataFrame(rows).sort_values("distance") if rows else pd.DataFrame(columns=["u", "v", "distance", "correlation", "ricciCurvature"])


def run_ricci_flow(
    G: nx.Graph,
    iterations: int = 10,
    step_size: float = 0.25,
    alpha: float = 0.5,
    method: str = "OTD",
    proc: int = 1,
    min_weight: float = 1e-4,
    max_weight: float = 10.0,
    recompute_curvature: bool = True,
    normalize_mean_weight: bool = True,
) -> Tuple[nx.Graph, pd.DataFrame]:
    """Run a simple discrete Ricci flow on graph edge distances.

    The update is the graph analogue of shrinking positive-curvature edges and
    stretching negative-curvature edges:

        w_{t+1}(u,v) = w_t(u,v) * (1 - step_size * kappa_t(u,v))

    Optional normalization keeps the average edge weight stable so the graph
    does not globally collapse or explode during flow iterations.
    """
    H = G.copy()
    history: List[Dict[str, float]] = []

    if H.number_of_edges() == 0:
        return H, pd.DataFrame(columns=["iteration", "avg_weight", "avg_ricci", "ricci_min", "ricci_max", "edges"])

    if not all("ricciCurvature" in data for _, _, data in H.edges(data=True)):
        H = compute_ricci_curvature(H, alpha=alpha, method=method, proc=proc)

    initial_weights = [float(data.get("weight", data.get("distance", 1.0))) for _, _, data in H.edges(data=True)]
    target_mean_weight = float(np.mean(initial_weights)) if initial_weights else 1.0

    for it in range(int(iterations) + 1):
        weights = [float(data.get("weight", data.get("distance", 1.0))) for _, _, data in H.edges(data=True)]
        curvatures = [float(data.get("ricciCurvature", 0.0)) for _, _, data in H.edges(data=True)]
        history.append({
            "iteration": it,
            "avg_weight": float(np.mean(weights)) if weights else 0.0,
            "avg_ricci": float(np.mean(curvatures)) if curvatures else 0.0,
            "ricci_min": float(np.min(curvatures)) if curvatures else 0.0,
            "ricci_max": float(np.max(curvatures)) if curvatures else 0.0,
            "edges": int(H.number_of_edges()),
        })

        if it == int(iterations):
            break

        for u, v, data in H.edges(data=True):
            w = float(data.get("weight", data.get("distance", 1.0)))
            kappa = float(data.get("ricciCurvature", 0.0))
            new_w = w * (1.0 - float(step_size) * kappa)
            new_w = float(np.clip(new_w, min_weight, max_weight))
            H[u][v]["weight"] = new_w
            H[u][v]["distance"] = new_w
            H[u][v]["flow_weight"] = new_w

        if normalize_mean_weight and H.number_of_edges() > 0:
            current_weights = [float(data.get("weight", 1.0)) for _, _, data in H.edges(data=True)]
            current_mean = float(np.mean(current_weights)) if current_weights else 0.0
            if current_mean > 0:
                scale = target_mean_weight / current_mean
                for u, v in H.edges():
                    w = float(H[u][v].get("weight", 1.0)) * scale
                    w = float(np.clip(w, min_weight, max_weight))
                    H[u][v]["weight"] = w
                    H[u][v]["distance"] = w
                    H[u][v]["flow_weight"] = w

        if recompute_curvature:
            H = compute_ricci_curvature(H, alpha=alpha, method=method, proc=proc)

    return H, pd.DataFrame(history)

def compare_before_after_flow(before: nx.Graph, after: nx.Graph) -> pd.DataFrame:
    """Compare edge weights/curvatures before and after Ricci flow.

    This function is imported by app.py. It is intentionally robust: if an edge
    exists only before or only after, the missing side is filled with NaN.
    """
    edge_keys = set(tuple(sorted((u, v))) for u, v in before.edges()) | set(tuple(sorted((u, v))) for u, v in after.edges())
    rows = []
    for u, v in sorted(edge_keys):
        b = before.get_edge_data(u, v, default={})
        a = after.get_edge_data(u, v, default={})
        before_w = float(b.get("weight", b.get("distance", np.nan))) if b else np.nan
        after_w = float(a.get("weight", a.get("distance", np.nan))) if a else np.nan
        before_k = float(b.get("ricciCurvature", np.nan)) if b else np.nan
        after_k = float(a.get("ricciCurvature", np.nan)) if a else np.nan
        rows.append({
            "u": u,
            "v": v,
            "before_weight": before_w,
            "after_weight": after_w,
            "delta_weight": after_w - before_w if np.isfinite(before_w) and np.isfinite(after_w) else np.nan,
            "before_ricci": before_k,
            "after_ricci": after_k,
            "delta_ricci": after_k - before_k if np.isfinite(before_k) and np.isfinite(after_k) else np.nan,
            "correlation": float(b.get("correlation", a.get("correlation", np.nan))),
        })
    return pd.DataFrame(rows).sort_values("delta_weight", ascending=False, na_position="last")


def plot_ricci_flow_history(history: pd.DataFrame) -> go.Figure:
    """Plot Ricci-flow diagnostics returned by run_ricci_flow."""
    fig = go.Figure()
    if history is None or history.empty:
        fig.update_layout(title="Ricci flow history: no data", height=420)
        return fig
    fig.add_trace(go.Scatter(x=history["iteration"], y=history["avg_weight"], mode="lines+markers", name="Average weight"))
    fig.add_trace(go.Scatter(x=history["iteration"], y=history["avg_ricci"], mode="lines+markers", name="Average Ricci", yaxis="y2"))
    fig.add_trace(go.Scatter(x=history["iteration"], y=history["ricci_min"], mode="lines", name="Ricci min", yaxis="y2"))
    fig.add_trace(go.Scatter(x=history["iteration"], y=history["ricci_max"], mode="lines", name="Ricci max", yaxis="y2"))
    fig.update_layout(
        title="Ricci flow diagnostics",
        height=460,
        xaxis_title="Iteration",
        yaxis_title="Average financial distance / weight",
        yaxis2={"title": "Ricci curvature", "overlaying": "y", "side": "right"},
        hovermode="x unified",
    )
    return fig


# -----------------------------------------------------------------------------
# v10.1 capital-flow extension
# -----------------------------------------------------------------------------

def download_market_data(tickers: Sequence[str], period: str = "1y", interval: str = "1d") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download real market Close, Volume, and dollar-volume = Close * Volume.

    Important: auto_adjust=False is used so prices are the raw Yahoo/market
    Close, not adjusted close.  This makes ``prices.tail()`` match the
    displayed Close more closely than synthetic demo data or adjusted data.
    """
    if yf is None:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")
    yf.set_tz_cache_location(".")
    tickers = parse_tickers(tickers)
    data = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=False,
        group_by="column",
        progress=False,
        threads=True,
    )
    if data.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = data.columns.get_level_values(0)
        prices = data["Close"].copy() if "Close" in lvl0 else data[lvl0[0]].copy()
        volumes = data["Volume"].copy() if "Volume" in lvl0 else pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    else:
        close_col = "Close" if "Close" in data.columns else data.columns[0]
        vol_col = "Volume" if "Volume" in data.columns else None
        prices = data[[close_col]].copy(); prices.columns = [tickers[0]]
        volumes = data[[vol_col]].copy() if vol_col else pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        if vol_col:
            volumes.columns = [tickers[0]]

    prices = prices.dropna(axis=1, how="all").ffill().dropna(how="all")
    volumes = volumes.reindex(index=prices.index, columns=prices.columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dollar_volume = (prices * volumes).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return prices, volumes, dollar_volume


def make_demo_market_data(
    tickers: Sequence[str] = DEFAULT_TICKERS[:8],
    n_days: int = 260,
    seed: int = 7,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Synthetic prices plus IPO-aware volume and dollar-volume."""
    prices = make_demo_prices(tickers=tickers, n_days=n_days, seed=seed)
    rng = np.random.default_rng(seed + 101)
    volumes = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for i, t in enumerate(prices.columns):
        base = rng.lognormal(mean=15.2 + 0.18 * (i % 5), sigma=0.35)
        shock = rng.lognormal(mean=0.0, sigma=0.35, size=len(prices))
        trend = np.linspace(0.85, 1.15 + 0.05 * (i % 3), len(prices))
        volumes[t] = base * shock * trend
        volumes.loc[prices[t].isna(), t] = 0.0
    dollar_volume = (prices * volumes).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return prices, volumes, dollar_volume


def _window_capital_stats(window_dollar_volume: Optional[pd.DataFrame], nodes: Sequence[str]) -> Dict[str, float]:
    if window_dollar_volume is None or window_dollar_volume.empty:
        return {str(n): 0.0 for n in nodes}
    cols = [c for c in nodes if c in window_dollar_volume.columns]
    cap = window_dollar_volume[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).sum(axis=0)
    return {str(k): float(v) for k, v in cap.items()}


def _capital_similarity(a: float, b: float) -> float:
    """0..1 similarity of two positive dollar-volume masses."""
    if a <= 0 or b <= 0:
        return 0.0
    hi = max(a, b); lo = min(a, b)
    return float(lo / hi) if hi > 0 else 0.0


def build_graph_from_window(
    window_returns: pd.DataFrame,
    max_distance: float = 1.05,
    min_abs_corr: float = 0.30,
    keep_top_edges: Optional[int] = None,
    min_node_obs: int = 1,
    min_pair_obs: int = 4,
    graph_mode: str = "threshold",
    k_neighbors: int = 3,
    min_corr: float = 0.05,
    max_bridges: int = 0,
    window_dollar_volume: Optional[pd.DataFrame] = None,
    use_capital_weighting: bool = True,
    capital_alpha: float = 0.35,
) -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    """IPO-aware graph with optional capital-flow weighting.

    The geometric distance still starts from correlation, but a second manifold
    adds market mass: dollar_volume = price * volume.  Edges get attributes:
    edge_capital_flow, capital_similarity, raw_distance, and effective distance.
    """
    min_pair_obs = max(3, int(min_pair_obs))
    min_node_obs = max(1, int(min_node_obs))
    all_cols = list(window_returns.columns)
    node_cols = all_cols  # common-sense display: requested tickers stay visible pre-IPO
    active_cols = [c for c in all_cols if window_returns[c].notna().sum() >= min_node_obs]
    pair_cols = [c for c in all_cols if window_returns[c].notna().sum() >= min_pair_obs]
    clean = window_returns[pair_cols].copy()
    if len(pair_cols) >= 2:
        corr = clean.corr(min_periods=min_pair_obs).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        corr = pd.DataFrame(index=pair_cols, columns=pair_cols, dtype=float)
    raw_dist = financial_distance_from_corr(corr) if len(corr) else pd.DataFrame(index=pair_cols, columns=pair_cols, dtype=float)

    capital_mass = _window_capital_stats(window_dollar_volume, node_cols)
    total_capital = float(sum(max(0.0, v) for v in capital_mass.values()))

    G = nx.Graph()
    for n in node_cols:
        valid_obs = int(window_returns[n].notna().sum()) if n in window_returns.columns else 0
        mass = float(capital_mass.get(str(n), 0.0))
        G.add_node(
            n,
            status="active" if n in active_cols else "inactive_pre_ipo_or_no_data",
            valid_obs=valid_obs,
            capital_mass=mass,
            capital_share=(mass / total_capital) if total_capital > 0 else 0.0,
        )

    def edge_values(u: str, v: str) -> Tuple[float, float, float, float, float]:
        rho = float(corr.loc[u, v])
        d0 = float(raw_dist.loc[u, v])
        mi = float(capital_mass.get(str(u), 0.0)); mj = float(capital_mass.get(str(v), 0.0))
        cap_flow = float(abs(rho) * np.sqrt(max(mi, 0.0) * max(mj, 0.0)))
        cap_sim = _capital_similarity(mi, mj)
        if use_capital_weighting and cap_flow > 0:
            scale = 1.0 + float(capital_alpha) * np.log1p(cap_flow / max(total_capital, 1.0))
            d_eff = d0 / scale
        else:
            d_eff = d0
        return rho, d0, d_eff, cap_flow, cap_sim

    candidates: List[Tuple[str, str, float, float, float, float, float]] = []
    tickers = list(corr.columns)
    mode = str(graph_mode).lower().replace(" ", "")

    if mode in {"knn", "k-nearest", "nearest", "knn+bridges", "knnbridges"} and len(tickers) >= 2:
        dist_for_rank = raw_dist.copy()
        for u in tickers:
            nearest = dist_for_rank.loc[u].drop(index=u, errors="ignore").sort_values().head(max(1, int(k_neighbors)))
            for v in nearest.index:
                rho, d0, de, cf, cs = edge_values(str(u), str(v))
                if np.isfinite(rho) and np.isfinite(de) and rho >= float(min_corr):
                    a, b = sorted((str(u), str(v)))
                    candidates.append((a, b, d0, de, rho, cf, cs))
    else:
        for i, u in enumerate(tickers):
            for v in tickers[i + 1:]:
                rho, d0, de, cf, cs = edge_values(str(u), str(v))
                if np.isfinite(rho) and np.isfinite(de) and de <= max_distance and abs(rho) >= min_abs_corr:
                    candidates.append((str(u), str(v), d0, de, rho, cf, cs))

    # de-duplicate kNN nominations and rank by effective distance
    dedup: Dict[Tuple[str, str], Tuple[str, str, float, float, float, float, float]] = {}
    for u, v, d0, de, rho, cf, cs in candidates:
        key = tuple(sorted((u, v)))
        if key not in dedup or de < dedup[key][3]:
            dedup[key] = (key[0], key[1], d0, de, rho, cf, cs)
    candidates = list(dedup.values())
    if keep_top_edges is not None and keep_top_edges > 0:
        candidates = sorted(candidates, key=lambda x: x[3])[: int(keep_top_edges)]

    for u, v, d0, de, rho, cf, cs in candidates:
        G.add_edge(
            u, v,
            weight=float(de), distance=float(de), raw_distance=float(d0),
            correlation=float(rho), edge_capital_flow=float(cf), capital_similarity=float(cs),
            edge_source="capital_weighted" if use_capital_weighting else "correlation_only",
        )

    if mode in {"knn+bridges", "knnbridges"} and max_bridges > 0 and len(corr.columns) >= 2:
        existing = {tuple(sorted((u, v))) for u, v in G.edges()}
        bridge_candidates = []
        for i, u in enumerate(tickers):
            for v in tickers[i + 1:]:
                key = tuple(sorted((str(u), str(v))))
                if key in existing:
                    continue
                rho, d0, de, cf, cs = edge_values(str(u), str(v))
                if np.isfinite(rho) and rho > 0:
                    bridge_candidates.append((str(u), str(v), d0, de, rho, cf, cs))
        for u, v, d0, de, rho, cf, cs in sorted(bridge_candidates, key=lambda x: x[3])[: int(max_bridges)]:
            G.add_edge(u, v, weight=float(de), distance=float(de), raw_distance=float(d0), correlation=float(rho), edge_capital_flow=float(cf), capital_similarity=float(cs), bridge=True, edge_source="capital_bridge")

    return G, corr, raw_dist


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
    graph_mode: str = "threshold",
    k_neighbors: int = 3,
    min_corr: float = 0.05,
    max_bridges: int = 0,
    dollar_volume: Optional[pd.DataFrame] = None,
    use_capital_weighting: bool = True,
    capital_alpha: float = 0.35,
) -> FrameData:
    window_returns = returns.iloc[start : start + window_size]
    window_dv = dollar_volume.iloc[start : start + window_size] if dollar_volume is not None and not dollar_volume.empty else None
    end_date = str(window_returns.index[-1])[:19]
    G0, corr, dist = build_graph_from_window(
        window_returns,
        max_distance=max_distance,
        min_abs_corr=min_abs_corr,
        keep_top_edges=keep_top_edges,
        min_node_obs=min_node_obs,
        min_pair_obs=min_pair_obs,
        graph_mode=graph_mode,
        k_neighbors=k_neighbors,
        min_corr=min_corr,
        max_bridges=max_bridges,
        window_dollar_volume=window_dv,
        use_capital_weighting=use_capital_weighting,
        capital_alpha=capital_alpha,
    )
    G = compute_ricci_curvature(G0, alpha=alpha, method=method, proc=proc)
    node_cluster = compute_components(G)
    stats = compute_window_stats(G, end_date)
    return FrameData(G=G, node_cluster=node_cluster, stats=stats, corr=corr, dist=dist)


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
    frames = [build_frame(returns, start=s, window_size=window_size, **kwargs) for s in starts]
    return frames, starts


def capital_flow_table(G: nx.Graph) -> pd.DataFrame:
    rows = []
    total = sum(float(d.get("edge_capital_flow", 0.0)) for _, _, d in G.edges(data=True))
    for u, v, data in G.edges(data=True):
        cf = float(data.get("edge_capital_flow", 0.0))
        rows.append({
            "u": u, "v": v,
            "edge_capital_flow": cf,
            "edge_flow_share": cf / total if total > 0 else 0.0,
            "capital_similarity": float(data.get("capital_similarity", np.nan)),
            "effective_distance": float(data.get("distance", data.get("weight", np.nan))),
            "raw_corr_distance": float(data.get("raw_distance", np.nan)),
            "correlation": float(data.get("correlation", np.nan)),
            "ricciCurvature": float(data.get("ricciCurvature", 0.0)),
        })
    return pd.DataFrame(rows).sort_values("edge_capital_flow", ascending=False) if rows else pd.DataFrame()


def node_capital_table(G: nx.Graph) -> pd.DataFrame:
    rows = []
    for n, data in G.nodes(data=True):
        rows.append({
            "ticker": n,
            "status": data.get("status", "active"),
            "valid_obs": int(data.get("valid_obs", 0)),
            "capital_mass": float(data.get("capital_mass", 0.0)),
            "capital_share": float(data.get("capital_share", 0.0)),
            "degree": int(G.degree(n)),
        })
    return pd.DataFrame(rows).sort_values("capital_mass", ascending=False) if rows else pd.DataFrame()


def cluster_capital_table(G: nx.Graph, node_cluster: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    if node_cluster is None:
        node_cluster = compute_components(G)
    rows = []
    for n, data in G.nodes(data=True):
        rows.append({"cluster": int(node_cluster.get(n, -1)), "ticker": n, "capital_mass": float(data.get("capital_mass", 0.0))})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    total = float(df["capital_mass"].sum())
    out = df.groupby("cluster").agg(
        tickers=("ticker", lambda s: ", ".join(map(str, sorted(s)))),
        capital_mass=("capital_mass", "sum"),
        nodes=("ticker", "count"),
    ).reset_index()
    out["capital_share"] = out["capital_mass"] / total if total > 0 else 0.0
    return out.sort_values("capital_mass", ascending=False)


def plot_capital_flow_bars(flow_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    fig = go.Figure()
    if flow_df is None or flow_df.empty:
        fig.update_layout(title="Capital-flow edges: no data", height=420)
        return fig
    d = flow_df.head(int(top_n)).copy()
    d["edge"] = d["u"].astype(str) + "-" + d["v"].astype(str)
    fig.add_trace(go.Bar(
        x=d["edge"], y=d["edge_capital_flow"],
        text=[f"{100*x:.1f}%" for x in d["edge_flow_share"]], textposition="outside",
        hovertext=[f"corr={r:.3f}<br>Ricci={k:.3f}<br>effective d={ed:.3f}" for r,k,ed in zip(d["correlation"], d["ricciCurvature"], d["effective_distance"])],
        hoverinfo="x+y+text", name="Capital flow",
    ))
    fig.update_layout(title="Top capital-flow transport edges", height=430, xaxis_title="Edge", yaxis_title="Dollar-volume weighted flow", margin={"l":40,"r":20,"t":60,"b":100})
    return fig


def summarize_edges(G: nx.Graph) -> pd.DataFrame:
    rows = []
    for u, v, data in G.edges(data=True):
        rows.append({
            "u": u,
            "v": v,
            "distance": float(data.get("distance", data.get("weight", np.nan))),
            "raw_distance": float(data.get("raw_distance", np.nan)),
            "correlation": float(data.get("correlation", np.nan)),
            "ricciCurvature": float(data.get("ricciCurvature", 0.0)),
            "edge_capital_flow": float(data.get("edge_capital_flow", 0.0)),
            "capital_similarity": float(data.get("capital_similarity", np.nan)),
            "edge_source": data.get("edge_source", ""),
        })
    return pd.DataFrame(rows).sort_values("distance") if rows else pd.DataFrame(columns=["u","v","distance","raw_distance","correlation","ricciCurvature","edge_capital_flow","capital_similarity","edge_source"])

# -----------------------------------------------------------------------------
# v10.4 Dynamic 3D Ricci-capital manifold
# -----------------------------------------------------------------------------

def compute_stable_layout_3d(
    base_graph: nx.Graph,
    seed: int = 42,
    scale: float = 1.0,
    layout_k: float = 0.45,
    iterations: int = 600,
) -> Dict[str, Tuple[float, float, float]]:
    """Stable 3D spring layout used as the common x/y topology backbone.

    The z coordinate returned here is only a geometric fallback.  The final
    Ricci-capital manifold normally replaces z with frame-by-frame Ricci stress
    or capital mass so the vertical axis has financial meaning.
    """
    if base_graph.number_of_nodes() == 0:
        return {}
    if base_graph.number_of_edges() == 0:
        pos2 = nx.circular_layout(base_graph, scale=scale)
        return {n: (float(x), float(y), 0.0) for n, (x, y) in pos2.items()}
    pos = nx.spring_layout(
        base_graph,
        dim=3,
        seed=int(seed),
        weight="weight",
        k=float(layout_k),
        iterations=int(iterations),
        scale=float(scale),
    )
    return {n: (float(x), float(y), float(z)) for n, (x, y, z) in pos.items()}


def _node_ricci_stress(G: nx.Graph, node: str) -> float:
    """Node stress = negative mean incident Ricci curvature.

    High positive values are fragile / negative-curvature zones.  Coherent
    positive-curvature basins get lower z values.
    """
    vals: List[float] = []
    for nbr in G.neighbors(node) if node in G else []:
        vals.append(float(G[node][nbr].get("ricciCurvature", 0.0)))
    if not vals:
        return 0.0
    return float(-np.mean(vals))


def _node_z_value(G: nx.Graph, node: str, z_mode: str = "ricci_stress") -> float:
    mode = str(z_mode).lower().replace(" ", "_")
    if mode in {"capital", "capital_mass", "log_capital"}:
        mass = float(G.nodes[node].get("capital_mass", 0.0)) if node in G.nodes else 0.0
        return float(np.log1p(max(0.0, mass)))
    if mode in {"spring", "layout"}:
        return np.nan
    # default: high z means stress / negative curvature.
    return _node_ricci_stress(G, node)


def _all_nodes_from_frames(frames: Sequence[FrameData], positions: Optional[Dict[str, Tuple[float, ...]]] = None) -> List[str]:
    nodes = set()
    if positions:
        nodes.update(str(n) for n in positions.keys())
    for fd in frames:
        nodes.update(str(n) for n in fd.G.nodes())
    return sorted(nodes)


def _edge_universe_from_frames(frames: Sequence[FrameData]) -> List[Tuple[str, str]]:
    edges = set()
    for fd in frames:
        for u, v in fd.G.edges():
            edges.add(tuple(sorted((str(u), str(v)))))
    return sorted(edges)


def _capital_node_sizes(G: nx.Graph, nodes: Sequence[str], min_size: float = 5.0, max_size: float = 26.0) -> List[float]:
    masses = np.array([float(G.nodes[n].get("capital_mass", 0.0)) if n in G.nodes else 0.0 for n in nodes], dtype=float)
    vals = np.log1p(np.maximum(masses, 0.0))
    if np.nanmax(vals) > np.nanmin(vals):
        norm = (vals - np.nanmin(vals)) / (np.nanmax(vals) - np.nanmin(vals))
    else:
        norm = np.zeros_like(vals)
    return [float(min_size + (max_size - min_size) * x) for x in norm]


def _transport_edge_widths(G: nx.Graph, all_edges: Sequence[Tuple[str, str]], min_width: float = 1.0, max_width: float = 8.0) -> Dict[Tuple[str, str], float]:
    vals = []
    for u, v in all_edges:
        vals.append(float(G[u][v].get("edge_capital_flow", 0.0)) if G.has_edge(u, v) else 0.0)
    arr = np.log1p(np.maximum(np.array(vals, dtype=float), 0.0))
    if arr.size and np.nanmax(arr) > np.nanmin(arr):
        norm = (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))
    else:
        norm = np.zeros_like(arr)
    return {tuple(e): float(min_width + (max_width - min_width) * n) for e, n in zip(all_edges, norm)}


def _manifold_3d_ranges(
    frames: Sequence[FrameData],
    positions_3d: Dict[str, Tuple[float, float, float]],
    z_mode: str = "ricci_stress",
) -> Tuple[List[float], List[float], List[float]]:
    nodes = _all_nodes_from_frames(frames, positions_3d)
    xs = [float(positions_3d.get(n, (0.0, 0.0, 0.0))[0]) for n in nodes]
    ys = [float(positions_3d.get(n, (0.0, 0.0, 0.0))[1]) for n in nodes]
    zs: List[float] = []
    for fd in frames:
        for n in nodes:
            if str(z_mode).lower() in {"spring", "layout"}:
                zs.append(float(positions_3d.get(n, (0.0, 0.0, 0.0))[2]))
            elif n in fd.G.nodes:
                zs.append(float(_node_z_value(fd.G, n, z_mode)))
            else:
                zs.append(0.0)
    def pad_range(vals: Sequence[float], pad_ratio: float = 0.15) -> List[float]:
        finite = [float(v) for v in vals if np.isfinite(v)]
        if not finite:
            return [-1.0, 1.0]
        lo, hi = min(finite), max(finite)
        if abs(hi - lo) < 1e-9:
            return [lo - 1.0, hi + 1.0]
        pad = (hi - lo) * pad_ratio
        return [lo - pad, hi + pad]
    return pad_range(xs), pad_range(ys), pad_range(zs, 0.25)


def _frame_traces_3d(
    fd: FrameData,
    positions_3d: Dict[str, Tuple[float, float, float]],
    all_nodes: Sequence[str],
    all_edges: Sequence[Tuple[str, str]],
    z_mode: str = "ricci_stress",
    show_edge_weight_labels: bool = True,
    edge_label_top_n: int = 30,
) -> List[go.Scatter3d]:
    G = fd.G
    traces: List[go.Scatter3d] = []
    widths = _transport_edge_widths(G, all_edges)

    # One trace per edge keeps Ricci color and capital-transport width meaningful.
    current_edges_by_distance = sorted(
        G.edges(data=True),
        key=lambda e: float(e[2].get("distance", e[2].get("weight", np.inf))),
    )[: max(0, int(edge_label_top_n))]
    labeled_pairs = {tuple(sorted((str(u), str(v)))) for u, v, _ in current_edges_by_distance} if show_edge_weight_labels else set()
    label_x: List[float] = []
    label_y: List[float] = []
    label_z: List[float] = []
    label_text: List[str] = []
    label_hover: List[str] = []

    def node_xyz(n: str) -> Tuple[float, float, float]:
        x, y, z_layout = positions_3d.get(n, (0.0, 0.0, 0.0))
        if str(z_mode).lower() in {"spring", "layout"}:
            z = float(z_layout)
        elif n in G.nodes:
            z = float(_node_z_value(G, n, z_mode))
        else:
            z = 0.0
        return float(x), float(y), float(z)

    for u, v in all_edges:
        if G.has_edge(u, v):
            data = G[u][v]
            x0, y0, z0 = node_xyz(u)
            x1, y1, z1 = node_xyz(v)
            kappa = float(data.get("ricciCurvature", 0.0))
            rho = float(data.get("correlation", np.nan))
            d = float(data.get("distance", data.get("weight", np.nan)))
            flow = float(data.get("edge_capital_flow", 0.0))
            traces.append(go.Scatter3d(
                x=[x0, x1], y=[y0, y1], z=[z0, z1],
                mode="lines",
                line={"width": widths.get(tuple(sorted((u, v))), 2.0), "color": ricci_to_hex(kappa)},
                hoverinfo="text",
                text=(f"{u}-{v}<br>weight={d:.4f}<br>correlation={rho:.4f}<br>"
                      f"Ricci={kappa:.4f}<br>capital transport={flow:,.0f}"),
                showlegend=False,
            ))
            if tuple(sorted((u, v))) in labeled_pairs:
                label_x.append((x0 + x1) / 2.0)
                label_y.append((y0 + y1) / 2.0)
                label_z.append((z0 + z1) / 2.0)
                label_text.append(f"w={d:.2f}")
                label_hover.append(f"{u}-{v}<br>weight={d:.4f}<br>transport={flow:,.0f}")
        else:
            traces.append(go.Scatter3d(
                x=[None, None], y=[None, None], z=[None, None],
                mode="lines", line={"width": 0.0, "color": "rgba(0,0,0,0)"},
                hoverinfo="skip", showlegend=False,
            ))

    traces.append(go.Scatter3d(
        x=label_x, y=label_y, z=label_z,
        mode="text", text=label_text,
        textfont={"size": 10, "color": "#111111"},
        hoverinfo="text", hovertext=label_hover,
        showlegend=False,
    ))

    degrees = dict(G.degree())
    node_x: List[float] = []
    node_y: List[float] = []
    node_z: List[float] = []
    labels: List[str] = []
    hover: List[str] = []
    colors: List[str] = []
    sizes = _capital_node_sizes(G, all_nodes)
    for n in all_nodes:
        x, y, z = node_xyz(n)
        node_x.append(x); node_y.append(y); node_z.append(z); labels.append(str(n))
        status = str(G.nodes[n].get("status", "active")) if n in G.nodes else "inactive_pre_ipo_or_no_data"
        mass = float(G.nodes[n].get("capital_mass", 0.0)) if n in G.nodes else 0.0
        share = float(G.nodes[n].get("capital_share", 0.0)) if n in G.nodes else 0.0
        cid = fd.node_cluster.get(n, -1)
        if status.startswith("inactive"):
            colors.append("#BDBDBD")
        else:
            colors.append(COMPONENT_COLORS[int(cid) % len(COMPONENT_COLORS)])
        hover.append(
            f"Ticker: {n}<br>Status: {status}<br>Component: {cid}<br>Degree: {degrees.get(n, 0)}<br>"
            f"Capital mass: {mass:,.0f}<br>Capital share: {share:.3%}<br>z stress: {z:.4f}"
        )
    traces.append(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode="markers+text",
        text=labels,
        textposition="top center",
        textfont={"size": 12, "color": "#111111"},
        marker={"size": sizes, "color": colors, "opacity": 0.92, "line": {"width": 1.2, "color": "#111111"}},
        hoverinfo="text", hovertext=hover,
        showlegend=False,
    ))
    return traces


def _stats_annotation_3d(stats: WindowStats) -> dict:
    return {
        "text": (
            f"<b>Date:</b> {stats.end_date}<br>"
            f"<b>Avg Ricci:</b> {stats.avg_ricci:.4f}<br>"
            f"<b>Clusters:</b> {stats.num_clusters}<br>"
            f"<b>Nodes:</b> {stats.num_nodes} &nbsp; <b>Edges:</b> {stats.num_edges}<br>"
            f"<b>Density:</b> {stats.density:.3f}<br>"
            f"<b>HMM:</b> {stats.hmm_state} - {stats.regime_name}"
        ),
        "xref": "paper", "yref": "paper", "x": 1.02, "y": 0.98,
        "showarrow": False, "align": "left", "bordercolor": "#999", "borderwidth": 1,
        "bgcolor": "rgba(255,255,255,0.92)", "font": {"size": 13, "color": "#111"},
    }


def build_3d_ricci_capital_animation(
    frames: Sequence[FrameData],
    positions_3d: Dict[str, Tuple[float, float, float]],
    frame_duration_ms: int = 700,
    title: str = "Dynamic 3D Ricci-capital manifold",
    z_mode: str = "ricci_stress",
    show_edge_weight_labels: bool = True,
    edge_label_top_n: int = 30,
) -> go.Figure:
    """Interactive rotating 3D animation of the final Ricci-capital manifold.

    Visual encoding:
    x,y = stable topology; z = Ricci stress by default; node size = dollar-volume
    capital mass; edge width = capital transport; edge color = Ricci curvature;
    animation = rolling-window market evolution.
    """
    if not frames:
        return go.Figure()
    all_nodes = _all_nodes_from_frames(frames, positions_3d)
    all_edges = _edge_universe_from_frames(frames)
    x_range, y_range, z_range = _manifold_3d_ranges(frames, positions_3d, z_mode=z_mode)

    initial_data = _frame_traces_3d(
        frames[0], positions_3d, all_nodes, all_edges,
        z_mode=z_mode,
        show_edge_weight_labels=show_edge_weight_labels,
        edge_label_top_n=edge_label_top_n,
    )
    trace_ids = list(range(len(initial_data)))
    fig = go.Figure(data=initial_data)
    fig.frames = [
        go.Frame(
            name=str(i),
            data=_frame_traces_3d(
                fd, positions_3d, all_nodes, all_edges,
                z_mode=z_mode,
                show_edge_weight_labels=show_edge_weight_labels,
                edge_label_top_n=edge_label_top_n,
            ),
            traces=trace_ids,
            layout=go.Layout(annotations=[_stats_annotation_3d(fd.stats)]),
        )
        for i, fd in enumerate(frames)
    ]
    steps = [
        {
            "method": "animate",
            "label": str(i + 1),
            "args": [[str(i)], {"mode": "immediate", "frame": {"duration": frame_duration_ms, "redraw": True}, "transition": {"duration": 0}}],
        }
        for i in range(len(frames))
    ]
    fig.update_layout(
        title=title,
        height=850,
        showlegend=False,
        hovermode="closest",
        annotations=[_stats_annotation_3d(frames[0].stats)],
        scene={
            "xaxis": {"visible": False, "range": x_range},
            "yaxis": {"visible": False, "range": y_range},
            "zaxis": {"visible": False, "range": z_range},
            "camera": {"eye": {"x": 1.8, "y": 1.8, "z": 1.15}},
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 220, "t": 60, "b": 70},
        plot_bgcolor="white",
        paper_bgcolor="white",
        updatemenus=[{
            "type": "buttons", "showactive": False,
            "x": 0.02, "y": -0.04, "xanchor": "left", "yanchor": "top",
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": frame_duration_ms, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
            ],
        }],
        sliders=[{"active": 0, "currentvalue": {"prefix": "Frame ", "font": {"size": 14}}, "pad": {"t": 40}, "steps": steps}],
    )
    return fig


def visualize_network_3d(
    G: nx.Graph,
    positions_3d: Optional[Dict[str, Tuple[float, float, float]]] = None,
    title: str = "3D Ricci-capital manifold",
    node_cluster: Optional[Dict[str, int]] = None,
    z_mode: str = "ricci_stress",
) -> go.Figure:
    """Static 3D view for one frame; use build_3d_ricci_capital_animation for rolling time."""
    if positions_3d is None:
        positions_3d = compute_stable_layout_3d(G)
    fd = FrameData(G=G, node_cluster=node_cluster or compute_components(G), stats=compute_window_stats(G, "selected"), corr=pd.DataFrame(), dist=pd.DataFrame())
    all_nodes = _all_nodes_from_frames([fd], positions_3d)
    all_edges = _edge_universe_from_frames([fd])
    x_range, y_range, z_range = _manifold_3d_ranges([fd], positions_3d, z_mode=z_mode)
    fig = go.Figure(data=_frame_traces_3d(fd, positions_3d, all_nodes, all_edges, z_mode=z_mode))
    fig.update_layout(
        title=title,
        height=800,
        showlegend=False,
        scene={
            "xaxis": {"visible": False, "range": x_range},
            "yaxis": {"visible": False, "range": y_range},
            "zaxis": {"visible": False, "range": z_range},
            "camera": {"eye": {"x": 1.8, "y": 1.8, "z": 1.15}},
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 0, "t": 50, "b": 20},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def surgery_risk_direction_table(G: nx.Graph) -> pd.DataFrame:
    """Direction score only: do not cut the graph in real-market data.

    High score means a capital-heavy negative-curvature bridge may separate in a
    later regime, but the graph remains intact because markets are usually on the
    path toward a state rather than at the final resolved state.
    """
    rows = []
    for u, v, data in G.edges(data=True):
        kappa = float(data.get("ricciCurvature", 0.0))
        d = float(data.get("distance", data.get("weight", 1.0)))
        flow = float(data.get("edge_capital_flow", 0.0))
        risk = max(0.0, -kappa) * max(0.0, d) * np.log1p(max(0.0, flow))
        rows.append({
            "u": u, "v": v,
            "ricciCurvature": kappa,
            "distance": d,
            "edge_capital_flow": flow,
            "surgery_risk_direction": float(risk),
            "interpretation": "possible future separation" if risk > 0 else "normal / coherent",
        })
    return pd.DataFrame(rows).sort_values("surgery_risk_direction", ascending=False) if rows else pd.DataFrame(columns=["u", "v", "ricciCurvature", "distance", "edge_capital_flow", "surgery_risk_direction", "interpretation"])
