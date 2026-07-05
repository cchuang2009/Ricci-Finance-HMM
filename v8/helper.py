"""
helper.py
Graduate lecture helpers for: Perelman / Ricci ideas in financial networks.

Version v8 adds discrete Ricci flow diagnostics.

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
    "RGTI", "NBIS",
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


def make_demo_prices(tickers: Sequence[str] = DEFAULT_TICKERS[:8], n_days: int = 260, seed: int = 7) -> pd.DataFrame:
    """Synthetic prices for offline lectures or when yfinance is unavailable."""
    rng = np.random.default_rng(seed)
    tickers = parse_tickers(tickers)
    # Three latent market factors: broad AI, memory/equipment, speculative beta.
    factors = rng.normal(0, [0.010, 0.014, 0.020], size=(n_days, 3))
    loadings = rng.uniform(-0.3, 1.2, size=(len(tickers), 3))
    noise = rng.normal(0, 0.012, size=(n_days, len(tickers)))
    returns = factors @ loadings.T + noise
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    return pd.DataFrame(prices, index=idx, columns=tickers)


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


def build_graph_from_window(
    window_returns: pd.DataFrame,
    max_distance: float = 1.05,
    min_abs_corr: float = 0.30,
    keep_top_edges: Optional[int] = None,
    min_node_obs: int = 1,
    min_pair_obs: int = 4,
) -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    """Build an IPO-aware correlation-distance graph from one rolling return window."""
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
        G.add_edge(u, v, weight=d, distance=d, correlation=rho)
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
) -> go.Figure:
    """Visualize one financial network with edge color = Ricci curvature."""
    if positions is None:
        positions = compute_stable_layout(G)
    if node_cluster is None:
        node_cluster = compute_components(G)

    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = positions.get(u, (0.0, 0.0))
        x1, y1 = positions.get(v, (0.0, 0.0))
        kappa = float(data.get("ricciCurvature", 0.0))
        rho = float(data.get("correlation", np.nan))
        d = float(data.get("distance", data.get("weight", np.nan)))
        edge_traces.append(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line={"width": 2.0, "color": ricci_to_hex(kappa)},
            hoverinfo="text",
            text=f"{u}-{v}<br>distance={d:.4f}<br>correlation={rho:.4f}<br>Ricci={kappa:.4f}",
            showlegend=False,
        ))

    degrees = dict(G.degree())
    node_x, node_y, labels, hover, colors, sizes = [], [], [], [], [], []
    for node in sorted(G.nodes()):
        x, y = positions.get(node, (0.0, 0.0))
        cid = node_cluster.get(node, 0)
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


def rolling_feature_table(frames: Sequence[FrameData]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "date": fd.stats.end_date,
            "avg_ricci": fd.stats.avg_ricci,
            "ricci_std": fd.stats.ricci_std,
            "clusters": fd.stats.num_clusters,
            "largest_component_ratio": fd.stats.largest_component / max(fd.stats.num_nodes, 1),
            "nodes": fd.stats.num_nodes,
            "edges": fd.stats.num_edges,
            "density": fd.stats.density,
        }
        for fd in frames
    ])


def compute_hmm_regimes(
    frames: Sequence[FrameData],
    returns: pd.DataFrame,
    starts: Sequence[int],
    window_size: int,
    n_components: int = 3,
    forward_days: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """Fit a Gaussian HMM to rolling Ricci-network features."""
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

    feature_cols = ["avg_ricci", "ricci_std", "clusters", "largest_component_ratio", "edges", "density"]
    if GaussianHMM is None or StandardScaler is None or len(feature_df) < max(8, n_components * 4):
        feature_df["hmm_state"] = -1
        feature_df["regime_name"] = "HMM unavailable or too few frames"
        return feature_df, {-1: "HMM unavailable or too few frames"}

    X = feature_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xs = StandardScaler().fit_transform(X)
    model = GaussianHMM(n_components=int(n_components), covariance_type="full", n_iter=1000, random_state=int(random_state))
    model.fit(Xs)
    states = model.predict(Xs)
    feature_df["hmm_state"] = states

    summary = feature_df.groupby("hmm_state").agg(
        avg_ricci=("avg_ricci", "mean"), density=("density", "mean"),
        largest_ratio=("largest_component_ratio", "mean"), next_return=(fwd_col, "mean"),
        count=("date", "count"),
    )
    regime_names: Dict[int, str] = {}
    for state, row in summary.iterrows():
        if row["avg_ricci"] < summary["avg_ricci"].median() and row["largest_ratio"] < summary["largest_ratio"].median():
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
    iterations: int = 8,
    step_size: float = 0.25,
    alpha: float = 0.5,
    method: str = "OTD",
    proc: int = 1,
    normalize_mean_weight: bool = True,
    min_weight: float = 1e-4,
) -> Tuple[nx.Graph, pd.DataFrame]:
    """Run a discrete Ollivier-Ricci flow on financial network edge distances.

    Interpretation for finance
    --------------------------
    Edges use financial distance as weight. Ricci flow updates each edge by

        w_{t+1}(u,v) = w_t(u,v) * (1 - step_size * kappa_t(u,v))

    where kappa_t is Ricci curvature. Positive-curvature edges shrink, reinforcing
    coherent clusters. Negative-curvature bridge/stress edges lengthen, exposing
    fragile connections. The flow is a diagnostic geometry transform, not a price
    forecast by itself.

    Returns
    -------
    flowed_graph:
        Final graph with updated weights/distances and latest curvature.
    history:
        Per-iteration summary table.
    """
    H = G.copy()
    history: List[Dict[str, float]] = []
    initial_mean = np.mean([float(d.get("weight", 1.0)) for _, _, d in H.edges(data=True)]) if H.number_of_edges() else 1.0

    for it in range(int(iterations) + 1):
        H = compute_ricci_curvature(H, alpha=alpha, method=method, proc=proc)
        kappas = np.array([float(d.get("ricciCurvature", 0.0)) for _, _, d in H.edges(data=True)], dtype=float)
        weights = np.array([float(d.get("weight", d.get("distance", 1.0))) for _, _, d in H.edges(data=True)], dtype=float)
        comps = list(nx.connected_components(H))
        history.append({
            "iteration": it,
            "avg_ricci": float(kappas.mean()) if len(kappas) else 0.0,
            "ricci_std": float(kappas.std()) if len(kappas) else 0.0,
            "avg_weight": float(weights.mean()) if len(weights) else 0.0,
            "weight_std": float(weights.std()) if len(weights) else 0.0,
            "clusters": float(len(comps)),
            "largest_component": float(max((len(c) for c in comps), default=0)),
            "density": float(nx.density(H)) if H.number_of_nodes() > 1 else 0.0,
        })
        if it == int(iterations):
            break
        for u, v, data in H.edges(data=True):
            w = float(data.get("weight", data.get("distance", 1.0)))
            kappa = float(data.get("ricciCurvature", 0.0))
            new_w = max(float(min_weight), w * (1.0 - float(step_size) * kappa))
            data["weight"] = new_w
            data["distance"] = new_w
        if normalize_mean_weight and H.number_of_edges() > 0:
            current_mean = np.mean([float(d.get("weight", 1.0)) for _, _, d in H.edges(data=True)])
            if current_mean > 0:
                scale = initial_mean / current_mean
                for _, _, data in H.edges(data=True):
                    data["weight"] = float(data["weight"]) * scale
                    data["distance"] = float(data["distance"]) * scale
    return H, pd.DataFrame(history)


def plot_ricci_flow_history(flow_history: pd.DataFrame) -> go.Figure:
    """Plot Ricci-flow diagnostics through iterations."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=flow_history["iteration"], y=flow_history["avg_ricci"], mode="lines+markers", name="Average Ricci"))
    fig.add_trace(go.Scatter(x=flow_history["iteration"], y=flow_history["avg_weight"], mode="lines+markers", name="Average edge distance"))
    fig.add_trace(go.Scatter(x=flow_history["iteration"], y=flow_history["clusters"], mode="lines+markers", name="Clusters", yaxis="y2"))
    fig.update_layout(
        title="Ricci flow diagnostics",
        height=500,
        xaxis_title="Flow iteration",
        yaxis_title="Curvature / distance",
        yaxis2={"title": "Clusters", "overlaying": "y", "side": "right"},
        hovermode="x unified",
    )
    return fig


def compare_before_after_flow(G_before: nx.Graph, G_after: nx.Graph) -> pd.DataFrame:
    """Compare edge distances and curvature before/after Ricci flow."""
    rows = []
    for u, v in G_before.edges():
        if not G_after.has_edge(u, v):
            continue
        b = G_before[u][v]
        a = G_after[u][v]
        rows.append({
            "u": u,
            "v": v,
            "distance_before": float(b.get("distance", b.get("weight", np.nan))),
            "distance_after": float(a.get("distance", a.get("weight", np.nan))),
            "distance_change": float(a.get("distance", a.get("weight", np.nan))) - float(b.get("distance", b.get("weight", np.nan))),
            "ricci_before": float(b.get("ricciCurvature", 0.0)),
            "ricci_after": float(a.get("ricciCurvature", 0.0)),
            "correlation": float(b.get("correlation", np.nan)),
        })
    return pd.DataFrame(rows).sort_values("distance_change") if rows else pd.DataFrame()
