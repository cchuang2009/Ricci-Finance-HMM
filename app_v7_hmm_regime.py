"""
app_v7_hmm_regime.py
Rolling Ricci financial network with Plotly animation, IPO nodes, and HMM hidden market regimes.

Main fixes over PyVis / v3
--------------------------
1. Plotly animation slider + Play button for true time-elapsed network changes.
2. Edge hover shows financial-distance weight, correlation, and Ricci curvature.
3. Optional visible edge-weight labels at edge midpoints.
4. Node labels remain visible during Plotly animation by using fixed trace counts.
5. Stable node positions across all frames.
6. New IPO / late-start tickers can enter after enough valid observations.
7. Optional MP4 export using MoviePy + Kaleido.
8. IPO tickers can display as isolated nodes before enough pairwise observations exist.
9. HMM regime detection from Ricci-network features and market volatility.

Run
---
pip install streamlit yfinance pandas numpy networkx plotly matplotlib GraphRicciCurvature pot networkit
# optional MP4 export:
pip install moviepy kaleido
streamlit run app_v7_hmm_regime.py
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

try:
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler
except Exception:
    GaussianHMM = None
    StandardScaler = None

try:
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
except Exception:
    OllivierRicci = None


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
    num_clusters: int
    largest_component: int
    num_nodes: int
    num_edges: int
    density: float


@dataclass
class FrameData:
    G: nx.Graph
    node_cluster: Dict[str, int]
    stats: WindowStats
    corr: pd.DataFrame
    dist: pd.DataFrame


@st.cache_data(show_spinner=False)
def download_prices(tickers: Tuple[str, ...], period: str, interval: str) -> pd.DataFrame:
    data = yf.download(
        list(tickers),
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
            field = data.columns.get_level_values(0)[0]
            prices = data[field].copy()
    else:
        close_col = "Close" if "Close" in data.columns else data.columns[0]
        prices = data[[close_col]].copy()
        prices.columns = [tickers[0]]

    prices = prices.dropna(axis=1, how="all").ffill().dropna(how="all")
    return prices


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.replace([np.inf, -np.inf], np.nan)
    returns = np.log(prices / prices.shift(1))
    return returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")


def parse_tickers(text: str) -> List[str]:
    out: List[str] = []
    for raw in text.replace("\n", ",").split(","):
        ticker = raw.strip().upper()
        if ticker:
            out.append(ticker)
    return list(dict.fromkeys(out))


def financial_distance_from_corr(corr: pd.DataFrame) -> pd.DataFrame:
    clipped = corr.clip(lower=-1.0, upper=1.0)
    dist = np.sqrt(2.0 * (1.0 - clipped))
    return pd.DataFrame(dist, index=corr.index, columns=corr.columns)


def build_graph_from_window(
    window_returns: pd.DataFrame,
    max_distance: float,
    min_abs_corr: float,
    keep_top_edges: Optional[int],
    min_valid_ratio: float = 0.80,
    min_node_obs: int = 1,
    min_pair_obs: int = 4,
) -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    # IPO-aware: separate node appearance from pairwise edge calculation.
    # A late-start ticker can display as an isolated node once it has min_node_obs
    # returns in the current window. Edges require min_pair_obs overlapping returns.
    min_pair_obs = max(3, int(min_pair_obs))
    min_node_obs = max(1, int(min_node_obs))

    node_cols = [c for c in window_returns.columns if window_returns[c].notna().sum() >= min_node_obs]
    pair_cols = [c for c in window_returns.columns if window_returns[c].notna().sum() >= min_pair_obs]

    clean = window_returns[pair_cols].copy()
    if len(pair_cols) >= 2:
        corr = clean.corr(min_periods=min_pair_obs).replace([np.inf, -np.inf], np.nan)
    else:
        corr = pd.DataFrame(index=pair_cols, columns=pair_cols, dtype=float)
    corr = corr.fillna(0.0)
    dist = financial_distance_from_corr(corr)

    G = nx.Graph()
    G.add_nodes_from(node_cols)
    tickers = list(corr.columns)

    candidates: List[Tuple[str, str, float, float]] = []
    for i, u in enumerate(tickers):
        for j in range(i + 1, len(tickers)):
            v = tickers[j]
            rho = float(corr.iloc[i, j])
            d = float(dist.iloc[i, j])
            if not np.isfinite(rho) or not np.isfinite(d):
                continue
            if d <= max_distance and abs(rho) >= min_abs_corr:
                candidates.append((u, v, d, rho))

    # Smaller distance = stronger positive correlation under d=sqrt(2(1-rho)).
    if keep_top_edges is not None and keep_top_edges > 0:
        candidates = sorted(candidates, key=lambda x: x[2])[:keep_top_edges]

    for u, v, d, rho in candidates:
        G.add_edge(u, v, weight=d, distance=d, correlation=rho)

    return G, corr, dist


def compute_ricci_curvature(G: nx.Graph, alpha: float, method: str, proc: int) -> nx.Graph:
    H = G.copy()
    if H.number_of_edges() == 0:
        return H

    if OllivierRicci is None:
        for u, v in H.edges():
            H[u][v]["ricciCurvature"] = 0.0
        return H

    try:
        orc = OllivierRicci(
            H,
            alpha=alpha,
            method=method,
            weight="weight",
            proc=proc,
            verbose="ERROR",
        )
        orc.compute_ricci_curvature()
        R = orc.G.copy()
    except Exception as exc:
        st.warning(f"Ricci computation failed for one frame; curvature set to 0. Error: {exc}")
        R = H.copy()

    for u, v in R.edges():
        R[u][v]["ricciCurvature"] = float(R[u][v].get("ricciCurvature", 0.0))
    return R


def compute_components(G: nx.Graph) -> Dict[str, int]:
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    node_cluster: Dict[str, int] = {}
    for cid, comp in enumerate(components):
        for node in comp:
            node_cluster[node] = cid
    return node_cluster


def compute_window_stats(G: nx.Graph, end_date: str) -> WindowStats:
    curv = [float(data.get("ricciCurvature", 0.0)) for _, _, data in G.edges(data=True)]
    comps = list(nx.connected_components(G))
    return WindowStats(
        end_date=end_date,
        avg_ricci=float(np.mean(curv)) if curv else 0.0,
        num_clusters=len(comps),
        largest_component=max((len(c) for c in comps), default=0),
        num_nodes=G.number_of_nodes(),
        num_edges=G.number_of_edges(),
        density=float(nx.density(G)) if G.number_of_nodes() > 1 else 0.0,
    )


def build_frame(
    returns: pd.DataFrame,
    starts: List[int],
    frame_idx: int,
    window_size: int,
    max_distance: float,
    min_abs_corr: float,
    keep_top_edges: Optional[int],
    alpha: float,
    method: str,
    proc: int,
    min_valid_ratio: float,
    min_node_obs: int,
    min_pair_obs: int,
) -> FrameData:
    start = starts[frame_idx]
    window_returns = returns.iloc[start : start + window_size]
    end_date = str(window_returns.index[-1])[:19]
    G0, corr, dist = build_graph_from_window(
        window_returns,
        max_distance=max_distance,
        min_abs_corr=min_abs_corr,
        keep_top_edges=keep_top_edges,
        min_valid_ratio=min_valid_ratio,
        min_node_obs=min_node_obs,
        min_pair_obs=min_pair_obs,
    )
    G = compute_ricci_curvature(G0, alpha=alpha, method=method, proc=proc)
    node_cluster = compute_components(G)
    stats = compute_window_stats(G, end_date=end_date)
    return FrameData(G=G, node_cluster=node_cluster, stats=stats, corr=corr, dist=dist)


def build_base_graph_for_layout(
    returns: pd.DataFrame,
    window_size: int,
    max_distance: float,
    min_abs_corr: float,
    keep_top_edges: Optional[int],
    min_valid_ratio: float,
    max_windows_for_layout: int = 24,
    min_node_obs: int = 1,
    min_pair_obs: int = 4,
) -> nx.Graph:
    base = nx.Graph()
    base.add_nodes_from(returns.columns)
    if len(returns) < window_size:
        return base

    starts = list(range(0, len(returns) - window_size + 1))
    if len(starts) > max_windows_for_layout:
        chosen = np.linspace(0, len(starts) - 1, max_windows_for_layout).astype(int)
        starts = [starts[i] for i in chosen]

    for start in starts:
        G, _, _ = build_graph_from_window(
            returns.iloc[start : start + window_size],
            max_distance=max_distance,
            min_abs_corr=min_abs_corr,
            keep_top_edges=keep_top_edges,
            min_valid_ratio=min_valid_ratio,
            min_node_obs=min_node_obs,
            min_pair_obs=min_pair_obs,
        )
        for u, v, data in G.edges(data=True):
            d = float(data.get("weight", 1.0))
            if base.has_edge(u, v):
                base[u][v]["weight"] = min(float(base[u][v].get("weight", d)), d)
            else:
                base.add_edge(u, v, weight=d)
    return base


def compute_stable_layout(base_graph: nx.Graph, seed: int, scale: float = 1.0) -> Dict[str, Tuple[float, float]]:
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
    rgba = cmap(norm(float(kappa)))
    return mpl.colors.to_hex(rgba)


def edge_universe(frames_data: List[FrameData]) -> List[Tuple[str, str]]:
    """All edges that appear at least once. Fixed list prevents Plotly animation trace mismatch."""
    edges = set()
    for fd in frames_data:
        for u, v in fd.G.edges():
            a, b = sorted((u, v))
            edges.add((a, b))
    return sorted(edges)


def make_plotly_traces(
    frame_data: FrameData,
    positions: Dict[str, Tuple[float, float]],
    all_edges: List[Tuple[str, str]],
    show_edge_weight_labels: bool,
    edge_label_top_n: int,
    node_label_size: int,
    node_size_base: int,
    edge_width_scale: float,
) -> List[go.Scatter]:
    G = frame_data.G
    traces: List[go.Scatter] = []

    # IMPORTANT: exactly one trace for every possible edge, every frame.
    # Missing edges are drawn as None. This keeps node trace index stable during animation.
    for u, v in all_edges:
        if G.has_edge(u, v):
            data = G[u][v]
            x0, y0 = positions.get(u, (0.0, 0.0))
            x1, y1 = positions.get(v, (0.0, 0.0))
            kappa = float(data.get("ricciCurvature", 0.0))
            rho = float(data.get("correlation", np.nan))
            d = float(data.get("distance", data.get("weight", np.nan)))
            width = 0.8 + edge_width_scale * min(1.0, abs(kappa))
            traces.append(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line={"width": width, "color": ricci_to_hex(kappa)},
                    hoverinfo="text",
                    text=f"{u} - {v}<br>weight / distance: {d:.4f}<br>correlation: {rho:.4f}<br>Ricci: {kappa:.4f}",
                    showlegend=False,
                )
            )
        else:
            traces.append(
                go.Scatter(
                    x=[None, None],
                    y=[None, None],
                    mode="lines",
                    line={"width": 0, "color": "rgba(0,0,0,0)"},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # One fixed edge-label trace per frame. Content can vary; trace index does not.
    label_x: List[float] = []
    label_y: List[float] = []
    label_text: List[str] = []
    hover_text: List[str] = []
    if show_edge_weight_labels and G.number_of_edges() > 0:
        selected = sorted(
            G.edges(data=True),
            key=lambda e: float(e[2].get("distance", e[2].get("weight", np.inf))),
        )[: max(0, int(edge_label_top_n))]
        for u, v, data in selected:
            x0, y0 = positions.get(u, (0.0, 0.0))
            x1, y1 = positions.get(v, (0.0, 0.0))
            d = float(data.get("distance", data.get("weight", np.nan)))
            rho = float(data.get("correlation", np.nan))
            kappa = float(data.get("ricciCurvature", 0.0))
            label_x.append((x0 + x1) / 2.0)
            label_y.append((y0 + y1) / 2.0)
            label_text.append(f"w={d:.2f}")
            hover_text.append(f"{u}-{v}<br>weight / distance: {d:.4f}<br>correlation: {rho:.4f}<br>Ricci: {kappa:.4f}")
    traces.append(
        go.Scatter(
            x=label_x,
            y=label_y,
            mode="text",
            text=label_text,
            textfont={"size": 11, "color": "#111111"},
            hoverinfo="text",
            hovertext=hover_text,
            showlegend=False,
        )
    )

    # One fixed node trace. Node points can enter/leave by frame.
    node_x: List[float] = []
    node_y: List[float] = []
    node_text: List[str] = []
    node_hover: List[str] = []
    node_color: List[str] = []
    node_size: List[int] = []
    degrees = dict(G.degree())
    for node in sorted(G.nodes()):
        x, y = positions.get(node, (0.0, 0.0))
        cid = frame_data.node_cluster.get(node, 0)
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_hover.append(f"Ticker: {node}<br>Component: {cid}<br>Degree: {degrees.get(node, 0)}")
        node_color.append(COMPONENT_COLORS[cid % len(COMPONENT_COLORS)])
        node_size.append(int(node_size_base + 2 * degrees.get(node, 0)))

    traces.append(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            textfont={"size": int(node_label_size), "color": "#111111"},
            hoverinfo="text",
            hovertext=node_hover,
            marker={
                "size": node_size,
                "color": node_color,
                "line": {"width": 1.5, "color": "#111111"},
            },
            showlegend=False,
        )
    )

    return traces


def frame_ricci_std(fd: FrameData) -> float:
    vals = [float(data.get("ricciCurvature", 0.0)) for _, _, data in fd.G.edges(data=True)]
    return float(np.std(vals)) if vals else 0.0


def compute_hmm_regimes(
    frames_data: List[FrameData],
    returns: pd.DataFrame,
    starts: List[int],
    window_size: int,
    n_components: int,
    forward_days: int,
    random_state: int,
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """Fit Gaussian HMM on rolling Ricci + market features and attach state info to frames."""
    rows = []
    market_ret_series = returns.mean(axis=1, skipna=True)

    for i, fd in enumerate(frames_data):
        start = starts[i]
        end = min(start + window_size, len(returns))
        win_market = market_ret_series.iloc[start:end].dropna()
        end_pos = end - 1
        next_end = min(end_pos + forward_days, len(market_ret_series) - 1)
        if end_pos + 1 <= next_end:
            next_ret = float(market_ret_series.iloc[end_pos + 1 : next_end + 1].sum())
        else:
            next_ret = np.nan
        rows.append(
            {
                "frame": i,
                "date": fd.stats.end_date,
                "avg_ricci": fd.stats.avg_ricci,
                "ricci_std": frame_ricci_std(fd),
                "clusters": fd.stats.num_clusters,
                "largest_component_ratio": fd.stats.largest_component / max(fd.stats.num_nodes, 1),
                "density": fd.stats.density,
                "num_edges": fd.stats.num_edges,
                "market_return_window_mean": float(win_market.mean()) if len(win_market) else 0.0,
                "market_vol_window": float(win_market.std()) if len(win_market) > 1 else 0.0,
                f"next_{forward_days}d_market_return": next_ret,
            }
        )

    feature_df = pd.DataFrame(rows)
    feature_cols = [
        "avg_ricci",
        "ricci_std",
        "clusters",
        "largest_component_ratio",
        "density",
        "num_edges",
        "market_return_window_mean",
        "market_vol_window",
    ]

    if GaussianHMM is None or StandardScaler is None:
        feature_df["hmm_state"] = -1
        feature_df["regime_name"] = "hmmlearn not installed"
        for fd in frames_data:
            fd.stats.hmm_state = -1
            fd.stats.regime_name = "hmmlearn not installed"
            fd.stats.regime_forward_return = np.nan
        return feature_df, {-1: "hmmlearn not installed"}

    clean = feature_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(clean) < max(8, n_components * 4):
        feature_df["hmm_state"] = -1
        feature_df["regime_name"] = "too few frames"
        for fd in frames_data:
            fd.stats.hmm_state = -1
            fd.stats.regime_name = "too few frames"
            fd.stats.regime_forward_return = np.nan
        return feature_df, {-1: "too few frames"}

    scaler = StandardScaler()
    X = scaler.fit_transform(clean)
    model = GaussianHMM(
        n_components=int(n_components),
        covariance_type="full",
        n_iter=1000,
        random_state=int(random_state),
    )
    model.fit(X)
    states = model.predict(X)
    feature_df["hmm_state"] = states

    fwd_col = f"next_{forward_days}d_market_return"
    summary = feature_df.groupby("hmm_state").agg(
        avg_ricci=("avg_ricci", "mean"),
        market_vol=("market_vol_window", "mean"),
        density=("density", "mean"),
        largest_ratio=("largest_component_ratio", "mean"),
        next_return=(fwd_col, "mean"),
        count=("frame", "count"),
    )

    regime_names: Dict[int, str] = {}
    for state, row in summary.iterrows():
        if row["avg_ricci"] < summary["avg_ricci"].median() and row["market_vol"] >= summary["market_vol"].median():
            name = "stress / fragmentation"
        elif row["avg_ricci"] >= summary["avg_ricci"].median() and row["density"] >= summary["density"].median():
            name = "coherent risk-on"
        else:
            name = "transition / rotation"
        regime_names[int(state)] = name

    feature_df["regime_name"] = feature_df["hmm_state"].map(regime_names)
    state_forward = summary["next_return"].to_dict()
    for i, fd in enumerate(frames_data):
        stt = int(feature_df.loc[i, "hmm_state"])
        fd.stats.hmm_state = stt
        fd.stats.regime_name = regime_names.get(stt, "unknown")
        fd.stats.regime_forward_return = float(state_forward.get(stt, np.nan))
    return feature_df, regime_names


def regime_color(state: int) -> str:
    palette = ["#4E79A7", "#E15759", "#59A14F", "#F28E2B", "#B07AA1"]
    if state < 0:
        return "#888888"
    return palette[state % len(palette)]


def stats_annotation(stats: WindowStats) -> dict:
    return {
        "text": (
            f"<b>Date:</b> {stats.end_date}<br>"
            f"<b>Average Ricci:</b> {stats.avg_ricci:.4f}<br>"
            f"<b>Clusters:</b> {stats.num_clusters}<br>"
            f"<b>Largest component:</b> {stats.largest_component}<br>"
            f"<b>Nodes:</b> {stats.num_nodes} &nbsp; <b>Edges:</b> {stats.num_edges}<br>"
            f"<b>Density:</b> {stats.density:.3f}<br>"
            f"<b>HMM regime:</b> {getattr(stats, 'hmm_state', -1)} - {getattr(stats, 'regime_name', 'not computed')}<br>"
            f"<b>Regime next return:</b> {getattr(stats, 'regime_forward_return', np.nan):.4f}"
        ),
        "xref": "paper",
        "yref": "paper",
        "x": 1.02,
        "y": 0.98,
        "showarrow": False,
        "align": "left",
        "bordercolor": "#999999",
        "borderwidth": 1,
        "bgcolor": "rgba(255,255,255,0.92)",
        "font": {"size": 13, "color": "#111111"},
    }


def build_plotly_figure(
    frames_data: List[FrameData],
    positions: Dict[str, Tuple[float, float]],
    show_edge_weight_labels: bool,
    edge_label_top_n: int,
    node_label_size: int,
    node_size_base: int,
    edge_width_scale: float,
    frame_duration_ms: int,
) -> go.Figure:
    if not frames_data:
        return go.Figure()

    all_edges = edge_universe(frames_data)

    def traces_for(fd: FrameData) -> List[go.Scatter]:
        return make_plotly_traces(
            fd,
            positions=positions,
            all_edges=all_edges,
            show_edge_weight_labels=show_edge_weight_labels,
            edge_label_top_n=edge_label_top_n,
            node_label_size=node_label_size,
            node_size_base=node_size_base,
            edge_width_scale=edge_width_scale,
        )

    fig = go.Figure(data=traces_for(frames_data[0]))
    fig.frames = [
        go.Frame(
            name=str(i),
            data=traces_for(fd),
            layout=go.Layout(annotations=[stats_annotation(fd.stats)]),
        )
        for i, fd in enumerate(frames_data)
    ]

    steps = [
        {
            "method": "animate",
            "label": str(i + 1),
            "args": [
                [str(i)],
                {"mode": "immediate", "frame": {"duration": frame_duration_ms, "redraw": True}, "transition": {"duration": 0}},
            ],
        }
        for i in range(len(frames_data))
    ]

    fig.update_layout(
        title="Rolling Ricci Financial Network v7 - Plotly Animation + HMM Regime",
        width=None,
        height=780,
        showlegend=False,
        hovermode="closest",
        annotations=[stats_annotation(frames_data[0].stats)],
        xaxis={"showgrid": False, "zeroline": False, "visible": False},
        yaxis={"showgrid": False, "zeroline": False, "visible": False, "scaleanchor": "x", "scaleratio": 1},
        margin={"l": 20, "r": 230, "t": 60, "b": 70},
        plot_bgcolor="white",
        paper_bgcolor="white",
        updatemenus=[
            {
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
                        "args": [None, {"frame": {"duration": frame_duration_ms, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Frame ", "font": {"size": 14}},
                "pad": {"t": 40},
                "steps": steps,
            }
        ],
    )
    return fig


def export_plotly_mp4(fig: go.Figure, output_path: str, fps: int = 2) -> None:
    """Export Plotly animation frames to MP4. Requires moviepy and kaleido."""
    from moviepy.editor import ImageSequenceClip

    if not fig.frames:
        raise RuntimeError("No Plotly frames to export.")

    png_paths: List[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, fr in enumerate(fig.frames):
            still = go.Figure(data=fr.data, layout=fig.layout)
            if fr.layout and fr.layout.annotations:
                still.update_layout(annotations=fr.layout.annotations)
            png_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
            still.write_image(png_path, width=1280, height=720, scale=2)
            png_paths.append(png_path)

        clip = ImageSequenceClip(png_paths, fps=fps)
        clip.write_videofile(output_path, codec="libx264", audio=False, logger=None)
        clip.close()


def frame_edge_table(fd: FrameData) -> pd.DataFrame:
    rows = []
    for u, v, data in fd.G.edges(data=True):
        rows.append(
            {
                "u": u,
                "v": v,
                "weight_financial_distance": float(data.get("distance", data.get("weight", np.nan))),
                "correlation": float(data.get("correlation", np.nan)),
                "ricciCurvature": float(data.get("ricciCurvature", 0.0)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["u", "v", "weight_financial_distance", "correlation", "ricciCurvature"])
    return pd.DataFrame(rows).sort_values("weight_financial_distance")


def main() -> None:
    st.set_page_config(page_title="Rolling Ricci Financial Network v7 HMM", layout="wide")
    st.title("Rolling Ricci Financial Network v7 + HMM Regime Detection")
    st.caption("v7: Plotly Ricci animation + IPO/isolated nodes + HMM hidden regime detection from Ricci-network features.")

    with st.sidebar:
        st.header("Data")
        tickers_text = st.text_area("Tickers", value=", ".join(DEFAULT_TICKERS), height=130)
        tickers = parse_tickers(tickers_text)
        period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
        interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m"], index=0)

        st.header("Rolling graph")
        window_size = st.slider("Rolling window size", 10, 120, 60, 5)
        min_valid_ratio = st.slider("Legacy min valid ratio", 0.00, 1.00, 0.00, 0.05, help="Kept for compatibility; v6 uses min observations below for IPO nodes/edges.")
        min_node_obs = st.slider("Min observations to show node", 1, 20, 1, 1)
        min_pair_obs = st.slider("Min overlapping observations for edge/correlation", 3, 60, 4, 1)
        step = st.slider("Frame step", 1, 20, 3, 1)
        max_frames = st.slider("Max animation frames", 5, 120, 40, 5)
        max_distance = st.slider("Max financial distance", 0.05, 2.0, 1.05, 0.05)
        min_abs_corr = st.slider("Minimum |correlation|", 0.00, 1.00, 0.30, 0.05)
        keep_top_edges_raw = st.number_input("Keep top-N shortest edges, 0 = no cap", 0, 1000, 0, 5)
        keep_top_edges = int(keep_top_edges_raw) if int(keep_top_edges_raw) > 0 else None

        st.header("Ricci")
        alpha = st.slider("Ollivier alpha", 0.0, 1.0, 0.5, 0.05)
        method = st.selectbox("Method", ["OTD", "ATD", "Sinkhorn"], index=0)
        proc = st.slider("Processes", 1, 8, 1, 1)

        st.header("HMM regime")
        enable_hmm = st.checkbox("Enable HMM regime detection", value=True)
        hmm_states = st.slider("HMM hidden states", 2, 5, 3, 1)
        hmm_forward_days = st.slider("Forward return days by regime", 1, 20, 5, 1)
        hmm_random_state = st.number_input("HMM random state", min_value=0, max_value=9999, value=42)

        st.header("Layout / labels")
        seed = st.number_input("Stable layout seed", min_value=0, max_value=9999, value=42)
        node_label_size = st.slider("Ticker label font size", 6, 48, 14, 1)
        node_size_base = st.slider("Node base size", 8, 60, 24, 1)
        edge_width_scale = st.slider("Edge width scale", 0.5, 12.0, 5.0, 0.5)
        show_edge_weight_labels = st.checkbox("Show edge weight labels", value=True)
        edge_label_top_n = st.slider("Number of visible edge labels", 0, 100, 30, 5)
        frame_duration_ms = st.slider("Plotly animation ms/frame", 100, 3000, 700, 100)

        st.header("MoviePy export")
        enable_mp4 = st.checkbox("Enable MP4 export button", value=False)
        mp4_fps = st.slider("MP4 FPS", 1, 10, 2, 1)

    if len(tickers) < 2:
        st.error("Please enter at least two tickers.")
        return

    with st.spinner("Downloading prices..."):
        prices = download_prices(tuple(tickers), period, interval)
    if prices.empty or prices.shape[1] < 2:
        st.error("No usable price data. Try fewer tickers or another period/interval.")
        return

    returns = prices_to_returns(prices).dropna(axis=1, how="all")
    if len(returns) < window_size:
        st.error(f"Not enough return rows ({len(returns)}) for window size {window_size}.")
        return
    if returns.shape[1] < 2:
        st.error("Not enough tickers with complete return data.")
        return

    all_starts = list(range(0, len(returns) - window_size + 1, step))
    if len(all_starts) > max_frames:
        chosen = np.linspace(0, len(all_starts) - 1, int(max_frames)).astype(int)
        starts = [all_starts[i] for i in chosen]
    else:
        starts = all_starts

    st.sidebar.success(f"Loaded {returns.shape[1]} tickers, {len(returns)} return rows, {len(starts)} animation frames")

    missing_from_yfinance = [t for t in tickers if t not in returns.columns]
    if missing_from_yfinance:
        st.warning("These requested tickers were not returned by yfinance: " + ", ".join(missing_from_yfinance))

    with st.expander("Ticker data availability diagnostics", expanded=False):
        avail_rows = returns.notna().sum().sort_values()
        st.dataframe(
            pd.DataFrame({"valid_return_rows": avail_rows, "requested": [idx in tickers for idx in avail_rows.index]}),
            width="stretch",
        )

    with st.spinner("Computing stable layout..."):
        base_graph = build_base_graph_for_layout(
            returns,
            window_size=window_size,
            max_distance=max_distance,
            min_abs_corr=min_abs_corr,
            keep_top_edges=keep_top_edges,
            min_valid_ratio=float(min_valid_ratio),
            min_node_obs=int(min_node_obs),
            min_pair_obs=int(min_pair_obs),
        )
        positions = compute_stable_layout(base_graph, seed=int(seed))

    with st.spinner("Computing rolling Ricci frames..."):
        frames_data = [
            build_frame(
                returns=returns,
                starts=starts,
                frame_idx=i,
                window_size=window_size,
                max_distance=max_distance,
                min_abs_corr=min_abs_corr,
                keep_top_edges=keep_top_edges,
                alpha=alpha,
                method=method,
                proc=proc,
                min_valid_ratio=float(min_valid_ratio),
                min_node_obs=int(min_node_obs),
                min_pair_obs=int(min_pair_obs),
            )
            for i in range(len(starts))
        ]

    if enable_hmm:
        with st.spinner("Fitting HMM hidden market regimes..."):
            hmm_feature_df, regime_names = compute_hmm_regimes(
                frames_data=frames_data,
                returns=returns,
                starts=starts,
                window_size=window_size,
                n_components=int(hmm_states),
                forward_days=int(hmm_forward_days),
                random_state=int(hmm_random_state),
            )
    else:
        hmm_feature_df = pd.DataFrame()
        regime_names = {}
        for fd in frames_data:
            fd.stats.hmm_state = -1
            fd.stats.regime_name = "disabled"
            fd.stats.regime_forward_return = np.nan

    fig = build_plotly_figure(
        frames_data=frames_data,
        positions=positions,
        show_edge_weight_labels=show_edge_weight_labels,
        edge_label_top_n=int(edge_label_top_n),
        node_label_size=int(node_label_size),
        node_size_base=int(node_size_base),
        edge_width_scale=float(edge_width_scale),
        frame_duration_ms=int(frame_duration_ms),
    )
    st.plotly_chart(fig, width="stretch")

    inspect_idx = st.slider("Inspect frame table", 0, len(frames_data) - 1, len(frames_data) - 1, 1)
    fd = frames_data[inspect_idx]
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Frame", f"{inspect_idx + 1}/{len(frames_data)}")
    c2.metric("Date", fd.stats.end_date)
    c3.metric("Avg Ricci", f"{fd.stats.avg_ricci:.4f}")
    c4.metric("Clusters", fd.stats.num_clusters)
    c5.metric("Largest comp.", fd.stats.largest_component)
    c6.metric("Edges", fd.stats.num_edges)
    c7.metric("HMM", f"{getattr(fd.stats, 'hmm_state', -1)}", getattr(fd.stats, 'regime_name', ''))

    if enable_hmm and not hmm_feature_df.empty:
        with st.expander("HMM regime diagnostics", expanded=True):
            st.write("Regime names are inferred from average Ricci, volatility, density, and largest-component ratio. They are not supervised labels.")
            st.dataframe(hmm_feature_df, width="stretch")
            fwd_col = f"next_{hmm_forward_days}d_market_return"
            if fwd_col in hmm_feature_df.columns:
                summary = hmm_feature_df.groupby(["hmm_state", "regime_name"]).agg(
                    avg_ricci=("avg_ricci", "mean"),
                    market_vol=("market_vol_window", "mean"),
                    density=("density", "mean"),
                    largest_component_ratio=("largest_component_ratio", "mean"),
                    next_return=(fwd_col, "mean"),
                    count=("frame", "count"),
                ).reset_index()
                st.dataframe(summary, width="stretch")

    with st.expander("Edge table: weight / correlation / Ricci", expanded=True):
        st.dataframe(frame_edge_table(fd), width="stretch")

    with st.expander("Correlation matrix"):
        st.dataframe(fd.corr, width="stretch")

    with st.expander("Financial distance matrix"):
        st.dataframe(fd.dist, width="stretch")

    if enable_mp4:
        st.warning("MP4 export needs `pip install moviepy kaleido`. It may take time.")
        if st.button("Export current Plotly animation to MP4"):
            out_path = os.path.abspath("rolling_ricci_network.mp4")
            try:
                with st.spinner("Exporting MP4 with MoviePy..."):
                    export_plotly_mp4(fig, out_path, fps=int(mp4_fps))
                with open(out_path, "rb") as f:
                    st.download_button("Download MP4", data=f, file_name="rolling_ricci_network.mp4", mime="video/mp4")
            except Exception as exc:
                st.error(f"MP4 export failed: {exc}")
                st.info("Install requirements: pip install moviepy kaleido")


if __name__ == "__main__":
    main()
