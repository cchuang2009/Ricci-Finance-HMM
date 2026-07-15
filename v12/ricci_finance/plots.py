from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_rolling_features(feature_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if feature_df is None or feature_df.empty:
        fig.update_layout(title="Rolling features: no data")
        return fig
    fig.add_trace(go.Scatter(x=feature_df["date"], y=feature_df["avg_ricci"], mode="lines+markers", name="Average Ricci"))
    fig.add_trace(go.Scatter(x=feature_df["date"], y=feature_df["density"], mode="lines+markers", name="Density"))
    fig.add_trace(go.Scatter(x=feature_df["date"], y=feature_df["clusters"], mode="lines+markers", name="Clusters", yaxis="y2"))
    fig.update_layout(
        title="Rolling Ricci-network observables", height=520,
        xaxis_title="Window end date", yaxis_title="Curvature / density",
        yaxis2={"title": "Cluster count", "overlaying": "y", "side": "right"},
        hovermode="x unified",
    )
    return fig


def plot_hmm_regimes(hmm_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if hmm_df is None or hmm_df.empty:
        fig.update_layout(title="HMM regimes: no data")
        return fig
    fig.add_trace(go.Scatter(x=hmm_df["date"], y=hmm_df["hmm_state"], mode="lines+markers", name="HMM state"))
    if "avg_ricci" in hmm_df:
        fig.add_trace(go.Scatter(x=hmm_df["date"], y=hmm_df["avg_ricci"], mode="lines", name="Average Ricci", yaxis="y2"))
    if "total_edge_capital_flow" in hmm_df:
        fig.add_trace(go.Scatter(x=hmm_df["date"], y=np.log1p(hmm_df["total_edge_capital_flow"].fillna(0)), mode="lines",
                                 name="log(1 + edge capital flow)", yaxis="y3"))
    fig.update_layout(
        title="HMM hidden regimes from Ricci + capital-flow features", height=520,
        yaxis={"title": "Hidden state"},
        yaxis2={"title": "Avg Ricci", "overlaying": "y", "side": "right"},
        yaxis3={"title": "log flow", "overlaying": "y", "side": "right", "anchor": "free", "position": 0.94},
        hovermode="x unified",
    )
    return fig


def plot_capital_flow_bars(flow_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    fig = go.Figure()
    if flow_df is None or flow_df.empty:
        fig.update_layout(title="Capital-flow edges: no data")
        return fig
    d = flow_df.head(int(top_n)).copy()
    d["edge"] = d["u"].astype(str) + "-" + d["v"].astype(str)
    fig.add_trace(go.Bar(
        x=d["edge"], y=d["edge_capital_flow"],
        text=[f"{100*x:.1f}%" for x in d["edge_flow_share"]],
        textposition="outside",
        hovertext=[
            f"corr={r:.3f}<br>Ricci={k:.3f}<br>effective d={ed:.3f}"
            for r, k, ed in zip(d["correlation"], d["ricciCurvature"], d["effective_distance"])
        ],
        hoverinfo="x+y+text",
    ))
    fig.update_layout(
        title="Top capital-flow transport edges", height=430,
        xaxis_title="Edge", yaxis_title="Dollar-volume weighted flow",
        margin={"l": 40, "r": 20, "t": 60, "b": 100},
    )
    return fig


def plot_ricci_flow_history(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if history is None or history.empty:
        fig.update_layout(title="Ricci flow history: no data")
        return fig
    for col in ["avg_ricci", "ricci_min", "ricci_max"]:
        if col in history:
            fig.add_trace(go.Scatter(x=history["iteration"], y=history[col], mode="lines+markers", name=col))
    fig.update_layout(title="Ricci-flow projection history", xaxis_title="Iteration", yaxis_title="Curvature")
    return fig
