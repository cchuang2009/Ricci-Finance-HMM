from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

import networkx as nx
import numpy as np
import plotly.graph_objects as go

PASTEL = [
    "#BFD7EA", "#FFD6A5", "#CDECCF", "#D9C2F0", "#FFCAD4",
    "#C9E4DE", "#F6EAC2", "#D6E2FF", "#E2F0CB", "#F1C0E8",
]


def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:.3f})"


def sector_palette(sectors: Mapping[str, str]) -> Dict[str, str]:
    names = sorted(set(sectors.values()))
    return {name: PASTEL[i % len(PASTEL)] for i, name in enumerate(names)}


def _node_sizes(graph: nx.Graph, minimum: float = 25, maximum: float = 62) -> Dict[str, float]:
    shares = np.array([
        max(float(graph.nodes[n].get("capital_share", 0.0)), 0.0)
        for n in graph.nodes
    ])
    if len(shares) == 0:
        return {}
    if np.ptp(shares) < 1e-12:
        scaled = np.full_like(shares, (minimum + maximum) / 2)
    else:
        scaled = minimum + (maximum - minimum) * (shares - shares.min()) / np.ptp(shares)
    return {str(node): float(size) for node, size in zip(graph.nodes, scaled)}


def network_figure(
    graph: nx.Graph,
    sectors: Mapping[str, str],
    title: str,
    seed: int = 42,
    node_opacity: float = 0.68,
    label_color: str = "#243447",
) -> go.Figure:
    if graph.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    pos = nx.spring_layout(graph, seed=seed, weight="correlation", k=1.15 / math.sqrt(max(graph.number_of_nodes(), 1)))
    palette = sector_palette(sectors)
    sizes = _node_sizes(graph)

    edge_traces = []
    for u, v, data in graph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        curvature = float(data.get("ricciCurvature", 0.0))
        corr = abs(float(data.get("correlation", 0.0)))
        color = "rgba(75,126,152,0.34)" if curvature >= 0 else "rgba(205,96,94,0.34)"
        edge_traces.append(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(width=0.7 + 2.0 * corr, color=color),
            hoverinfo="text",
            text=f"{u}–{v}<br>correlation={corr:.3f}<br>Ricci={curvature:.3f}",
            showlegend=False,
        ))

    node_traces = []
    label_traces = []
    for sector in sorted(set(sectors.get(str(n), "Other") for n in graph.nodes)):
        nodes = [n for n in graph.nodes if sectors.get(str(n), "Other") == sector]
        xs = [pos[n][0] for n in nodes]
        ys = [pos[n][1] for n in nodes]
        hover = []
        for n in nodes:
            attrs = graph.nodes[n]
            hover.append(
                f"<b>{n}</b><br>Sector: {sector}"
                f"<br>Capital share: {float(attrs.get('capital_share', 0.0)):.2%}"
                f"<br>Node Ricci: {float(attrs.get('ricciCurvature', 0.0)):.3f}"
                f"<br>Degree: {graph.degree[n]}"
            )
        color = palette.get(sector, "#D9E2EC")
        node_traces.append(go.Scatter(
            x=xs, y=ys, mode="markers", name=sector,
            marker=dict(
                size=[sizes[str(n)] for n in nodes], color=color,
                opacity=node_opacity,
                line=dict(width=1.3, color=_rgba(color, 0.95)),
            ),
            hovertemplate="%{customdata}<extra></extra>", customdata=hover,
        ))
        label_traces.append(go.Scatter(
            x=xs, y=ys, mode="text", text=[str(n) for n in nodes],
            textposition="middle center",
            textfont=dict(size=12, color=label_color, family="Arial Black"),
            hoverinfo="skip", showlegend=False,
        ))

    fig = go.Figure(data=edge_traces + node_traces + label_traces)
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        template="plotly_white", height=720,
        margin=dict(l=10, r=10, t=65, b=10),
        paper_bgcolor="#FBFCFE", plot_bgcolor="#FBFCFE",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0.0),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        hoverlabel=dict(bgcolor="white", font_size=12, font_color="#243447"),
    )
    return fig


def galaxy_positions(
    graph: nx.Graph,
    sectors: Mapping[str, str],
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Stable 3D sector-radial layout.

    Sector chooses azimuth. Node capital share controls radius. Average node
    curvature controls vertical displacement. Small deterministic jitter avoids
    exact overlap while preserving positions across reruns.
    """
    rng = np.random.default_rng(seed)
    sector_names = sorted(set(sectors.get(str(n), "Other") for n in graph.nodes))
    sector_angle = {
        name: 2 * math.pi * i / max(len(sector_names), 1)
        for i, name in enumerate(sector_names)
    }
    groups: Dict[str, list] = {name: [] for name in sector_names}
    for node in graph.nodes:
        groups[sectors.get(str(node), "Other")].append(node)

    positions: Dict[str, np.ndarray] = {}
    for sector, nodes in groups.items():
        nodes = sorted(nodes, key=str)
        base = sector_angle[sector]
        for idx, node in enumerate(nodes):
            attrs = graph.nodes[node]
            share = max(float(attrs.get("capital_share", 0.0)), 0.0)
            curvature = float(attrs.get("ricciCurvature", 0.0))
            local = (idx - (len(nodes) - 1) / 2) * 0.16
            angle = base + local
            radius = 1.7 + 3.8 * math.sqrt(share + 1e-6) + 0.15 * idx
            jitter = rng.normal(0, 0.025, size=3)
            positions[str(node)] = np.array([
                radius * math.cos(angle),
                radius * math.sin(angle),
                1.65 * curvature,
            ]) + jitter
    return positions


def galaxy_figure(
    graph: nx.Graph,
    sectors: Mapping[str, str],
    title: str,
    seed: int = 42,
) -> go.Figure:
    if graph.number_of_nodes() == 0:
        return go.Figure().update_layout(title=title)

    pos = galaxy_positions(graph, sectors, seed)
    palette = sector_palette(sectors)
    sizes = _node_sizes(graph, 12, 30)

    edge_traces = []
    for u, v, data in graph.edges(data=True):
        p0, p1 = pos[str(u)], pos[str(v)]
        curvature = float(data.get("ricciCurvature", 0.0))
        corr = abs(float(data.get("correlation", 0.0)))
        color = "rgba(62,120,146,0.24)" if curvature >= 0 else "rgba(206,89,86,0.24)"
        edge_traces.append(go.Scatter3d(
            x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
            mode="lines", line=dict(width=1 + 3 * corr, color=color),
            hoverinfo="text",
            text=f"{u}–{v}<br>|corr|={corr:.3f}<br>Ricci={curvature:.3f}",
            showlegend=False,
        ))

    node_traces = []
    label_traces = []
    for sector in sorted(set(sectors.get(str(n), "Other") for n in graph.nodes)):
        nodes = [str(n) for n in graph.nodes if sectors.get(str(n), "Other") == sector]
        coords = np.array([pos[n] for n in nodes])
        color = palette.get(sector, "#D9E2EC")
        hover = []
        for n in nodes:
            attrs = graph.nodes[n]
            hover.append(
                f"<b>{n}</b><br>Sector: {sector}"
                f"<br>Capital share: {float(attrs.get('capital_share', 0.0)):.2%}"
                f"<br>Node Ricci: {float(attrs.get('ricciCurvature', 0.0)):.3f}"
            )
        node_traces.append(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode="markers",
            name=sector,
            marker=dict(size=[sizes[n] for n in nodes], color=color, opacity=0.62,
                        line=dict(width=1, color=_rgba(color, 0.95))),
            customdata=hover, hovertemplate="%{customdata}<extra></extra>",
        ))
        label_traces.append(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode="text",
            text=nodes, textfont=dict(size=11, color="#F7FAFC", family="Arial Black"),
            hoverinfo="skip", showlegend=False,
        ))

    fig = go.Figure(data=edge_traces + node_traces + label_traces)
    fig.update_layout(
        title=dict(text=title, x=0.02), template="plotly_white", height=780,
        margin=dict(l=0, r=0, t=70, b=0), paper_bgcolor="#F8FAFD",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0.0),
        scene=dict(
            bgcolor="#182333",
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            zaxis=dict(title="Node Ricci", gridcolor="rgba(255,255,255,0.12)",
                       zerolinecolor="rgba(255,255,255,0.30)", color="#DCE7F2"),
            camera=dict(eye=dict(x=1.45, y=1.45, z=0.9)),
            aspectmode="cube",
        ),
        hoverlabel=dict(bgcolor="white", font_size=12, font_color="#243447"),
    )
    return fig


def curvature_bar_figure(graph: nx.Graph, title: str) -> go.Figure:
    rows = sorted(
        ((f"{u}–{v}", float(d.get("ricciCurvature", 0.0))) for u, v, d in graph.edges(data=True)),
        key=lambda item: item[1],
    )
    labels = [x[0] for x in rows]
    values = [x[1] for x in rows]
    colors = ["rgba(205,96,94,0.70)" if x < 0 else "rgba(75,126,152,0.70)" for x in values]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors))
    fig.update_layout(template="plotly_white", title=title, height=max(420, 24 * len(rows)),
                      margin=dict(l=20, r=20, t=55, b=20), xaxis_title="Ricci curvature",
                      paper_bgcolor="#FBFCFE", plot_bgcolor="#FBFCFE")
    return fig
