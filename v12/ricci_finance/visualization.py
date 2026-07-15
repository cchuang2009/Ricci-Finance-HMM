from __future__ import annotations
from typing import Dict, Iterable, Sequence
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import matplotlib as mpl
from .models import FrameData
from .graph import compute_components
from .rolling import compute_window_stats
from .story import FrameStory

COMPONENT_COLORS = ["#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F","#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC"]
RICCI_COLOR_MIN, RICCI_COLOR_MAX = -0.6, 0.6
RICCI_PLOTLY_COLORSCALE = "RdBu"


def ricci_to_hex(kappa: float) -> str:
    norm = mpl.colors.Normalize(vmin=RICCI_COLOR_MIN, vmax=RICCI_COLOR_MAX)
    return mpl.colors.to_hex(mpl.colormaps["coolwarm_r"](norm(float(kappa))))


def ricci_colorbar_trace_3d():
    return go.Scatter3d(
        x=[None, None], y=[None, None], z=[None, None], mode="markers",
        marker={"size": 0.1, "opacity": 0.0,
                "color": [RICCI_COLOR_MIN, RICCI_COLOR_MAX],
                "colorscale": RICCI_PLOTLY_COLORSCALE,
                "cmin": RICCI_COLOR_MIN, "cmax": RICCI_COLOR_MAX,
                "showscale": True,
                "colorbar": {"title": "Ricci κ<br>red− / blue+", "thickness": 16, "len": 0.62}},
        hoverinfo="skip", showlegend=False,
    )


def compute_stable_layout_3d(
    base_graph: nx.Graph, seed: int = 42, scale: float = 1.0,
    layout_k: float = 0.45, iterations: int = 600,
) -> Dict[str, tuple[float, float, float]]:
    if base_graph.number_of_nodes() == 0:
        return {}
    if base_graph.number_of_edges() == 0:
        pos = nx.circular_layout(base_graph, scale=scale)
        return {n: (float(x), float(y), 0.0) for n, (x, y) in pos.items()}
    pos = nx.spring_layout(base_graph, dim=3, seed=seed, weight="weight",
                           k=layout_k, iterations=iterations, scale=scale)
    return {n: tuple(map(float, xyz)) for n, xyz in pos.items()}


def _node_z(G: nx.Graph, node: str, z_mode: str) -> float:
    if z_mode == "capital_mass":
        return float(np.log1p(G.nodes[node].get("capital_mass", 0.0)))
    vals = [float(G[node][nbr].get("ricciCurvature", 0.0)) for nbr in G.neighbors(node)]
    return float(-np.mean(vals)) if vals else 0.0


def _ranges(values, pad=0.15):
    a = np.asarray(list(values), dtype=float)
    if not len(a) or not np.isfinite(a).any():
        return [-1.2, 1.2]
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    span = max(hi - lo, 1e-6)
    return [lo - pad*span, hi + pad*span]


def _frame_traces(
    fd: FrameData,
    positions: Dict[str, tuple[float, float,float]],
    all_edges: list[tuple[str,str]],
    z_mode: str,
):
    G = fd.G
    traces = []
    flows = [float(d.get("edge_capital_flow", 0.0)) for *_, d in G.edges(data=True)]
    flow_scale = np.nanpercentile(flows, 90) if flows and max(flows) > 0 else 1.0
    for u, v in all_edges:
        if G.has_edge(u, v):
            d = G[u][v]
            x0,y0,_ = positions.get(u, (0,0,0)); x1,y1,_ = positions.get(v,(0,0,0))
            z0,z1 = _node_z(G,u,z_mode), _node_z(G,v,z_mode)
            k = float(d.get("ricciCurvature",0))
            flow = float(d.get("edge_capital_flow",0))
            width = 1.0 + 6.0*np.sqrt(max(flow,0)/max(flow_scale,1e-12))
            traces.append(go.Scatter3d(
                x=[x0,x1], y=[y0,y1], z=[z0,z1], mode="lines",
                line={"width": width, "color": ricci_to_hex(k)},
                hovertext=f"{u}-{v}<br>corr={d.get('correlation',np.nan):.3f}<br>distance={d.get('distance',np.nan):.3f}<br>Ricci={k:.3f}<br>capital flow={flow:,.0f}",
                hoverinfo="text", showlegend=False,
            ))
        else:
            traces.append(go.Scatter3d(x=[None,None],y=[None,None],z=[None,None],mode="lines",
                                       line={"width":0,"color":"rgba(0,0,0,0)"},hoverinfo="skip",showlegend=False))
    masses = np.asarray([float(G.nodes[n].get("capital_mass",0)) for n in G.nodes()])
    mass_scale = np.nanpercentile(masses[masses>0],90) if np.any(masses>0) else 1.0
    xs=[];ys=[];zs=[];labels=[];colors=[];sizes=[];hover=[]
    for n in sorted(G.nodes()):
        x,y,_ = positions.get(n,(0,0,0)); z=_node_z(G,n,z_mode)
        cid=fd.node_cluster.get(n,0); status=G.nodes[n].get("status","active")
        mass=float(G.nodes[n].get("capital_mass",0))
        xs.append(x);ys.append(y);zs.append(z);labels.append(n)
        colors.append("#D0D0D0" if status!="active" else COMPONENT_COLORS[cid%len(COMPONENT_COLORS)])
        sizes.append(8+22*np.sqrt(max(mass,0)/max(mass_scale,1e-12)))
        hover.append(f"Ticker: {n}<br>cluster={cid}<br>status={status}<br>capital mass={mass:,.0f}<br>z={z:.3f}")
    traces.append(go.Scatter3d(
        x=xs,y=ys,z=zs,mode="markers+text",text=labels,textposition="top center",
        marker={"size":sizes,"color":colors,"line":{"width":1,"color":"#222"}},
        hovertext=hover,hoverinfo="text",showlegend=False,
    ))
    return traces


def _story_annotation(story: FrameStory | None) -> dict:
    if story is None:
        return {
            "text": "HMM story not computed.",
            "xref": "paper",
            "yref": "paper",
            "x": 1.02,
            "y": 0.98,
            "xanchor": "left",
            "yanchor": "top",
            "align": "left",
            "showarrow": False,
            "bordercolor": "#888",
            "borderwidth": 1,
            "bgcolor": "rgba(255,255,255,0.90)",
            "font": {"size": 12},
        }

    probability = (
        f"{story.state_probability:.0%}"
        if np.isfinite(story.state_probability)
        else "n/a"
    )
    changes = "<br>".join(
        f"• {item}" for item in story.changes[:4]
    )

    return {
        "text": (
            f"<b>{story.headline}</b><br>"
            f"Regime: {story.regime_name}<br>"
            f"Posterior confidence: {probability}<br><br>"
            f"{changes}<br><br>"
            f"<b>Fragile edge</b>: {story.strongest_fragile_edge}<br>"
            f"<b>Capital edge</b>: {story.strongest_capital_edge}"
        ),
        "xref": "paper",
        "yref": "paper",
        "x": 1.01,
        "y": 0.98,
        "xanchor": "left",
        "yanchor": "top",
        "align": "left",
        "showarrow": False,
        "bordercolor": "#777",
        "borderwidth": 1,
        "borderpad": 8,
        "bgcolor": "rgba(255,255,255,0.92)",
        "font": {"size": 12},
    }


def _story_title(
    base_title: str,
    frame_index: int,
    frame: FrameData,
    story: FrameStory | None,
) -> str:
    if story is None:
        return (
            f"{base_title}<br><sup>"
            f"Frame {frame_index + 1} · {frame.stats.end_date}"
            f"</sup>"
        )

    probability = (
        f"{story.state_probability:.0%}"
        if np.isfinite(story.state_probability)
        else "n/a"
    )

    return (
        f"{base_title}<br><sup>"
        f"Frame {frame_index + 1} · {story.date} · "
        f"{story.regime_name} · confidence {probability}"
        f"</sup>"
    )


def build_3d_ricci_capital_animation(
    frames: Sequence[FrameData],
    positions_3d: Dict[str, tuple[float, float, float]],
    z_mode: str = "ricci_stress",
    frame_duration_ms: int = 700,
    edge_label_top_n: int = 25,
    title: str = "Ricci Finance v12: dynamic 3D Ricci-capital manifold",
    stories: Sequence[FrameStory] | None = None,
):
    if not frames:
        return go.Figure()

    story_list = list(stories) if stories is not None else []
    all_edges = sorted(
        {
            tuple(sorted((u, v)))
            for fd in frames
            for u, v in fd.G.edges()
        }
    )

    initial_story = story_list[0] if story_list else None
    initial = _frame_traces(
        frames[0],
        positions_3d,
        all_edges,
        z_mode,
    )

    fig = go.Figure(
        data=initial + [ricci_colorbar_trace_3d()]
    )
    trace_ids = list(range(len(initial)))

    plotly_frames = []

    for index, frame in enumerate(frames):
        story = (
            story_list[index]
            if index < len(story_list)
            else None
        )

        plotly_frames.append(
            go.Frame(
                name=str(index),
                data=_frame_traces(
                    frame,
                    positions_3d,
                    all_edges,
                    z_mode,
                ),
                traces=trace_ids,
                layout=go.Layout(
                    title={
                        "text": _story_title(
                            title,
                            index,
                            frame,
                            story,
                        )
                    },
                    annotations=[
                        _story_annotation(story)
                    ],
                ),
            )
        )

    fig.frames = plotly_frames

    xs = [position[0] for position in positions_3d.values()]
    ys = [position[1] for position in positions_3d.values()]
    zs = [
        _node_z(frame.G, node, z_mode)
        for frame in frames
        for node in frame.G.nodes()
    ]

    steps = [
        {
            "method": "animate",
            "label": str(index + 1),
            "args": [
                [str(index)],
                {
                    "mode": "immediate",
                    "frame": {
                        "duration": frame_duration_ms,
                        "redraw": True,
                    },
                    "transition": {"duration": 0},
                },
            ],
        }
        for index in range(len(frames))
    ]

    fig.update_layout(
        title={
            "text": _story_title(
                title,
                0,
                frames[0],
                initial_story,
            )
        },
        height=850,
        showlegend=False,
        annotations=[_story_annotation(initial_story)],
        scene={
            "domain": {"x": [0.0, 0.77], "y": [0.0, 1.0]},
            "xaxis": {
                "visible": False,
                "range": _ranges(xs),
            },
            "yaxis": {
                "visible": False,
                "range": _ranges(ys),
            },
            "zaxis": {
                "visible": False,
                "range": _ranges(zs),
            },
            "camera": {
                "eye": {"x": 1.8, "y": 1.8, "z": 1.15}
            },
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 330, "t": 85, "b": 70},
        updatemenus=[
            {
                "type": "buttons",
                "x": 0.02,
                "y": -0.04,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {
                                    "duration": frame_duration_ms,
                                    "redraw": True,
                                },
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Frame "},
                "pad": {"t": 40},
                "steps": steps,
            }
        ],
    )

    return fig
def visualize_network_3d(
    G: nx.Graph,
    positions_3d: Dict[str, tuple[float,float,float]] | None = None,
    title: str = "3D Ricci-capital manifold",
    node_cluster: Dict[str,int] | None = None,
    z_mode: str = "ricci_stress",
):
    if positions_3d is None:
        positions_3d = compute_stable_layout_3d(G)
    fd = FrameData(G=G,node_cluster=node_cluster or compute_components(G),
                   stats=compute_window_stats(G,"selected"),
                   corr=None,dist=None)
    edges=sorted(tuple(sorted(e)) for e in G.edges())
    traces=_frame_traces(fd,positions_3d,edges,z_mode)
    xs=[p[0] for p in positions_3d.values()];ys=[p[1] for p in positions_3d.values()]
    zs=[_node_z(G,n,z_mode) for n in G.nodes()]
    fig=go.Figure(data=traces+[ricci_colorbar_trace_3d()])
    fig.update_layout(title=title,height=800,showlegend=False,
        scene={"xaxis":{"visible":False,"range":_ranges(xs)},
               "yaxis":{"visible":False,"range":_ranges(ys)},
               "zaxis":{"visible":False,"range":_ranges(zs)},
               "camera":{"eye":{"x":1.8,"y":1.8,"z":1.15}},
               "aspectmode":"cube"},
        margin={"l":0,"r":150,"t":50,"b":20})
    return fig
