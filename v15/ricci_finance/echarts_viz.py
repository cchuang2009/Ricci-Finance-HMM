from __future__ import annotations

import math
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd

PASTEL = [
    "#BFD7EA", "#FFD6A5", "#CDECCF", "#D9C2F0", "#FFCAD4",
    "#C9E4DE", "#F6EAC2", "#D6E2FF", "#E2F0CB", "#F1C0E8",
]


def sector_palette(sectors: Mapping[str, str]) -> dict[str, str]:
    names = sorted(set(sectors.values()))
    return {name: PASTEL[i % len(PASTEL)] for i, name in enumerate(names)}


def _node_sizes(graph: nx.Graph, minimum: float = 24, maximum: float = 60) -> dict[str, float]:
    nodes = list(graph.nodes)
    shares = np.asarray([max(float(graph.nodes[n].get("capital_share", 0.0)), 0.0) for n in nodes])
    if not len(nodes):
        return {}
    if np.ptp(shares) < 1e-12:
        values = np.full(len(nodes), (minimum + maximum) / 2)
    else:
        values = minimum + (maximum - minimum) * (shares - shares.min()) / np.ptp(shares)
    return {str(n): float(s) for n, s in zip(nodes, values)}


def network_options(graph: nx.Graph, sectors: Mapping[str, str], title: str,
                    seed: int = 42, node_opacity: float = 0.68,
                    label_color: str = "#243447") -> dict:
    if graph.number_of_nodes() == 0:
        return {"title": {"text": title}, "series": []}
    pos = nx.spring_layout(graph, seed=seed, weight="correlation",
                           k=1.15 / math.sqrt(max(graph.number_of_nodes(), 1)))
    palette = sector_palette(sectors)
    sizes = _node_sizes(graph)
    categories = sorted(set(sectors.get(str(n), "Other") for n in graph.nodes))
    category_index = {name: i for i, name in enumerate(categories)}
    nodes = []
    for n in graph.nodes:
        name = str(n)
        attrs = graph.nodes[n]
        sector = sectors.get(name, "Other")
        x, y = pos[n]
        nodes.append({
            "id": name, "name": name, "x": float(x * 620), "y": float(y * 620),
            "symbolSize": sizes[name], "category": category_index[sector],
            "value": [float(attrs.get("capital_share", 0.0)), float(attrs.get("ricciCurvature", 0.0)), int(graph.degree[n])],
            "itemStyle": {"color": palette.get(sector, "#D9E2EC"), "opacity": node_opacity,
                          "borderColor": "rgba(75,90,110,0.45)", "borderWidth": 1.2},
            "label": {"show": True, "color": label_color, "fontSize": 12, "fontWeight": "bold"},
        })
    links = []
    for u, v, d in graph.edges(data=True):
        curvature = float(d.get("ricciCurvature", 0.0))
        corr = abs(float(d.get("correlation", 0.0)))
        links.append({
            "source": str(u), "target": str(v),
            "value": [corr, curvature],
            "lineStyle": {"width": 0.7 + 2.2 * corr, "opacity": 0.34,
                          "color": "#4B7E98" if curvature >= 0 else "#CD605E",
                          "curveness": 0.06},
        })
    return {
        "backgroundColor": "#FBFCFE",
        "title": {"text": title, "left": 10, "top": 8, "textStyle": {"fontSize": 17, "color": "#243447"}},
        "tooltip": {"trigger": "item"},
        "legend": [{"type": "scroll", "top": 42, "data": categories}],
        "animationDurationUpdate": 450,
        "series": [{
            "type": "graph", "layout": "none", "data": nodes, "links": links,
            "categories": [{"name": name, "itemStyle": {"color": palette[name]}} for name in categories],
            "roam": True, "draggable": True, "focusNodeAdjacency": True,
            "label": {"show": True, "position": "inside"},
            "emphasis": {"focus": "adjacency", "lineStyle": {"width": 4, "opacity": 0.85}},
        }],
    }


def line_options(frame: pd.DataFrame, columns: Sequence[str], title: str) -> dict:
    dates = [pd.Timestamp(x).strftime("%Y-%m-%d") for x in frame["date"]]
    return {
        "backgroundColor": "#FBFCFE",
        "title": {"text": title, "left": 10},
        "tooltip": {"trigger": "axis"},
        "legend": {"type": "scroll", "top": 30},
        "grid": {"left": 55, "right": 25, "top": 75, "bottom": 55},
        "xAxis": {"type": "category", "data": dates, "axisLabel": {"hideOverlap": True}},
        "yAxis": {"type": "value", "scale": True, "splitLine": {"lineStyle": {"color": "#E8EEF5"}}},
        "dataZoom": [{"type": "inside"}, {"type": "slider", "height": 18}],
        "series": [{"name": c, "type": "line", "showSymbol": False, "smooth": False,
                    "lineStyle": {"width": 2}, "data": [None if pd.isna(v) else float(v) for v in frame[c]]}
                   for c in columns],
    }


def loss_options(losses: Sequence[float]) -> dict:
    return {
        "backgroundColor": "#FBFCFE", "title": {"text": "GCN training loss", "left": 10},
        "tooltip": {"trigger": "axis"}, "grid": {"left": 60, "right": 25, "top": 55, "bottom": 45},
        "xAxis": {"type": "category", "name": "Epoch", "data": list(range(1, len(losses)+1))},
        "yAxis": {"type": "value", "name": "Weighted CE", "scale": True},
        "series": [{"type": "line", "name": "Loss", "showSymbol": False, "smooth": True,
                    "lineStyle": {"width": 2.3}, "data": [float(x) for x in losses]}],
    }


def curvature_bar_options(graph: nx.Graph, title: str) -> dict:
    rows = sorted(((f"{u}–{v}", float(d.get("ricciCurvature", 0.0))) for u,v,d in graph.edges(data=True)), key=lambda x:x[1])
    return {
        "backgroundColor": "#FBFCFE", "title": {"text": title, "left": 10},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": 90, "right": 25, "top": 55, "bottom": 30, "containLabel": True},
        "xAxis": {"type": "value", "name": "Ricci curvature"},
        "yAxis": {"type": "category", "data": [r[0] for r in rows], "axisLabel": {"fontSize": 10}},
        "series": [{"type": "bar", "data": [{"value": v, "itemStyle": {"color": "#CD605E" if v < 0 else "#4B7E98", "opacity": .72}} for _,v in rows]}],
    }


def momentum_bar_options(momentum: pd.Series) -> dict:
    s = momentum.sort_values()
    return {
        "backgroundColor": "#FBFCFE", "title": {"text": "Sector momentum", "left": 10},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": 35, "right": 20, "top": 55, "bottom": 25, "containLabel": True},
        "xAxis": {"type": "value", "name": "5-day return"},
        "yAxis": {"type": "category", "data": [str(x) for x in s.index]},
        "series": [{"type": "bar", "data": [float(x) for x in s.values]}],
    }


def heatmap_options(frame: pd.DataFrame, title: str) -> dict:
    rows, cols = list(map(str, frame.index)), list(map(str, frame.columns))
    vals = [[j, i, float(frame.iloc[i,j])] for i in range(len(rows)) for j in range(len(cols)) if pd.notna(frame.iloc[i,j])]
    absmax = max([abs(x[2]) for x in vals], default=1.0)
    return {
        "backgroundColor": "#FBFCFE", "title": {"text": title, "left": 10},
        "tooltip": {"position": "top"},
        "grid": {"left": 70, "right": 25, "top": 65, "bottom": 65},
        "xAxis": {"type": "category", "data": cols, "splitArea": {"show": True}},
        "yAxis": {"type": "category", "data": rows, "splitArea": {"show": True}},
        "visualMap": {"min": -absmax, "max": absmax, "calculable": True, "orient": "horizontal", "left": "center", "bottom": 5},
        "series": [{"type": "heatmap", "data": vals, "label": {"show": True, "formatter": "{@[2]}"},
                    "emphasis": {"itemStyle": {"shadowBlur": 8, "shadowColor": "rgba(0,0,0,.25)"}}}],
    }
