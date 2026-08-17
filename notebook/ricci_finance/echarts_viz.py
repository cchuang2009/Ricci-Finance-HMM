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

import json
import uuid
from IPython.display import HTML, display

def display_echarts(options, width="100%", height="430px"):
    div_id = f"echarts_{uuid.uuid4().hex}"

    html = f"""
    <script src="https://cdn.jsdelivr.net/npm/echarts/dist/echarts.min.js"></script>

    <div id="{div_id}" style="width:{width};height:{height};"></div>

    <script>
    var chart = echarts.init(document.getElementById("{div_id}"));
    chart.setOption({json.dumps(options)});
    </script>
    """

    display(HTML(html))

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


def heatmap_options(
    frame: pd.DataFrame,
    title: str,
    *,
    label_mode: str = "largest",
    max_visible_labels: int = 36,
    x_rotate: int = 45,
    label_digits: int = 4,
) -> dict:
    """Build a readable ECharts heatmap for large sector matrices.

    ``label_mode`` may be ``largest``, ``all``, or ``hover``.  In largest
    mode only the cells with the greatest absolute values receive visible
    labels.  All cells retain a native ECharts tooltip.  No JavaScript
    formatter is used, which avoids Streamlit displaying formatter source.
    """
    rows = list(map(str, frame.index))
    cols = list(map(str, frame.columns))

    raw = []
    for i in range(len(rows)):
        for j in range(len(cols)):
            value = frame.iloc[i, j]
            if pd.notna(value):
                raw.append((j, i, float(value)))

    absmax = max([abs(item[2]) for item in raw], default=1.0)
    nonzero = [item for item in raw if abs(item[2]) > 0.0]
    ranked = sorted(nonzero, key=lambda item: abs(item[2]), reverse=True)
    visible_keys = {
        (j, i) for j, i, _ in ranked[:max(0, int(max_visible_labels))]
    }

    def compact_number(value: float) -> str:
        magnitude = abs(value)
        if magnitude == 0:
            return "0"
        if magnitude < 10 ** (-max(1, int(label_digits))):
            return f"{value:.2e}"
        return f"{value:.{int(label_digits)}f}".rstrip("0").rstrip(".")

    vals = []
    mode = str(label_mode).lower()
    for j, i, value in raw:
        if mode == "all":
            show_label = True
        elif mode == "largest":
            show_label = (j, i) in visible_keys
        else:
            show_label = False

        formatted = compact_number(value)
        vals.append({
            "name": f"{rows[i]} → {cols[j]}",
            "value": [j, i, value, formatted],
            "label": {"show": bool(show_label)},
        })

    return {
        "backgroundColor": "#FBFCFE",
        "title": {"text": title, "left": 10},
        "tooltip": {
            "trigger": "item",
            "formatter": "{b}<br/>Flow: {@[3]}",
        },
        "grid": {
            "left": 190,
            "right": 70,
            "top": 70,
            "bottom": 155,
            "containLabel": False,
        },
        "xAxis": {
            "type": "category",
            "data": cols,
            "splitArea": {"show": True},
            "axisLabel": {
                "interval": 0,
                "rotate": int(x_rotate),
                "fontSize": 11,
                "overflow": "truncate",
                "width": 130,
            },
        },
        "yAxis": {
            "type": "category",
            "data": rows,
            "splitArea": {"show": True},
            "axisLabel": {
                "fontSize": 11,
                "overflow": "truncate",
                "width": 180,
            },
        },
        "dataZoom": [
            {"type": "slider", "xAxisIndex": 0, "bottom": 72, "height": 18},
            {"type": "inside", "xAxisIndex": 0},
            {"type": "slider", "yAxisIndex": 0, "right": 15, "width": 18},
            {"type": "inside", "yAxisIndex": 0},
        ],
        "visualMap": {
            "min": -absmax,
            "max": absmax,
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": 15,
        },
        "series": [{
            "type": "heatmap",
            "data": vals,
            "label": {
                "show": False,
                "fontSize": 9,
                "formatter": "{@[3]}",
            },
            "emphasis": {
                "itemStyle": {
                    "shadowBlur": 8,
                    "shadowColor": "rgba(0,0,0,.25)",
                }
            },
        }],
    }


def capital_flow_animation_options(
    aligned_frames,
    close,
    sector_map,
    interval_ms=800,
):
    options = []

    for frame in aligned_frames:
        graph = round_graph_inplace(frame["graph"].copy())
        date = pd.Timestamp(frame["date"])

        momentum = sector_momentum(
            close,
            sector_map,
            date,
        ).round(ROUND_DIGITS)

        flow = sector_flow_matrix(
            graph,
            sector_map,
            momentum,
        ).round(ROUND_DIGITS)

        # Continue building animation...