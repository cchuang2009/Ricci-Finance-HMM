from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from streamlit_echarts import st_echarts
import streamlit as st

from ricci_finance.data import prepare_market_data
from ricci_finance.gnn import train_gcn_regime
from ricci_finance.hmm import (
    DEFAULT_HMM_FEATURES,
    build_regime_labels,
    current_run_length,
    fit_gaussian_hmm,
    forecast_hmm_methods,
    switch_rate,
)
from ricci_finance.pipeline import build_rolling_frames
from ricci_finance.sectors import assign_sectors, sector_flow_matrix, sector_momentum
from ricci_finance.surgery import graph_surgery
from ricci_finance.visualization import galaxy_figure
from ricci_finance.echarts_viz import (
    curvature_bar_options, heatmap_options, line_options, loss_options,
    momentum_bar_options, network_options,
)

st.set_page_config(
    page_title="Ricci Finance V15 Final",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)
RANDOM_STATE = 42

ROUND_DIGITS = 3


def round_numeric(value, digits: int = ROUND_DIGITS):
    """Round scalar float-like values while preserving non-numeric objects."""
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    return value


def round_dataframe(df: pd.DataFrame, digits: int = ROUND_DIGITS) -> pd.DataFrame:
    """Return a copy with all numeric columns rounded for display/export."""
    result = df.copy()
    numeric_columns = result.select_dtypes(include=[np.number]).columns
    result[numeric_columns] = result[numeric_columns].round(digits)
    return result


def round_graph_inplace(graph: nx.Graph, digits: int = ROUND_DIGITS) -> nx.Graph:
    """Round every floating-point node and edge attribute in a graph."""
    for _, attributes in graph.nodes(data=True):
        for key, value in list(attributes.items()):
            attributes[key] = round_numeric(value, digits)
    for _, _, attributes in graph.edges(data=True):
        for key, value in list(attributes.items()):
            attributes[key] = round_numeric(value, digits)
    return graph


def round_nested(value, digits: int = ROUND_DIGITS):
    """Recursively round values used in ECharts options and JSON output."""
    if isinstance(value, dict):
        return {key: round_nested(item, digits) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(round_nested(item, digits) for item in value)
    if isinstance(value, list):
        return [round_nested(item, digits) for item in value]
    if isinstance(value, np.ndarray):
        return [round_nested(item, digits) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    return round_numeric(value, digits)


def round_plotly_figure(figure, digits: int = ROUND_DIGITS):
    """Rebuild a Plotly figure with all numeric JSON values rounded."""
    return go.Figure(round_nested(figure.to_dict(), digits))


def capital_flow_animation_options(
    aligned_frames: list[dict],
    close: pd.DataFrame,
    sectors: dict[str, str],
    interval_ms: int = 900,
) -> dict:
    """Build an animated sector-flow matrix with gradual change colors.

    Each cell label is the current dimensionless flow score. Its background
    color represents the cell's change from the preceding frame: increasingly
    green for an increase, increasingly red for a decrease, and near-neutral
    for little or no change. The score is model-derived and is not USD.
    """
    snapshots: list[tuple[str, pd.DataFrame]] = []
    all_sector_names: set[str] = set()

    for frame in aligned_frames:
        graph = round_graph_inplace(frame["graph"].copy())
        date = pd.Timestamp(frame["date"])
        momentum = sector_momentum(close, sectors, date).round(ROUND_DIGITS)
        flow = sector_flow_matrix(graph, sectors, momentum).round(ROUND_DIGITS)
        if flow.empty:
            continue
        flow.index = flow.index.astype(str)
        flow.columns = flow.columns.astype(str)
        all_sector_names.update(flow.index)
        all_sector_names.update(flow.columns)
        snapshots.append((str(date.date()), flow))

    if not snapshots:
        return {
            "title": {"text": "No capital-flow animation data"},
            "series": [],
        }

    sector_names = sorted(all_sector_names)
    normalized: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    change_values: list[float] = []
    previous_matrix: pd.DataFrame | None = None

    for date_label, flow in snapshots:
        matrix = flow.reindex(
            index=sector_names, columns=sector_names, fill_value=0.0
        ).fillna(0.0).round(ROUND_DIGITS)

        change = (
            pd.DataFrame(0.0, index=sector_names, columns=sector_names)
            if previous_matrix is None
            else matrix - previous_matrix
        ).round(ROUND_DIGITS)

        change_values.extend(change.to_numpy(dtype=float).ravel().tolist())
        normalized.append((date_label, matrix, change))
        previous_matrix = matrix.copy()

    change_abs_max = max((abs(float(v)) for v in change_values), default=1.0)
    change_abs_max = max(round(change_abs_max, ROUND_DIGITS), 0.001)

    options = []
    for date_label, matrix, change in normalized:
        # [destination index, source index, current score, change from prior frame]
        heatmap_data = [
            [
                column_index,
                row_index,
                round(float(matrix.iloc[row_index, column_index]), ROUND_DIGITS),
                round(float(change.iloc[row_index, column_index]), ROUND_DIGITS),
            ]
            for row_index in range(len(sector_names))
            for column_index in range(len(sector_names))
        ]

        options.append({
            "title": {
                "text": f"Sector capital-flow change — {date_label}",
                "subtext": (
                    "Cell number = current dimensionless flow score; "
                    "cell color = change from previous frame"
                ),
                "left": "center",
                "textStyle": {"fontSize": 24, "fontWeight": "bold"},
                "subtextStyle": {"fontSize": 16},
            },
            "series": [{
                "name": "Capital-flow score and change",
                "type": "heatmap",
                "data": heatmap_data,
                "label": {
                    "show": True,
                    "formatter": "{@[2]}",
                    "fontSize": 16,
                    "fontWeight": "bold",
                },
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowColor": "rgba(0,0,0,0.35)",
                    }
                },
            }],
        })

    return round_nested({
        "baseOption": {
            "timeline": {
                "axisType": "category",
                "autoPlay": True,
                "playInterval": int(interval_ms),
                "loop": True,
                "bottom": 0,
                "data": [date_label for date_label, *_ in normalized],
                "label": {"formatter": "{value}", "fontSize": 14},
                "controlStyle": {"itemSize": 22},
            },
            "tooltip": {
                "trigger": "item",
                "textStyle": {"fontSize": 15},
                "formatter": (
                    "function (p) {"
                    "var current = Number(p.value[2]).toFixed(3);"
                    "var delta = Number(p.value[3]).toFixed(3);"
                    "var sign = Number(p.value[3]) > 0 ? '+' : '';"
                    "return '<b>' + p.name + '</b><br/>' +"
                    "'Current score: ' + current + '<br/>' +"
                    "'Change: ' + sign + delta;"
                    "}"
                ),
            },
            "grid": {"top": 110, "left": 150, "right": 125, "bottom": 105},
            "xAxis": {
                "type": "category",
                "data": sector_names,
                "name": "Destination sector",
                "nameTextStyle": {"fontSize": 18, "fontWeight": "bold"},
                "splitArea": {"show": True},
                "axisLabel": {"rotate": 25, "fontSize": 17, "fontWeight": "bold"},
            },
            "yAxis": {
                "type": "category",
                "data": sector_names,
                "name": "Source sector",
                "nameTextStyle": {"fontSize": 18, "fontWeight": "bold"},
                "splitArea": {"show": True},
                "axisLabel": {"fontSize": 17, "fontWeight": "bold"},
            },
            "visualMap": {
                "type": "continuous",
                "dimension": 3,
                "min": -change_abs_max,
                "max": change_abs_max,
                "calculable": True,
                "orient": "vertical",
                "right": 5,
                "top": 145,
                "precision": ROUND_DIGITS,
                "text": ["Increase", "Decrease"],
                "textStyle": {"fontSize": 15, "fontWeight": "bold"},
                "inRange": {
                    "color": [
                        "#8b1d1d",
                        "#d96b6b",
                        "#f4d4d4",
                        "#f2f2f2",
                        "#d3ecd9",
                        "#65b87a",
                        "#176b35",
                    ]
                },
            },
            "series": [{"type": "heatmap", "data": []}],
            "animationDurationUpdate": min(int(interval_ms * 0.75), 700),
            "animationEasingUpdate": "cubicInOut",
        },
        "options": options,
    })

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.35rem; padding-bottom: 2.5rem;}
    [data-testid="stMetric"] {
        background: rgba(250,252,255,0.92);
        border: 1px solid rgba(120,145,170,0.22);
        padding: 0.75rem 0.9rem;
        border-radius: 0.85rem;
        box-shadow: 0 3px 14px rgba(44,62,80,0.05);
    }
    [data-testid="stSidebar"] {background: #F5F8FC;}
    .v15-card {
        border: 1px solid rgba(120,145,170,0.20);
        border-radius: 0.9rem;
        padding: 0.8rem 1rem;
        background: #FBFCFE;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load(tickers: tuple[str, ...], period: str):
    return prepare_market_data(tickers, period=period)


st.title("Ricci Finance V15 Final")
st.caption(
    "No TensorFlow · No POT · SciPy-LP Ollivier / Forman curvature · "
    "Gaussian HMM · pure-PyTorch GCN · improved Plotly network and 3D Galaxy"
)

with st.sidebar:
    st.subheader("Market universe")
    ticker_text = st.text_area(
        "Tickers",
        "NVDA, AMD, AVGO, ANET, MRVL, MU, LRCX, KLAC, AMAT, ASML",
        height=100,
    )
    period = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=3)

    st.subheader("Rolling graph")
    window = st.slider("Window", 30, 126, 63)
    step = st.slider("Step", 1, 21, 5)
    max_frames = st.slider("Maximum frames", 15, 120, 50)
    k = st.slider("Neighbors k", 2, 8, 4)
    min_corr = st.slider("Minimum |correlation|", 0.0, 0.9, 0.2, 0.05)

    st.subheader("Curvature")
    engine_label = st.radio(
        "Engine",
        ["Forman — fast", "Ollivier LP — SciPy transport"],
        index=0,
    )
    curvature_engine = "forman" if engine_label.startswith("Forman") else "ollivier_lp"
    alpha = st.slider(
        "Ollivier idleness α", 0.0, 1.0, 0.5, 0.05,
        disabled=curvature_engine != "ollivier_lp",
    )

    st.subheader("Regime and GNN")
    states = st.slider("HMM states", 2, 5, 2)
    run_gnn = st.checkbox("Run GNN", True)
    epochs = st.slider("GNN epochs", 20, 500, 150, 10)
    hidden = st.select_slider("Hidden width", [8, 16, 24, 32, 64], value=24)

    st.subheader("Visualization")
    node_opacity = st.slider("Node opacity", 0.25, 1.0, 0.68, 0.05)
    label_theme = st.radio("2D label color", ["Dark", "Light"], horizontal=True)
    label_color = "#243447" if label_theme == "Dark" else "#F7FAFC"
    capital_flow_speed = st.slider(
        "Capital-flow animation speed (ms)", 300, 2500, 900, 100
    )

    run_analysis = st.button("Run analysis", type="primary", width="stretch")

if not run_analysis:
    st.info("Choose parameters and click **Run analysis**.")
    st.stop()

tickers = list(dict.fromkeys(
    item.strip().upper()
    for item in ticker_text.replace("\n", ",").split(",")
    if item.strip()
))
if len(tickers) < 3:
    st.error("Enter at least three tickers.")
    st.stop()

try:
    market = load(tuple(tickers), period)
    close = market["close"]
    returns = market["returns"]
    dollar_volume = market["dollar_volume"]

    progress_bar = st.progress(0, text="Building rolling graphs")

    def progress(index, count, date):
        progress_bar.progress(
            index / max(count, 1),
            text=f"Frame {index}/{count}: {pd.Timestamp(date).date()}",
        )

    frames, features = build_rolling_frames(
        returns,
        dollar_volume,
        window=window,
        step=step,
        max_frames=max_frames,
        k=k,
        min_corr=min_corr,
        alpha=alpha,
        curvature_engine=curvature_engine,
        progress=progress,
    )
    progress_bar.empty()

    hmm = fit_gaussian_hmm(
        features,
        list(DEFAULT_HMM_FEATURES),
        states,
        RANDOM_STATE,
    )

    aligned_frames = [frames[index] for index in hmm.valid_index]
    aligned_features = features.iloc[hmm.valid_index].reset_index(drop=True).copy()
    labels = hmm.states
    names = build_regime_labels(aligned_features, labels)
    aligned_features["hmm_state"] = labels
    aligned_features["regime_name"] = [names[int(x)] for x in labels]
    aligned_features["probability"] = hmm.posterior.max(axis=1)
    aligned_features_display = round_dataframe(aligned_features)

    latest = aligned_frames[-1]
    latest_graph: nx.Graph = round_graph_inplace(latest["graph"].copy())
    sectors = assign_sectors({
        node for frame in aligned_frames for node in frame["graph"].nodes
    })

    metric_columns = st.columns(6)
    metric_columns[0].metric("Current regime", aligned_features.iloc[-1].regime_name)
    metric_columns[1].metric("State", int(labels[-1]))
    metric_columns[2].metric("Confidence", f"{aligned_features.iloc[-1].probability:.1%}")
    metric_columns[3].metric("Frames", len(labels))
    metric_columns[4].metric("Nodes / edges", f"{latest_graph.number_of_nodes()} / {latest_graph.number_of_edges()}")
    metric_columns[5].metric("Curvature", latest["curvature_engine"])

    tabs = st.tabs([
        "Overview",
        "Network",
        "3D Galaxy",
        "Edge curvature",
        "Sector flow",
        "Surgery",
        "GNN",
        "Data",
    ])

    with tabs[0]:
        plotted = [
            name for name in [
                "avg_ricci", "negative_edge_ratio", "edge_stability",
                "capital_concentration",
            ] if name in aligned_features
        ]
        st_echarts(options=round_nested(line_options(aligned_features_display, plotted, "Regime feature history")), height="470px", key="regime_features")
        left, right = st.columns([1.2, 1])
        with left:
            forecast = forecast_hmm_methods(
                hmm.model,
                hmm.posterior[-1],
                int(labels[-1]),
                5,
                current_run_length=current_run_length(labels),
            )
            forecast["regime"] = forecast.state.map(names)
            forecast_display = round_dataframe(forecast)
            st.dataframe(forecast_display, width="stretch", hide_index=True)
        with right:
            st.markdown(
                f"""
                <div class="v15-card">
                <b>Regime diagnostics</b><br><br>
                Switch rate: <b>{switch_rate(labels):.3f}</b><br>
                Current run length: <b>{current_run_length(labels)}</b> frames<br>
                Latest graph date: <b>{pd.Timestamp(latest['date']).date()}</b><br>
                Curvature engine: <b>{latest['curvature_engine']}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tabs[1]:
        network_opts = network_options(
            latest_graph, sectors,
            f"Latest market network — {pd.Timestamp(latest['date']).date()}",
            node_opacity=node_opacity, label_color=label_color,
        )
        st_echarts(options=round_nested(network_opts), height="720px", key="latest_network")
        st.caption(
            "Pastel color = sector · node size = capital share · blue edge = non-negative curvature · "
            "red edge = negative curvature · edge width = |correlation|."
        )

    with tabs[2]:
        galaxy = galaxy_figure(
            latest_graph,
            sectors,
            f"3D Ricci Galaxy — {pd.Timestamp(latest['date']).date()}",
        )
        galaxy = round_plotly_figure(galaxy)
        st.plotly_chart(galaxy, width="stretch", config={"displaylogo": False})
        st.caption(
            "Sector determines orbital direction, capital share influences radius, and node Ricci curvature controls height. "
            "Drag to rotate; scroll to zoom."
        )

    with tabs[3]:
        edge_rows = []
        for u, v, data in latest_graph.edges(data=True):
            edge_rows.append({
                "source": u,
                "target": v,
                "correlation": data.get("correlation", np.nan),
                "distance": data.get("distance", np.nan),
                "weight": data.get("weight", np.nan),
                "ricciCurvature": data.get("ricciCurvature", np.nan),
                "wassersteinDistance": data.get("wassersteinDistance", np.nan),
                "engine": data.get("ricci_engine", curvature_engine),
            })
        edge_table = round_dataframe(pd.DataFrame(edge_rows).sort_values("ricciCurvature"))
        left, right = st.columns([1.25, 1])
        with left:
            st_echarts(
                options=round_nested(curvature_bar_options(latest_graph, "Latest edge curvature")), height="520px", key="curvature_bar"
            )
        with right:
            st.dataframe(edge_table, width="stretch", hide_index=True, height=520)

    with tabs[4]:
        momentum = sector_momentum(close, sectors, latest["date"]).round(ROUND_DIGITS)
        flow = sector_flow_matrix(latest_graph, sectors, momentum).round(ROUND_DIGITS)
        left, right = st.columns([0.8, 1.4])
        with left:
            st_echarts(options=round_nested(momentum_bar_options(momentum)), height="430px", key="sector_momentum")
        with right:
            st_echarts(options=round_nested(heatmap_options(flow, "Sector capital-flow matrix")), height="440px", key="sector_flow")

        st.subheader("Capital-flow animation")
        flow_animation = capital_flow_animation_options(
            aligned_frames,
            close,
            sectors,
            interval_ms=capital_flow_speed,
        )
        st_echarts(
            options=flow_animation,
            height="760px",
            key="capital_flow_animation",
        )
        st.caption(
            "The matrix is a dimensionless model-derived flow score, not a dollar total. "
            "Rows are source sectors and columns are destination sectors. Each number is the current "
            "flow score; its gradually changing color shows the cell-level increase or decrease from "
            "the previous frame. Stronger green means a larger increase, stronger red a larger decrease."
        )

    with tabs[5]:
        operated, info = graph_surgery(latest_graph)
        operated = round_graph_inplace(operated)
        info = round_nested(info)
        before, after = st.columns(2)
        with before:
            st_echarts(options=round_nested(network_options(latest_graph, sectors, "Before graph surgery", node_opacity=node_opacity)), height="620px", key="surgery_before")
        with after:
            st_echarts(options=round_nested(network_options(operated, sectors, "After graph surgery", node_opacity=node_opacity)), height="620px", key="surgery_after")
        st.json(info)

    with tabs[6]:
        if not run_gnn:
            st.info("Enable **Run GNN** in the sidebar.")
        elif len(labels) < 10:
            st.warning("At least 10 aligned frames are required for GNN.")
        else:
            with st.spinner("Training pure-PyTorch GCN"):
                result = train_gcn_regime(
                    [frame["graph"] for frame in aligned_frames],
                    labels,
                    sectors,
                    epochs=epochs,
                    hidden=hidden,
                    random_state=RANDOM_STATE,
                )
            left, middle, right, baseline = st.columns(4)
            left.metric("Test accuracy", f"{result.accuracy:.3f}")
            middle.metric("Balanced accuracy", f"{result.balanced_accuracy:.3f}")
            right.metric("Device", result.device)
            majority = float(np.bincount(result.labels[result.test_indices]).max() / len(result.test_indices))
            baseline.metric("Majority baseline", f"{majority:.3f}")

            indices = result.test_indices
            comparison = pd.DataFrame({
                "date": aligned_features.date.iloc[indices].to_numpy(),
                "HMM": result.labels[indices],
                "GCN": result.predictions[indices],
                "correct": result.labels[indices] == result.predictions[indices],
            })
            comparison = round_dataframe(comparison)
            left, right = st.columns([1.1, 1])
            with left:
                st.dataframe(comparison, width="stretch", hide_index=True, height=360)
            with right:
                st_echarts(options=round_nested(loss_options(np.round(result.losses, ROUND_DIGITS))), height="330px", key="gnn_loss")
            st.caption(result.note)

    with tabs[7]:
        st.dataframe(aligned_features_display, width="stretch")
        st.download_button(
            "Download features CSV",
            aligned_features_display.to_csv(index=False),
            "v15_final_features.csv",
            "text/csv",
        )

except Exception as exc:
    st.exception(exc)
