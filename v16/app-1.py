from __future__ import annotations

ROUND_DIGITS = 4
RANDOM_STATE = 42

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from streamlit_echarts import st_echarts
import streamlit as st

from ricci_finance.data import prepare_market_data
from ricci_finance.gnn import train_graph_regime
from ricci_finance.hmm import (
    DEFAULT_HMM_FEATURES,
    build_regime_labels,
    current_run_length,
    fit_gaussian_hmm,
    forecast_hmm_methods,
    switch_rate,
)
from ricci_finance.pipeline import build_rolling_frames
from ricci_finance.sectors import (
    KNOWN,
    assign_sector_memberships,
    primary_sector_map,
    parse_sector_map,
    sector_flow_matrix,
    sector_momentum,
)
from ricci_finance.sector_objects import fetch_yfinance_profiles, memberships_map
from ricci_finance.surgery import graph_surgery
from ricci_finance.visualization import galaxy_figure
from ricci_finance.dynamic import (
    build_dynamic_sector_history,
    compare_cluster_labels,
    dynamic_primary_history,
    ricci_communities,
    sector_evolution_table,
)
from ricci_finance.advanced_visualization import (
    attention_heatmap,
    community_animation,
    embedding_figure,
    galaxy_animation,
    reduce_embeddings,
)
from ricci_finance.echarts_viz import (
    curvature_bar_options, heatmap_options, line_options, loss_options,
    momentum_bar_options, network_options,
)

from ricci_finance.helper import (round_numeric,round_dataframe,round_graph_inplace, 
                                 round_nested,round_plotly_figure,
                                 capital_flow_animation_options
)

st.set_page_config(
    page_title="Ricci Finance V16 Starter",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
    .v16-card {
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


def default_sector_text(tickers: list[str]) -> str:
    """Return one editable TICKER=Sector line for every requested ticker."""
    return "\n".join(f"{ticker}={KNOWN.get(ticker, '')}" for ticker in tickers)


def compact_flow_matrix(
    frame: pd.DataFrame,
    max_sectors: int,
    other_label: str = "Other sectors",
) -> pd.DataFrame:
    """Keep the largest sectors and aggregate the remainder without losing flow."""
    matrix = frame.fillna(0.0).astype(float).copy()
    labels = list(dict.fromkeys([*map(str, matrix.index), *map(str, matrix.columns)]))
    matrix = matrix.reindex(index=labels, columns=labels, fill_value=0.0)

    if len(labels) <= max_sectors:
        return matrix

    importance = matrix.abs().sum(axis=1) + matrix.abs().sum(axis=0)
    keep_count = max(1, int(max_sectors) - 1)
    keep = list(importance.nlargest(keep_count).index)
    remainder = [label for label in labels if label not in keep]

    compact = matrix.loc[keep, keep].copy()
    compact[other_label] = matrix.loc[keep, remainder].sum(axis=1)
    compact.loc[other_label] = 0.0
    compact.loc[other_label, keep] = matrix.loc[remainder, keep].sum(axis=0)
    compact.loc[other_label, other_label] = float(
        matrix.loc[remainder, remainder].to_numpy().sum()
    )
    return compact


st.title("Ricci Finance V16 Starter")
st.caption(
    "V15 foundation · automatic or manual normalized multi-sector profiles · "
    "weighted GNN features · backward-compatible primary-sector visualizations"
)

with st.sidebar:
    st.subheader("Market universe")
    ticker_text = st.text_area(
        "Tickers (comma or line separated)",
        "NVDA, AMD, AVGO, MRVL, ANET, BNT, CBRS, LITE, LRCX, KLAC, AMAT, MU, INTC, AAPL, QNT, QBTS, SPCX, META, IONQ",
        height=100,
        help="Add a new ticker here, then add the same ticker and its required sector below.",
    )
    sidebar_tickers = list(dict.fromkeys(
        item.strip().upper()
        for item in ticker_text.replace("\n", ",").split(",")
        if item.strip()
    ))
    sector_mode = st.radio(
        "Sector source",
        ["Automatic", "Hybrid", "Manual"],
        horizontal=True,
        help=(
            "Automatic uses Yahoo sector, industry, and business-summary themes. "
            "Hybrid uses manual rows as complete per-ticker overrides."
        ),
    )
    sector_text = st.text_area(
        "Sector overrides (optional except in Manual mode)",
        default_sector_text(sidebar_tickers),
        height=260,
        help=(
            "Single: NVDA=Semiconductors. Weighted: "
            "NVDA=Semiconductors:6|AI:3|DataCenter:1. "
            "Weights are normalized automatically for each ticker."
        ),
        disabled=sector_mode == "Automatic",
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
    gnn_model = st.radio("Graph model", ["GCN", "GAT"], horizontal=True)
    embedding_method = st.selectbox("Embedding reduction", ["PCA", "UMAP", "t-SNE"])
    embedding_dimensions = st.radio("Embedding view", [2, 3], horizontal=True)

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
    if sector_mode == "Automatic":
        sector_overrides = {}
    elif sector_mode == "Manual":
        sector_overrides = parse_sector_map(sector_text, required_tickers=tickers)
    else:
        sector_overrides = parse_sector_map(sector_text, required_tickers=())
except ValueError as sector_error:
    st.error(str(sector_error))
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
    all_nodes = {str(node) for frame in aligned_frames for node in frame["graph"].nodes}
    if sector_mode in {"Automatic", "Hybrid"}:
        profiles = fetch_yfinance_profiles(
            sorted(all_nodes),
            overrides=sector_overrides if sector_mode == "Hybrid" else None,
        )
        sector_memberships = memberships_map(profiles)
    else:
        sector_memberships = assign_sector_memberships(
            all_nodes,
            sector_overrides,
            require_all=True,
        )
    sectors = primary_sector_map(sector_memberships)

    # V16 dynamic sector weights and communities are computed for every aligned frame.
    dynamic_sector_history = build_dynamic_sector_history(
        aligned_frames,
        sector_memberships,
    )
    dynamic_sector_primary = dynamic_primary_history(dynamic_sector_history)
    community_history = [
        ricci_communities(frame["graph"])
        for frame in aligned_frames
    ]
    evolution_table = sector_evolution_table(
        aligned_frames,
        dynamic_sector_history,
    )

    gnn_result = None
    if run_gnn and len(labels) >= 10:
        with st.spinner(f"Training pure-PyTorch {gnn_model}"):
            gnn_result = train_graph_regime(
                [frame["graph"] for frame in aligned_frames],
                labels,
                dynamic_sector_history,
                model_type=gnn_model,
                epochs=epochs,
                hidden=hidden,
                random_state=RANDOM_STATE,
            )

    metric_columns = st.columns(6)
    metric_columns[0].metric("Current regime", aligned_features.iloc[-1].regime_name)
    metric_columns[1].metric("State", int(labels[-1]))
    metric_columns[2].metric("Confidence", f"{aligned_features.iloc[-1].probability:.1%}")
    metric_columns[3].metric("Frames", len(labels))
    metric_columns[4].metric("Nodes / edges", f"{latest_graph.number_of_nodes()} / {latest_graph.number_of_edges()}")
    metric_columns[5].metric("Curvature", latest["curvature_engine"])

    with st.expander("V16 normalized multi-sector profiles", expanded=False):
        profile_rows = []
        for ticker in sorted(sector_memberships):
            values = sector_memberships[ticker]
            profile_rows.append({
                "ticker": ticker,
                "primary_sector": sectors[ticker],
                "weight_sum": sum(values.values()),
                "memberships": " | ".join(
                    f"{name}:{weight:.3f}"
                    for name, weight in sorted(values.items(), key=lambda item: -item[1])
                ),
            })
        st.dataframe(round_dataframe(pd.DataFrame(profile_rows)), width="stretch")

    tabs = st.tabs([
        "Overview",
        "Network",
        "3D Galaxy",
        "Sector evolution",
        "Communities",
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
        st.caption(
            "Output: each line tracks a rolling graph statistic through time. "
            "avg_ricci measures average local robustness; negative_edge_ratio measures fragile/bridge-like links; "
            "edge_stability measures persistence of connections; capital_concentration measures whether activity is dominated by few nodes."
        )
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
            "Output: pastel node color = sector; node size = capital share; edge width = |correlation|; "
            "blue edge = non-negative curvature and red edge = negative curvature. "
            "Conclusion: large nodes connected by thick red edges deserve attention because they combine capital importance, strong co-movement, and structural fragility."
        )

    with tabs[2]:
        height_source = st.radio(
            "Galaxy height",
            ["Ricci curvature", "GNN embedding"],
            horizontal=True,
            disabled=gnn_result is None,
        )
        galaxy_embeddings = None
        if height_source == "GNN embedding" and gnn_result is not None:
            galaxy_embeddings = gnn_result.graph_embeddings
        dynamic_galaxy = galaxy_animation(
            aligned_frames,
            dynamic_sector_history,
            embeddings=galaxy_embeddings,
            interval_ms=capital_flow_speed,
        )
        st.plotly_chart(dynamic_galaxy, width="stretch", config={"displaylogo": False})
        st.caption(
            "Angle is the weighted sector/theme direction, radius is capital share, and height is node Ricci curvature "
            "or the selected GNN latent coordinate. Press Play to follow the rolling-window evolution."
        )

    with tabs[3]:
        st.subheader("Dynamic sector weights")
        selected_ticker = st.selectbox("Ticker", sorted(all_nodes), key="evolution_ticker")
        ticker_evolution = evolution_table[evolution_table.ticker == selected_ticker]
        pivot = ticker_evolution.pivot_table(index="date", columns="sector", values="weight", fill_value=0.0)
        st.line_chart(pivot, height=520)
        st.dataframe(round_dataframe(ticker_evolution), width="stretch", hide_index=True, height=300)
        st.caption(
            "Each line is a rolling membership weight. Static Yahoo/theme information remains the anchor, while "
            "neighbor-sector affinity, capital share and local Ricci structure produce gradual changes over time."
        )

    with tabs[4]:
        st.subheader("Temporal Ricci-community animation")
        community_fig = community_animation(
            aligned_frames,
            community_history,
            interval_ms=capital_flow_speed,
        )
        st.plotly_chart(community_fig, width="stretch", config={"displaylogo": False})

        latest_gnn_clusters = {}
        if gnn_result is not None and gnn_result.node_embeddings is not None:
            latest_node_embeddings = gnn_result.node_embeddings[-1]
            cluster_count = max(2, min(states, len(gnn_result.nodes)))
            try:
                from sklearn.cluster import KMeans
                cluster_values = KMeans(n_clusters=cluster_count, random_state=RANDOM_STATE, n_init=10).fit_predict(latest_node_embeddings)
                latest_gnn_clusters = dict(zip(gnn_result.nodes, map(int, cluster_values)))
            except Exception:
                latest_gnn_clusters = {}

        comparison_table = compare_cluster_labels(
            profiles if sector_mode in {"Automatic", "Hybrid"} else None,
            dynamic_sector_history[-1],
            community_history[-1],
            latest_gnn_clusters,
        )
        st.subheader("Automatic label comparison")
        st.dataframe(comparison_table, width="stretch", hide_index=True)
        st.caption(
            "Yahoo sector is fundamental metadata; detected theme is the largest dynamic V16 membership; "
            "Ricci community comes from the current market graph; GNN latent cluster comes from the latest learned node embedding."
        )

    with tabs[5]:
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
        st.caption(
            "Output: correlation is signed co-movement; distance is sqrt(2(1-correlation)); weight is the graph transport/cost weight; "
            "Ricci curvature describes local redundancy versus bridge fragility; Wasserstein distance appears for Ollivier-LP. "
            "Conclusion: sort ascending to inspect the most negative and potentially fragile edges first."
        )

    with tabs[6]:
        st.caption(
            "Sector definitions come directly from the validated sidebar mapping. "
            "Adding a ticker therefore requires adding its sector before analysis."
        )
        momentum = sector_momentum(close, sector_memberships, latest["date"]).round(ROUND_DIGITS)
        flow = sector_flow_matrix(latest_graph, sector_memberships, momentum).round(ROUND_DIGITS)

        st.subheader("Sector momentum")
        st_echarts(
            options=round_nested(momentum_bar_options(momentum)),
            height="520px",
            key="sector_momentum",
        )
        st.caption(
            "Sector momentum is displayed separately at full width. Each ticker contributes "
            "to every sector according to its normalized V16 membership weights."
        )

        st.divider()
        st.subheader("Sector capital-flow matrix")

        matrix_control_1, matrix_control_2, matrix_control_3 = st.columns([1.1, 1.0, 1.0])
        with matrix_control_1:
            max_matrix_sectors = st.slider(
                "Maximum displayed sectors",
                min_value=6,
                max_value=max(6, min(30, len(flow.index))),
                value=min(12, max(6, len(flow.index))),
                help=(
                    "The most important sectors are retained. Remaining rows and columns "
                    "are aggregated into Other sectors, so total flow is preserved."
                ),
                key="max_matrix_sectors",
            )
        with matrix_control_2:
            show_matrix_values = st.checkbox(
                "Show values inside cells",
                value=False,
                help="Leave this off for large matrices and read exact values from the tooltip.",
                key="show_matrix_values",
            )
        with matrix_control_3:
            matrix_label_rotation = st.slider(
                "Column-label rotation",
                min_value=0,
                max_value=90,
                value=45,
                step=15,
                key="matrix_label_rotation",
            )

        display_flow = compact_flow_matrix(flow, max_matrix_sectors).round(ROUND_DIGITS)
        matrix_height = max(620, 42 * len(display_flow.index) + 250)

        st_echarts(
            options=round_nested(
                heatmap_options(
                    display_flow,
                    "Sector capital-flow matrix",
                    show_labels=show_matrix_values,
                    x_rotate=matrix_label_rotation,
                    label_digits=ROUND_DIGITS,
                )
            ),
            height=f"{matrix_height}px",
            key="sector_flow",
        )
        st.caption(
            "Cell labels are hidden by default to prevent overlap. Hover for exact values, "
            "use the horizontal/vertical zoom sliders, or reduce the displayed sectors. "
            "Aggregated sectors are included under ‘Other sectors’, preserving total flow."
        )

        st.divider()
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

    with tabs[7]:
      st.subheader("Graph Surgery")

      # -------------------------------------------------------
      # Interactive surgery threshold
      # -------------------------------------------------------
      removal_fraction = st.slider(
        "Fraction of negative-curvature edges to remove",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.05,
        key="surgery_fraction",
      )

      # Collect negative Ricci curvatures
      negative_edges = [
        float(data.get("ricciCurvature", 0.0))
        for _, _, data in latest_graph.edges(data=True)
        if float(data.get("ricciCurvature", 0.0)) < 0
      ]

      if len(negative_edges) == 0:
        st.warning("This graph contains no negative-curvature edges.")
        surgery_threshold = float("-inf")
      elif removal_fraction == 0:
        surgery_threshold = float("-inf")
      else:
        surgery_threshold = float(
            np.quantile(negative_edges, removal_fraction)
        )

      # -------------------------------------------------------
      # Perform surgery
      # -------------------------------------------------------
      operated, info = graph_surgery(
        latest_graph,
        curvature_threshold=surgery_threshold,
      )

      operated = round_graph_inplace(operated)

      info.update(
        {
            "threshold": surgery_threshold,
            "edges_before": latest_graph.number_of_edges(),
            "edges_after": operated.number_of_edges(),
            "components_before": nx.number_connected_components(latest_graph),
            "components_after": nx.number_connected_components(operated),
            "removed_edges": (
                latest_graph.number_of_edges()
                - operated.number_of_edges()
            ),
        }
      )

      info = round_nested(info)

      if info["removed_edges"] == 0:
        st.warning(
            "No edges were removed. "
            "Increase the removal fraction."
        )

      # -------------------------------------------------------
      # Curvature statistics
      # -------------------------------------------------------
      if latest_graph.number_of_edges():

        curvatures = np.array(
            [
                float(
                    data.get("ricciCurvature", 0.0)
                )
                for _, _, data in latest_graph.edges(data=True)
            ]
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Minimum",
            f"{curvatures.min():.3f}",
        )

        c2.metric(
            "Maximum",
            f"{curvatures.max():.3f}",
        )

        c3.metric(
            "Negative edges",
            int(np.sum(curvatures < 0)),
        )

        c4.metric(
            "Removed",
            info["removed_edges"],
        )

      # -------------------------------------------------------
      # Network comparison
      # -------------------------------------------------------
      before, after = st.columns(2)

      with before:

        st_echarts(
            options=round_nested(
                network_options(
                    latest_graph,
                    sectors,
                    "Before graph surgery",
                    node_opacity=node_opacity,
                    label_color=label_color,
                )
            ),
            height="620px",
            key="surgery_before",
        )

      with after:

        st_echarts(
            options=round_nested(
                network_options(
                    operated,
                    sectors,
                    "After graph surgery",
                    node_opacity=node_opacity,
                    label_color=label_color,
                )
            ),
            height="620px",
            key="surgery_after",
        )

      st.json(info)

      st.caption(
        "Edges with the most negative Ricci curvature are "
        "removed first. "
        "This is a stress-test of the market network rather "
        "than a claim that market relationships literally "
        "disappear."
        )  

    with tabs[8]:
        if not run_gnn:
            st.info("Enable **Run GNN** in the sidebar.")
        elif len(labels) < 10:
            st.warning("At least 10 aligned frames are required for GNN.")
        else:
            result = gnn_result
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
            st.caption(
                result.note + " Output: accuracy compares graph-model predictions with HMM-derived labels; balanced accuracy reduces class-imbalance bias; "
                "the loss curve should generally decline. Accuracy above the majority baseline suggests graph structure contains regime information."
            )

            st.subheader("GNN graph embedding visualization")
            reduced = reduce_embeddings(
                result.graph_embeddings,
                dimensions=embedding_dimensions,
                method=embedding_method,
                random_state=RANDOM_STATE,
            )
            embedding_plot = embedding_figure(
                reduced,
                result.predictions,
                aligned_features.date,
                dimensions=embedding_dimensions,
                title=f"{result.model_type} latent graph embeddings — {embedding_method}",
            )
            st.plotly_chart(embedding_plot, width="stretch", config={"displaylogo": False})

            if result.model_type == "GAT" and result.attention is not None:
                st.subheader("Latest GAT attention matrix")
                attention_plot = attention_heatmap(
                    result.attention[-1],
                    result.nodes,
                    title=f"GAT attention — {pd.Timestamp(latest['date']).date()}",
                )
                st.plotly_chart(attention_plot, width="stretch", config={"displaylogo": False})
                st.caption(
                    "Rows are querying tickers and columns are attended tickers. High values identify relationships "
                    "the GAT used most strongly for the latest graph representation."
                )

    with tabs[9]:
        st.caption(
            "Output: one row per aligned rolling frame, containing the graph statistics used by HMM and downstream models. "
            "Conclusion: this table is the audit trail for verifying every chart and exporting reproducible results."
        )
        st.dataframe(aligned_features_display, width="stretch")
        st.download_button(
            "Download features CSV",
            aligned_features_display.to_csv(index=False),
            "v15_final_features.csv",
            "text/csv",
        )

except Exception as exc:
    st.exception(exc)
