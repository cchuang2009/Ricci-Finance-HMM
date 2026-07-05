"""
app_v10.py - Ricci Finance v10

v10 = v8 sparse rolling topology + optional v9 Ricci-flow / surgery layer.

Run:
    streamlit run app_v10.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from helper import (
    DEFAULT_TICKERS,
    build_base_graph_for_layout,
    build_plotly_animation,
    build_rolling_frames,
    compare_before_after_flow,
    compute_components,
    compute_hmm_regimes,
    compute_stable_layout,
    download_prices,
    make_demo_prices,
    parse_tickers,
    perform_financial_surgery,
    plot_ricci_flow_history,
    plot_rolling_features,
    plot_sector_flow,
    prices_to_returns,
    rolling_feature_table,
    run_ricci_flow,
    sector_flow_table,
    summarize_edges,
    visualize_network,
)

st.set_page_config(page_title="Ricci Finance v10", layout="wide")
st.title("Ricci Finance v10: v8 Market Topology + Optional v9 Flow/Surgery")
st.caption(
    "Default view is the v8-style sparse rolling topology for market clusters, IPO emergence, and sector rotation. "
    "The v9 Ricci-flow/surgery layer is optional and runs only on the selected frame."
)

with st.sidebar:
    st.header("Data")
    data_mode = st.radio("Source", ["Synthetic lecture data", "yfinance live download"], index=0)
    tickers_text = st.text_area("Tickers", value=", ".join(DEFAULT_TICKERS), height=130)
    tickers = parse_tickers(tickers_text)
    if data_mode == "yfinance live download":
        period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
        interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m"], index=0)
    else:
        n_days = st.slider("Synthetic trading days", 120, 900, 320, 20)
        synthetic_seed = st.number_input("Synthetic seed", min_value=0, max_value=9999, value=7)

    st.header("v8 topology layer")
    window_size = st.slider("Rolling window size", 10, 180, 60, 5)
    step = st.slider("Frame step", 1, 30, 5, 1)
    max_frames = st.slider("Max animation frames", 5, 140, 45, 5)
    edge_mode = st.selectbox("Edge mode", ["threshold", "knn", "hybrid"], index=0, help="v8-style market topology works best with threshold mode.")
    positive_only = st.checkbox("Positive correlation only", value=True)
    max_distance = st.slider("Max financial distance", 0.05, 2.00, 1.05, 0.05)
    min_abs_corr = st.slider("Minimum correlation threshold", 0.00, 1.00, 0.30, 0.05)
    keep_top_edges_raw = st.number_input("Keep top-N shortest threshold edges; 0 = no cap", 0, 3000, 0, 5)
    keep_top_edges = int(keep_top_edges_raw) if int(keep_top_edges_raw) > 0 else None
    knn_k = st.slider("kNN neighbors, only for knn/hybrid", 1, 10, 3, 1)
    min_node_obs = st.slider("Min observations to show node", 1, 30, 1, 1)
    min_pair_obs = st.slider("Min overlapping observations for edge", 3, 100, 4, 1)

    st.header("Ricci curvature")
    alpha = st.slider("Ollivier alpha", 0.0, 1.0, 0.5, 0.05)
    method = st.selectbox("Method", ["OTD", "ATD", "Sinkhorn"], index=0)
    proc = st.slider("Processes", 1, 8, 1, 1)

    st.header("Plotly layout / labels")
    seed = st.number_input("Stable layout seed", min_value=0, max_value=9999, value=42)
    node_label_size = st.slider("Ticker label font size", 6, 48, 14, 1)
    node_size_base = st.slider("Node base size", 8, 60, 24, 1)
    edge_width_scale = st.slider("Edge width scale", 0.5, 12.0, 5.0, 0.5)
    show_edge_weight_labels = st.checkbox("Show visible edge weights", value=True)
    edge_label_top_n = st.slider("Number of visible edge labels", 0, 150, 40, 5)
    frame_duration_ms = st.slider("Animation ms/frame", 100, 3000, 700, 100)

    st.header("Optional v9 layer")
    enable_flow_layer = st.checkbox("Enable Ricci flow layer", value=False)
    flow_iterations = st.slider("Flow iterations", 1, 40, 10, 1)
    flow_step = st.slider("Flow step size", 0.01, 1.00, 0.25, 0.01)
    normalize_flow = st.checkbox("Normalize mean edge distance", value=True)
    enable_surgery = st.checkbox("Apply financial surgery after flow", value=False)
    curvature_threshold = st.slider("Singular curvature threshold", -1.0, 0.0, -0.35, 0.05)
    distance_quantile = st.slider("Long-edge quantile", 0.50, 0.99, 0.80, 0.01)
    use_bridge_test = st.checkbox("Require bridge or long edge", value=True)

    st.header("HMM / capital proxy")
    enable_hmm = st.checkbox("Enable HMM regime detection", value=True)
    hmm_states = st.slider("HMM hidden states", 2, 5, 3, 1)
    hmm_forward_days = st.slider("Forward return days by regime", 1, 20, 5, 1)
    hmm_random_state = st.number_input("HMM random state", min_value=0, max_value=9999, value=42)
    flow_lookback = st.slider("Capital-flow proxy lookback", 5, 80, 20, 5)

if len(tickers) < 2:
    st.error("Please enter at least two tickers.")
    st.stop()

with st.spinner("Loading prices..."):
    if data_mode == "Synthetic lecture data":
        prices = make_demo_prices(tickers=tickers, n_days=int(n_days), seed=int(synthetic_seed), ipo_tickers=("QNT", "BNT"), ipo_start_frac=0.58)
    else:
        prices = download_prices(tuple(tickers), period, interval)

if prices.empty or prices.shape[1] < 2:
    st.error("No usable price data. Try fewer tickers or another period/interval.")
    st.stop()

returns = prices_to_returns(prices).dropna(axis=1, how="all")
if len(returns) < window_size:
    st.error(f"Not enough return rows ({len(returns)}) for window size {window_size}.")
    st.stop()

with st.expander("Ticker data availability diagnostics", expanded=False):
    st.dataframe(pd.DataFrame({"valid_return_rows": returns.notna().sum().sort_values()}), width="stretch")

with st.spinner("Computing v8-style rolling topology frames..."):
    frames, starts = build_rolling_frames(
        returns,
        window_size=int(window_size),
        step=int(step),
        max_frames=int(max_frames),
        max_distance=float(max_distance),
        min_abs_corr=float(min_abs_corr),
        keep_top_edges=keep_top_edges,
        min_node_obs=int(min_node_obs),
        min_pair_obs=int(min_pair_obs),
        edge_mode=str(edge_mode),
        knn_k=int(knn_k),
        positive_only=bool(positive_only),
        alpha=float(alpha),
        method=str(method),
        proc=int(proc),
    )

base_graph = build_base_graph_for_layout(frames, all_nodes=returns.columns)
positions = compute_stable_layout(base_graph, seed=int(seed))

if enable_hmm:
    with st.spinner("Fitting HMM hidden market regimes..."):
        hmm_feature_df, regime_names = compute_hmm_regimes(
            frames, returns, starts, int(window_size), int(hmm_states), int(hmm_forward_days), int(hmm_random_state)
        )
else:
    hmm_feature_df = pd.DataFrame()

fig = build_plotly_animation(
    frames,
    positions,
    frame_duration_ms=int(frame_duration_ms),
    node_label_size=int(node_label_size),
    node_size_base=int(node_size_base),
    edge_width_scale=float(edge_width_scale),
    title="Ricci Finance v10 - v8 Sparse Rolling Market Topology",
    show_edge_weight_labels=bool(show_edge_weight_labels),
    edge_label_top_n=int(edge_label_top_n),
)
st.plotly_chart(fig, width="stretch")

inspect_idx = st.slider("Inspect frame", 0, len(frames) - 1, len(frames) - 1, 1)
fd = frames[inspect_idx]

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Frame", f"{inspect_idx + 1}/{len(frames)}")
c2.metric("Date", fd.stats.end_date)
c3.metric("Avg Ricci", f"{fd.stats.avg_ricci:.4f}")
c4.metric("Clusters", fd.stats.num_clusters)
c5.metric("Largest comp.", fd.stats.largest_component)
c6.metric("Edges", fd.stats.num_edges)
c7.metric("Entropy", f"{fd.stats.graph_entropy:.3f}")

st.info(
    "Interpretation: the top animation is the v8-style raw market topology. Multiple clusters mean sparse correlation geometry is separating themes. "
    "Capital-flow proxy below is return-based, so AI/Quantum may receive capital even if Ricci topology does not merge them into one cluster."
)

left, right = st.columns([2, 1])
with left:
    st.subheader("Selected raw topology frame")
    st.plotly_chart(
        visualize_network(fd.G, positions=positions, title="Raw v8 topology frame", node_cluster=fd.node_cluster, show_edge_weight_labels=show_edge_weight_labels, edge_label_top_n=edge_label_top_n),
        width="stretch",
    )
with right:
    st.subheader("Capital-flow proxy")
    flow_df = sector_flow_table(returns.iloc[: starts[inspect_idx] + window_size], lookback=int(flow_lookback))
    st.dataframe(flow_df, width="stretch")

st.plotly_chart(plot_sector_flow(flow_df), width="stretch")

with st.expander("Raw edge table: distance / correlation / Ricci", expanded=True):
    st.dataframe(summarize_edges(fd.G), width="stretch")

features = rolling_feature_table(frames)
st.subheader("Rolling topology diagnostics")
st.plotly_chart(plot_rolling_features(features), width="stretch")
st.dataframe(features, width="stretch")

if enable_hmm and not hmm_feature_df.empty:
    with st.expander("HMM regime diagnostics", expanded=True):
        st.dataframe(hmm_feature_df, width="stretch")
        if "regime_name" in hmm_feature_df.columns:
            st.dataframe(hmm_feature_df.groupby(["hmm_state", "regime_name"]).size().reset_index(name="count"), width="stretch")

if enable_flow_layer:
    st.header("Optional v9 layer: Ricci flow and financial surgery on selected frame")
    st.markdown(
        "This layer starts from the selected v8 topology. Ricci flow deforms edge distances; optional surgery cuts singular bridge/long negative-curvature edges."
    )
    with st.spinner("Running Ricci flow on selected v8 frame..."):
        flowed_G, flow_history = run_ricci_flow(
            fd.G,
            iterations=int(flow_iterations),
            step_size=float(flow_step),
            alpha=float(alpha),
            method=str(method),
            proc=int(proc),
            normalize_mean_weight=bool(normalize_flow),
        )
    st.plotly_chart(plot_ricci_flow_history(flow_history), width="stretch")

    if enable_surgery:
        surgery = perform_financial_surgery(
            flowed_G,
            curvature_threshold=float(curvature_threshold),
            distance_quantile=float(distance_quantile),
            use_bridge_test=bool(use_bridge_test),
            remove_isolated_nodes=False,
        )
        after_G = surgery.after
        removed_edges = [(u, v) for u, v, _, _ in surgery.removed_edges]
    else:
        surgery = None
        after_G = flowed_G
        removed_edges = []

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Before flow: raw v8 topology**")
        st.plotly_chart(visualize_network(fd.G, positions=positions, title="Before flow", node_cluster=fd.node_cluster, show_edge_weight_labels=show_edge_weight_labels, edge_label_top_n=edge_label_top_n), width="stretch")
    with col_b:
        st.markdown("**After Ricci flow**")
        st.plotly_chart(visualize_network(flowed_G, positions=positions, title="After Ricci flow", node_cluster=compute_components(flowed_G), show_edge_weight_labels=show_edge_weight_labels, edge_label_top_n=edge_label_top_n), width="stretch")
    with col_c:
        st.markdown("**After optional surgery**")
        st.plotly_chart(visualize_network(after_G, positions=positions, title="After surgery" if enable_surgery else "Flow result", node_cluster=compute_components(after_G), highlight_edges=removed_edges, show_edge_weight_labels=show_edge_weight_labels, edge_label_top_n=edge_label_top_n), width="stretch")

    with st.expander("Ricci-flow edge comparison", expanded=True):
        st.dataframe(compare_before_after_flow(fd.G, flowed_G), width="stretch")
    if surgery is not None:
        with st.expander("Surgery report: removed singular edges", expanded=True):
            st.write("Before stats", surgery.before_stats)
            st.write("After stats", surgery.after_stats)
            st.dataframe(surgery.report, width="stretch")
else:
    st.warning("Ricci flow layer is disabled. Turn it on in the sidebar to run v9 diagnostics on this v8 topology frame.")

with st.expander("Correlation matrix for selected frame"):
    st.dataframe(fd.corr, width="stretch")
with st.expander("Financial distance matrix for selected frame"):
    st.dataframe(fd.dist, width="stretch")
