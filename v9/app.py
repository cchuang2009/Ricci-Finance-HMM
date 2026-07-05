"""
app.py - Ricci Finance v9

Streamlit app in the v7 style, upgraded with v9 Ricci flow + Perelman-inspired financial surgery.

Run:
    streamlit run app.py
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

st.set_page_config(page_title="Ricci Finance v9", layout="wide")
st.title("Rolling Ricci Finance v9: Flow, Singularity, and Financial Surgery")
st.caption(
    "v9 keeps the v7 Plotly animation design, then adds Ricci flow, singular-edge detection, "
    "Perelman-inspired surgery, component entropy, and HMM hidden market regimes."
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
        n_days = st.slider("Synthetic trading days", 120, 900, 300, 20)
        synthetic_seed = st.number_input("Synthetic seed", min_value=0, max_value=9999, value=7)

    st.header("Rolling graph")
    window_size = st.slider("Rolling window size", 10, 180, 60, 5)
    step = st.slider("Frame step", 1, 30, 5, 1)
    max_frames = st.slider("Max animation frames", 5, 140, 45, 5)
    edge_mode = st.selectbox("Edge mode", ["threshold", "knn", "hybrid"], index=2)
    knn_k = st.slider("kNN neighbors", 1, 10, 3, 1)
    positive_only = st.checkbox("Positive correlation only", value=False)
    max_distance = st.slider("Max financial distance", 0.05, 2.00, 1.15, 0.05)
    min_abs_corr = st.slider("Minimum correlation threshold", 0.00, 1.00, 0.20, 0.05)
    keep_top_edges_raw = st.number_input("Keep top-N shortest threshold edges; 0 = no cap", 0, 3000, 0, 5)
    keep_top_edges = int(keep_top_edges_raw) if int(keep_top_edges_raw) > 0 else None
    min_node_obs = st.slider("Min observations to show node", 1, 30, 1, 1)
    min_pair_obs = st.slider("Min overlapping observations for edge", 3, 100, 4, 1)

    st.header("Ricci curvature")
    alpha = st.slider("Ollivier alpha", 0.0, 1.0, 0.5, 0.05)
    method = st.selectbox("Method", ["OTD", "ATD", "Sinkhorn"], index=0)
    proc = st.slider("Processes", 1, 8, 1, 1)

    st.header("Ricci flow")
    flow_iterations = st.slider("Flow iterations", 1, 40, 10, 1)
    flow_step = st.slider("Flow step size", 0.01, 1.00, 0.25, 0.01)
    normalize_flow = st.checkbox("Normalize mean edge distance", value=True)

    st.header("Surgery")
    surgery_on = st.checkbox("Enable financial surgery", value=True)
    surgery_on_flowed = st.checkbox("Apply surgery after Ricci flow", value=True)
    curvature_threshold = st.slider("Singular curvature threshold", -1.5, 0.0, -0.35, 0.05)
    distance_quantile = st.slider("Long-edge quantile", 0.10, 1.00, 0.80, 0.05)
    use_bridge_test = st.checkbox("Require bridge or long edge", value=True)
    remove_isolates = st.checkbox("Remove isolated nodes after surgery", value=False)

    st.header("HMM regime")
    enable_hmm = st.checkbox("Enable HMM", value=True)
    hmm_states = st.slider("Hidden states", 2, 5, 3, 1)
    hmm_forward_days = st.slider("Forward-return days", 1, 20, 5, 1)
    hmm_seed = st.number_input("HMM random state", min_value=0, max_value=9999, value=42)

    st.header("Layout / animation")
    layout_seed = st.number_input("Network layout seed", min_value=0, max_value=9999, value=42)
    node_label_size = st.slider("Ticker label font size", 6, 48, 14, 1)
    node_size_base = st.slider("Node base size", 8, 60, 24, 1)
    edge_width_scale = st.slider("Edge width scale", 0.5, 12.0, 5.0, 0.5)
    show_edge_weight_labels = st.checkbox("Show edge weight labels", value=True)
    edge_label_top_n = st.slider("Number of visible edge labels", 0, 120, 30, 5)
    frame_duration_ms = st.slider("Plotly animation ms/frame", 100, 3000, 700, 100)

    st.header("Capital-flow proxy")
    flow_lookback = st.slider("Sector flow lookback", 5, 120, 20, 5)

if len(tickers) < 2:
    st.error("Please enter at least two tickers.")
    st.stop()

with st.expander("What v9 adds: flow + cut + surgery", expanded=True):
    st.markdown(
        """
        **Curvature** measures local market geometry. **Ricci flow** deforms edge distances: positive-curvature
        links contract, while negative-curvature bridge links stretch. **Financial surgery** then cuts singular
        bridge edges after flow. In finance, this approximates removing unstable contagion channels to reveal
        whether the market is truly one coherent component or only connected through fragile speculative bridges.
        """
    )

try:
    with st.spinner("Loading prices..."):
        if data_mode == "Synthetic lecture data":
            prices = make_demo_prices(tickers=tickers, n_days=int(n_days), seed=int(synthetic_seed))
        else:
            prices = download_prices(tickers=tickers, period=period, interval=interval)
except Exception as exc:
    st.error(f"Price loading failed: {exc}")
    st.stop()

if prices.empty or prices.shape[1] < 2:
    st.error("No usable price data. Try synthetic data, fewer tickers, or another yfinance period/interval.")
    st.stop()

returns = prices_to_returns(prices).dropna(axis=1, how="all")
if len(returns) < window_size:
    st.error(f"Not enough return rows ({len(returns)}) for window size {window_size}.")
    st.stop()

with st.spinner("Building rolling Ricci frames..."):
    frames, starts = build_rolling_frames(
        returns=returns,
        window_size=int(window_size),
        step=int(step),
        max_frames=int(max_frames),
        max_distance=float(max_distance),
        min_abs_corr=float(min_abs_corr),
        keep_top_edges=keep_top_edges,
        alpha=float(alpha),
        method=str(method),
        proc=int(proc),
        min_node_obs=int(min_node_obs),
        min_pair_obs=int(min_pair_obs),
        edge_mode=str(edge_mode),
        knn_k=int(knn_k),
        positive_only=bool(positive_only),
    )

feature_df = rolling_feature_table(frames)
if enable_hmm:
    with st.spinner("Fitting HMM hidden market regimes..."):
        hmm_df, regime_names = compute_hmm_regimes(
            frames=frames,
            returns=returns,
            starts=starts,
            window_size=int(window_size),
            n_components=int(hmm_states),
            forward_days=int(hmm_forward_days),
            random_state=int(hmm_seed),
        )
else:
    hmm_df = pd.DataFrame()
    regime_names = {}

base_graph = build_base_graph_for_layout(frames, all_nodes=returns.columns)
positions = compute_stable_layout(base_graph, seed=int(layout_seed))

st.sidebar.success(f"Loaded {returns.shape[1]} tickers, {len(returns)} return rows, {len(frames)} frames")

missing_from_data = [t for t in tickers if t not in returns.columns]
if missing_from_data:
    st.warning("These requested tickers were not returned or usable: " + ", ".join(missing_from_data))

with st.expander("Ticker data availability diagnostics", expanded=False):
    avail_rows = returns.notna().sum().sort_values()
    st.dataframe(pd.DataFrame({"valid_return_rows": avail_rows}), use_container_width=True)

st.subheader("1. Rolling Ricci financial network animation")
fig_anim = build_plotly_animation(
    frames,
    positions,
    frame_duration_ms=int(frame_duration_ms),
    node_label_size=int(node_label_size),
    node_size_base=int(node_size_base),
    edge_width_scale=float(edge_width_scale),
    title="Rolling Ricci Financial Network v9 - Flow + Surgery Ready",
)
st.plotly_chart(fig_anim, use_container_width=True)

st.subheader("2. Rolling geometry observables")
st.plotly_chart(plot_rolling_features(feature_df), use_container_width=True)
st.dataframe(feature_df, use_container_width=True)

st.subheader("2b. Capital-flow proxy by theme")
st.markdown(
    "Ricci geometry is built from correlation-distance relations, not actual fund-flow data. "
    "This panel adds a separate return/breadth proxy, so AI or Quantum can show capital attraction "
    "even when they do not form one Ricci cluster."
)
sector_flow = sector_flow_table(returns, lookback=int(flow_lookback))
st.plotly_chart(plot_sector_flow(sector_flow), use_container_width=True)
st.dataframe(sector_flow, use_container_width=True)

st.subheader("3. Inspect one frame")
inspect_idx = st.slider("Inspect frame table", 0, len(frames) - 1, len(frames) - 1, 1)
fd = frames[inspect_idx]

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Frame", f"{inspect_idx + 1}/{len(frames)}")
c2.metric("Date", fd.stats.end_date)
c3.metric("Avg Ricci", f"{fd.stats.avg_ricci:.4f}")
c4.metric("Min Ricci", f"{fd.stats.ricci_min:.4f}")
c5.metric("Clusters", fd.stats.num_clusters)
c6.metric("Edges", fd.stats.num_edges)
c7.metric("Entropy", f"{fd.stats.graph_entropy:.3f}")

st.plotly_chart(
    visualize_network(fd.G, positions=positions, title=f"Selected frame before flow - {fd.stats.end_date}", node_cluster=fd.node_cluster),
    use_container_width=True,
)

with st.expander("Edge table: distance / correlation / Ricci", expanded=True):
    st.dataframe(summarize_edges(fd.G), use_container_width=True)

st.subheader("4. Ricci flow on selected frame")
flowed_G, flow_history = run_ricci_flow(
    fd.G,
    iterations=int(flow_iterations),
    step_size=float(flow_step),
    alpha=float(alpha),
    method=str(method),
    proc=int(proc),
    normalize_mean_weight=bool(normalize_flow),
)
st.plotly_chart(plot_ricci_flow_history(flow_history), use_container_width=True)

left, right = st.columns(2)
with left:
    st.markdown("**Before Ricci flow**")
    st.plotly_chart(visualize_network(fd.G, positions=positions, title="Before Ricci flow", node_cluster=fd.node_cluster), use_container_width=True)
with right:
    st.markdown("**After Ricci flow**")
    st.plotly_chart(visualize_network(flowed_G, positions=positions, title="After Ricci flow"), use_container_width=True)

flow_comparison = compare_before_after_flow(fd.G, flowed_G)
with st.expander("Before/after Ricci-flow edge comparison", expanded=True):
    st.dataframe(flow_comparison, use_container_width=True)
    st.info("Positive distance_change means the edge stretched under Ricci flow; these are candidates for unstable bridge behavior.")

st.subheader("5. Perelman-inspired financial surgery")
if surgery_on:
    surgery_base = flowed_G if surgery_on_flowed else fd.G
    surgery_result = perform_financial_surgery(
        surgery_base,
        curvature_threshold=float(curvature_threshold),
        distance_quantile=float(distance_quantile),
        use_bridge_test=bool(use_bridge_test),
        remove_isolated_nodes=bool(remove_isolates),
    )
    removed_pairs = [(u, v) for u, v, _, _ in surgery_result.removed_edges]
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Removed singular edges", len(surgery_result.removed_edges))
    s2.metric("Clusters before", int(surgery_result.before_stats["clusters"]))
    s3.metric("Clusters after", int(surgery_result.after_stats["clusters"]))
    s4.metric("Entropy after", f"{surgery_result.after_stats['component_entropy']:.3f}")

    st.markdown("**Singular-edge report**")
    st.dataframe(surgery_result.report, use_container_width=True)

    left2, right2 = st.columns(2)
    with left2:
        st.markdown("**Surgery input graph**")
        st.plotly_chart(visualize_network(surgery_result.before, positions=positions, title="Before surgery", highlight_edges=removed_pairs), use_container_width=True)
    with right2:
        st.markdown("**Post-surgery graph**")
        st.plotly_chart(visualize_network(surgery_result.after, positions=positions, title="After surgery"), use_container_width=True)
else:
    st.info("Financial surgery disabled in the sidebar.")

if enable_hmm:
    st.subheader("6. HMM hidden market regimes from Ricci-network features")
    st.write("Regime names are unsupervised labels inferred from curvature, entropy, density, and forward-return diagnostics.")
    st.dataframe(hmm_df, use_container_width=True)
    if not hmm_df.empty and "hmm_state" in hmm_df.columns:
        fwd_col = f"next_{hmm_forward_days}d_market_return"
        agg = {
            "avg_ricci": "mean",
            "ricci_min": "mean",
            "density": "mean",
            "component_entropy": "mean",
            "largest_component_ratio": "mean",
            "date": "count",
        }
        if fwd_col in hmm_df.columns:
            agg[fwd_col] = "mean"
        summary = hmm_df.groupby(["hmm_state", "regime_name"]).agg(agg).rename(columns={"date": "count"}).reset_index()
        st.dataframe(summary, use_container_width=True)

st.subheader("7. Export")
st.download_button("Download rolling feature table CSV", data=feature_df.to_csv(index=False).encode("utf-8"), file_name="ricci_rolling_features_v9.csv", mime="text/csv")
st.download_button("Download selected-frame edge table CSV", data=summarize_edges(fd.G).to_csv(index=False).encode("utf-8"), file_name="ricci_selected_frame_edges_v9.csv", mime="text/csv")
st.download_button("Download Ricci-flow comparison CSV", data=flow_comparison.to_csv(index=False).encode("utf-8"), file_name="ricci_flow_comparison_v9.csv", mime="text/csv")
if surgery_on:
    st.download_button("Download surgery report CSV", data=surgery_result.report.to_csv(index=False).encode("utf-8"), file_name="ricci_surgery_report_v9.csv", mime="text/csv")
