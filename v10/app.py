"""
app.py
Streamlit v8 app for: Perelman / Ollivier Ricci Theory to Finance.

Features
--------
1. Download market prices or use synthetic lecture data.
2. Build rolling correlation-distance financial networks.
3. Compute Ollivier-Ricci curvature on edges.
4. Run discrete Ricci flow on the selected frame.
5. Fit HMM hidden regimes from rolling Ricci-network features.

Run
---
pip install streamlit yfinance pandas numpy networkx plotly matplotlib scikit-learn hmmlearn
# Optional but recommended for true Ollivier-Ricci curvature:
pip install GraphRicciCurvature pot networkit
streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from helper import (
    DEFAULT_TICKERS,
    build_base_graph_for_layout,
    build_rolling_frames,
    compare_before_after_flow,
    compute_components,
    compute_hmm_regimes,
    compute_stable_layout,
    download_market_data,
    make_demo_market_data,
    parse_tickers,
    plot_ricci_flow_history,
    plot_rolling_features,
    plot_hmm_regimes,
    prices_to_returns,
    rolling_feature_table,
    run_ricci_flow,
    summarize_edges,
    visualize_network,
    animate_rolling_networks,
    capital_flow_table,
    node_capital_table,
    cluster_capital_table,
    plot_capital_flow_bars,
    compute_stable_layout_3d,
    build_3d_ricci_capital_animation,
    visualize_network_3d,
    surgery_risk_direction_table,
)


st.set_page_config(page_title="Ricci Finance v10", layout="wide")

st.title("Ricci Finance v10.4: Dynamic 3D Ricci-Capital Manifold")
st.caption(
    "Graduate lecture demo: convert correlations to financial geometry, measure Ricci curvature, "
    "then add dollar-volume mass to estimate capital transport among clusters."
)

with st.sidebar:
    st.header("Data")
    data_mode = st.radio("Source", ["yfinance live download", "Synthetic lecture data"], index=0)
    tickers_text = st.text_area("Tickers", value=", ".join(DEFAULT_TICKERS), height=120)
    tickers = parse_tickers(tickers_text)

    if data_mode == "yfinance live download":
        period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
        interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m"], index=0)
    else:
        n_days = st.slider("Synthetic trading days", 120, 800, 260, 20)
        seed = st.number_input("Synthetic seed", min_value=0, max_value=9999, value=7)

    st.header("Rolling network")
    window_size = st.slider("Rolling window size", 10, 160, 60, 5)
    step = st.slider("Frame step", 1, 30, 5, 1)
    max_frames = st.slider("Max frames", 5, 120, 40, 5)
    graph_mode = st.selectbox(
        "Graph relation mode",
        ["threshold", "knn", "knn+bridges"],
        index=2,
        help="threshold = only strong edges; knn = every ticker keeps nearest positive-correlation neighbors; knn+bridges = market-map view.",
    )
    k_neighbors = st.slider("kNN neighbors", 1, 8, 3, 1)
    min_corr = st.slider("Minimum positive correlation for kNN", -0.20, 0.50, 0.05, 0.01)
    max_bridges = st.slider("Bridge edges", 0, 10, 3, 1, help="Adds strongest weak positive cross-theme edges so clusters do not drift too far apart.")
    max_distance = st.slider("Max financial distance", 0.05, 2.00, 1.35, 0.05)
    min_abs_corr = st.slider("Minimum |correlation| for threshold mode", 0.00, 1.00, 0.10, 0.05)
    keep_top_edges_raw = st.number_input("Keep top-N shortest edges; 0 = no cap", 0, 2000, 0, 5)
    keep_top_edges = int(keep_top_edges_raw) if int(keep_top_edges_raw) > 0 else None
    min_node_obs = st.slider("Min observations to activate node", 1, 30, 1, 1)
    min_pair_obs = st.slider("Min overlapping observations for edge", 3, 80, 4, 1)

    st.header("Capital-flow manifold")
    use_capital_weighting = st.checkbox("Use dollar-volume capital weighting", value=True)
    capital_alpha = st.slider("Capital weighting strength", 0.00, 1.00, 0.35, 0.05,
                              help="0 = pure correlation distance; higher values make high dollar-volume synchronized edges closer.")

    st.header("Ricci curvature")
    alpha = st.slider("Ollivier alpha", 0.0, 1.0, 0.5, 0.05)
    method = st.selectbox("Method", ["OTD", "ATD", "Sinkhorn"], index=0)
    proc = st.slider("Processes", 1, 8, 1, 1)

    st.header("Ricci flow")
    flow_iterations = st.slider("Flow iterations", 1, 30, 8, 1)
    flow_step = st.slider("Flow step size", 0.01, 1.00, 0.25, 0.01)
    normalize_flow = st.checkbox("Normalize mean edge distance", value=True)

    st.header("HMM regimes")
    enable_hmm = st.checkbox("Enable HMM", value=True)
    hmm_states = st.slider("Hidden states", 2, 5, 3, 1)
    hmm_forward_days = st.slider("Forward-return days", 1, 20, 5, 1)
    hmm_seed = st.number_input("HMM random state", min_value=0, max_value=9999, value=42)

    st.header("Layout")
    layout_seed = st.number_input("Network layout seed", min_value=0, max_value=9999, value=42)
    layout_k = st.slider("Cluster spacing k", 0.15, 1.20, 0.45, 0.05, help="Lower values pull disconnected clusters closer; higher values spread them apart.")

    st.header("3D manifold")
    enable_3d = st.checkbox("Enable dynamic 3D Ricci-capital animation", value=True)
    z_mode = st.selectbox("3D z-axis", ["ricci_stress", "capital_mass", "spring"], index=0)
    frame_duration_ms = st.slider("3D animation frame duration (ms)", 200, 2000, 700, 100)
    edge_label_top_n_3d = st.slider("3D edge labels top-N", 0, 80, 25, 5)


if len(tickers) < 2:
    st.error("Please enter at least two tickers.")
    st.stop()

try:
    with st.spinner("Loading prices, volume, and dollar-volume..."):
        if data_mode == "Synthetic lecture data":
            prices, volumes, dollar_volume = make_demo_market_data(tickers=tickers, n_days=int(n_days), seed=int(seed))
        else:
            prices, volumes, dollar_volume = download_market_data(tickers=tickers, period=period, interval=interval)
except Exception as exc:
    st.error(f"Market-data loading failed: {exc}")
    st.stop()

if prices.empty or prices.shape[1] < 2:
    st.error("No usable price data. Try synthetic data, fewer tickers, or another yfinance period/interval.")
    st.stop()

returns = prices_to_returns(prices).dropna(axis=1, how="all")
dollar_volume = dollar_volume.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
if len(returns) < window_size:
    st.error(f"Not enough return rows ({len(returns)}) for window size {window_size}.")
    st.stop()

with st.expander("Lecture idea: how and why Ricci flow matters", expanded=True):
    st.markdown(
        """
        **Curvature** measures local network geometry. In finance, positive Ricci curvature usually means a
        coherent, redundant cluster: many nearby routes, shared factor exposure, and synchronized movement.
        Negative curvature often marks a bridge: a fragile edge connecting different clusters, where stress can
        transmit across sectors or themes.

        **Ricci flow** is the next step. It repeatedly updates edge distances by curvature:

        `new_distance = old_distance × (1 - step_size × Ricci_curvature)`

        Positive-curvature links contract; negative-curvature links stretch. For a graduate lecture, this is useful
        because it turns a static correlation graph into a geometric stress test: which clusters become more compact,
        which bridges become unstable, and whether the market geometry converges, fragments, or rotates.

        **Capital-flow correction:** correlation geometry shows shape, but dollar-volume gives market mass.
        The v10.1 edge distance can be capital-weighted, so high-liquidity synchronized links become closer
        and clusters can be read as capital basins. Pre-IPO tickers such as QNT are still displayed as inactive
        nodes before enough observations exist; their edges appear only after valid return/volume history exists.

        **Relation mode tip:** use `threshold` to study only strong relations; use `knn+bridges` for a
        more sensible market map.
        """
    )

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
        graph_mode=str(graph_mode),
        k_neighbors=int(k_neighbors),
        min_corr=float(min_corr),
        max_bridges=int(max_bridges),
        dollar_volume=dollar_volume,
        use_capital_weighting=bool(use_capital_weighting),
        capital_alpha=float(capital_alpha),
    )

feature_df = rolling_feature_table(frames)
if enable_hmm:
    with st.spinner("Fitting HMM regimes..."):
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
positions = compute_stable_layout(base_graph, seed=int(layout_seed), layout_k=float(layout_k))
positions_3d = compute_stable_layout_3d(base_graph, seed=int(layout_seed), layout_k=float(layout_k))


if enable_3d:
    st.subheader("Dynamic 3D Ricci-capital manifold")
    st.caption("x/y = stable topology, z = Ricci stress or selected mode, node size = dollar-volume capital mass, edge width = capital transport, edge color = Ricci curvature, animation = rolling-window evolution.")
    fig3d_anim = build_3d_ricci_capital_animation(
        frames=frames,
        positions_3d=positions_3d,
        frame_duration_ms=int(frame_duration_ms),
        title="Dynamic 3D Ricci-capital manifold",
        z_mode=str(z_mode),
        edge_label_top_n=int(edge_label_top_n_3d),
    )
    st.plotly_chart(fig3d_anim, width="stretch")

st.subheader("Animated rolling Ricci network")
st.caption("Play the rolling-window market geometry. Edge labels show current financial distance / weight.")
animated_fig = animate_rolling_networks(
    frames=frames,
    positions=positions,
    title="Rolling Ricci financial network",
)
st.plotly_chart(animated_fig, width="stretch")

st.subheader("1. Rolling Ricci-network observables")
st.plotly_chart(plot_rolling_features(feature_df), width="stretch")
st.dataframe(feature_df, width="stretch")

st.subheader("2. Inspect one rolling frame")
inspect_idx = st.slider("Frame", 0, len(frames) - 1, len(frames) - 1, 1)
fd = frames[inspect_idx]

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Date", fd.stats.end_date)
c2.metric("Avg Ricci", f"{fd.stats.avg_ricci:.4f}")
c3.metric("Ricci σ", f"{fd.stats.ricci_std:.4f}")
c4.metric("Clusters", fd.stats.num_clusters)
c5.metric("Edges", fd.stats.num_edges)
c6.metric("Density", f"{fd.stats.density:.3f}")

st.plotly_chart(
    visualize_network(
        fd.G,
        positions=positions,
        title=f"Ricci financial network — frame {inspect_idx + 1}, {fd.stats.end_date}",
        node_cluster=fd.node_cluster,
    ),
    width="stretch",
)

if enable_3d:
    st.plotly_chart(
        visualize_network_3d(
            fd.G,
            positions_3d=positions_3d,
            title=f"3D Ricci-capital manifold — frame {inspect_idx + 1}, {fd.stats.end_date}",
            node_cluster=fd.node_cluster,
            z_mode=str(z_mode),
        ),
        width="stretch",
    )

with st.expander("Edge table: distance, correlation, curvature, and capital flow", expanded=True):
    st.markdown(
        """
        **Column meaning**

        - `u`, `v`: the two tickers connected by this edge.
        - `distance`: the effective financial distance used by the graph. Smaller means stronger synchronized movement after optional capital weighting.
        - `raw_distance`: pure correlation distance before capital correction, when available.
        - `correlation`: rolling-window return correlation between the two tickers.
        - `ricciCurvature`: Ollivier-Ricci curvature. Positive means coherent/redundant local basin; negative means bridge-like stress channel.
        - `edge_capital_flow`: dollar-volume weighted transport intensity across the edge. Larger means more capital mass moving through that relation.
        - `capital_similarity`: similarity of the two tickers' dollar-volume masses. Higher values make the pair closer when capital weighting is enabled.
        - `edge_source`: whether the edge came from kNN, threshold, or bridge construction.
        """
    )
    st.dataframe(summarize_edges(fd.G), width="stretch")

st.subheader("3. Capital-flow transport among clusters")
flow_df = capital_flow_table(fd.G)
node_cap_df = node_capital_table(fd.G)
cluster_cap_df = cluster_capital_table(fd.G, fd.node_cluster)

st.plotly_chart(plot_capital_flow_bars(flow_df), width="stretch")
cc1, cc2, cc3 = st.columns(3)
with cc1:
    st.markdown("**Node market mass**")
    st.dataframe(node_cap_df, width="stretch")
with cc2:
    st.markdown("**Cluster capital basins**")
    st.dataframe(cluster_cap_df, width="stretch")
with cc3:
    st.markdown("**Top transport edges**")
    st.dataframe(flow_df.head(20), width="stretch")

st.subheader("4. Surgery-risk direction, not actual surgery")
st.caption("Real markets are usually on the path toward a regime, not at a final Perelman-resolved state. Keep the graph intact and rank possible future separation signals instead of cutting edges.")
st.dataframe(surgery_risk_direction_table(fd.G).head(30), width="stretch")

st.subheader("5. Ricci flow projection on selected frame")
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

st.caption(
    "Before/after Ricci flow is shown in 3D when enabled: x/y preserve the stable market topology, "
    "z shows Ricci stress or the selected 3D mode, node size shows capital mass, and edge color shows curvature."
)
left, right = st.columns(2)
with left:
    st.markdown("**Before flow — 3D view**" if enable_3d else "**Before flow — 2D fallback**")
    if enable_3d:
        st.plotly_chart(
            visualize_network_3d(
                fd.G,
                positions_3d=positions_3d,
                title="Before Ricci flow — 3D Ricci-capital manifold",
                node_cluster=fd.node_cluster,
                z_mode=str(z_mode),
            ),
            width="stretch",
        )
    else:
        st.plotly_chart(
            visualize_network(fd.G, positions=positions, title="Before Ricci flow", node_cluster=fd.node_cluster),
            width="stretch",
        )
with right:
    st.markdown("**After flow — 3D view**" if enable_3d else "**After flow — 2D fallback**")
    if enable_3d:
        st.plotly_chart(
            visualize_network_3d(
                flowed_G,
                positions_3d=positions_3d,
                title="After Ricci flow — 3D Ricci-capital manifold",
                node_cluster=compute_components(flowed_G),
                z_mode=str(z_mode),
            ),
            width="stretch",
        )
    else:
        st.plotly_chart(
            visualize_network(flowed_G, positions=positions, title="After Ricci flow"),
            width="stretch",
        )

with st.expander("Before/after Ricci-flow edge comparison", expanded=True):
    comparison = compare_before_after_flow(fd.G, flowed_G)
    st.dataframe(comparison, width="stretch")
    if not comparison.empty:
        st.markdown(
            "**Reading the table:** negative `delta_weight` means the link contracted under flow; "
            "positive `delta_weight` means the link stretched and may act as a fragile bridge."
        )

if enable_hmm:
    st.subheader("6. HMM hidden market regimes from Ricci + capital features")
    st.write("Regime names are unsupervised labels inferred from curvature, density, cluster structure, and forward return diagnostics.")
    st.dataframe(hmm_df, width="stretch")
    if not hmm_df.empty and "hmm_state" in hmm_df.columns:
        fwd_col = f"next_{hmm_forward_days}d_market_return"
        agg = {
            "avg_ricci": "mean",
            "density": "mean",
            "largest_component_ratio": "mean",
            "date": "count",
        }
        if fwd_col in hmm_df.columns:
            agg[fwd_col] = "mean"
        summary = hmm_df.groupby(["hmm_state", "regime_name"]).agg(agg).rename(columns={"date": "count"}).reset_index()
        st.dataframe(summary, width="stretch")

st.subheader("7. Export data for lecture or research")
st.download_button(
    "Download rolling feature table CSV",
    data=feature_df.to_csv(index=False).encode("utf-8"),
    file_name="ricci_rolling_features_v10.csv",
    mime="text/csv",
)
st.download_button(
    "Download selected-frame edge table CSV",
    data=summarize_edges(fd.G).to_csv(index=False).encode("utf-8"),
    file_name="ricci_selected_frame_edges_v10.csv",
    mime="text/csv",
)
st.download_button(
    "Download Ricci-flow comparison CSV",
    data=compare_before_after_flow(fd.G, flowed_G).to_csv(index=False).encode("utf-8"),
    file_name="ricci_flow_comparison_v10.csv",
    mime="text/csv",
)
st.download_button(
    "Download capital-flow edge table CSV",
    data=flow_df.to_csv(index=False).encode("utf-8"),
    file_name="capital_flow_edges_v10_1.csv",
    mime="text/csv",
)
st.download_button(
    "Download cluster-capital table CSV",
    data=cluster_cap_df.to_csv(index=False).encode("utf-8"),
    file_name="cluster_capital_basins_v10_1.csv",
    mime="text/csv",
)
