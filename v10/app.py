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
    compute_hmm_regimes,
    compute_stable_layout,
    download_prices,
    make_demo_prices,
    parse_tickers,
    plot_ricci_flow_history,
    plot_rolling_features,
    prices_to_returns,
    rolling_feature_table,
    run_ricci_flow,
    summarize_edges,
    visualize_network,
)


st.set_page_config(page_title="Ricci Finance v10", layout="wide")

st.title("Ricci Finance v10: Curvature, Ricci Flow, and Hidden Market Regimes")
st.caption(
    "Graduate lecture demo: convert correlations to financial geometry, measure Ricci curvature, "
    "then use Ricci flow to expose stable clusters, fragile bridges, and market-regime structure."
)

with st.sidebar:
    st.header("Data")
    data_mode = st.radio("Source", ["Synthetic lecture data", "yfinance live download"], index=0)
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
    min_node_obs = st.slider("Min observations to show node", 1, 30, 1, 1)
    min_pair_obs = st.slider("Min overlapping observations for edge", 3, 80, 4, 1)

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


if len(tickers) < 2:
    st.error("Please enter at least two tickers.")
    st.stop()

try:
    with st.spinner("Loading prices..."):
        if data_mode == "Synthetic lecture data":
            prices = make_demo_prices(tickers=tickers, n_days=int(n_days), seed=int(seed))
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

        **Relation mode tip:** use `threshold` to study only strong relations; use `knn+bridges` for a
        more sensible market map. In `threshold` mode, one isolated ticker creates an extra cluster.
        In `knn+bridges` mode, each ticker keeps nearest neighbors and a few weak cross-theme bridges.
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

st.subheader("1. Rolling Ricci-network observables")
st.plotly_chart(plot_rolling_features(feature_df), use_container_width=True)
st.dataframe(feature_df, use_container_width=True)

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
    use_container_width=True,
)

with st.expander("Edge table: distance, correlation, curvature", expanded=True):
    st.dataframe(summarize_edges(fd.G), use_container_width=True)

st.subheader("3. Ricci flow on selected frame")
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
    st.markdown("**Before flow**")
    st.plotly_chart(
        visualize_network(fd.G, positions=positions, title="Before Ricci flow", node_cluster=fd.node_cluster),
        use_container_width=True,
    )
with right:
    st.markdown("**After flow**")
    st.plotly_chart(
        visualize_network(flowed_G, positions=positions, title="After Ricci flow"),
        use_container_width=True,
    )

with st.expander("Before/after Ricci-flow edge comparison", expanded=True):
    comparison = compare_before_after_flow(fd.G, flowed_G)
    st.dataframe(comparison, use_container_width=True)
    if not comparison.empty:
        st.markdown(
            "**Reading the table:** negative `distance_change` means the link contracted under flow; "
            "positive `distance_change` means the link stretched and may act as a fragile bridge."
        )

if enable_hmm:
    st.subheader("4. HMM hidden market regimes from Ricci-network features")
    st.write("Regime names are unsupervised labels inferred from curvature, density, cluster structure, and forward return diagnostics.")
    st.dataframe(hmm_df, use_container_width=True)
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
        st.dataframe(summary, use_container_width=True)

st.subheader("5. Export data for lecture or research")
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
