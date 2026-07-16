from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ricci_finance import *

st.set_page_config(page_title="Ricci Finance v14 Final", layout="wide")
st.title("Ricci Finance v14 Final — Ricci-Capital Regime Manifold")
st.caption(
    "Final yfinance research application: Ricci-capital rolling graphs, "
    "Viterbi/posterior HMM, duration-aware HSMM, optional pomegranate, consensus, and story animation."
)

with st.sidebar:
    st.markdown("### Market data")
    st.caption("The final application uses yfinance data only.")
    period = st.selectbox("History period", ["1y", "2y", "5y", "10y", "max"], index=2)
    interval = st.selectbox("Interval", ["1d", "1h"], index=0)
    tickers = parse_tickers(st.text_area("Tickers", ", ".join(DEFAULT_TICKERS), height=120))
    window = st.slider("Rolling window", 20, 160, 60, 5)
    step = st.slider("Frame step", 1, 30, 5)
    max_frames = st.slider("Maximum frames", 10, 80, 30, 5)
    graph_mode = st.selectbox("Graph mode", ["threshold", "knn", "knn+bridges"], index=2)
    k_neighbors = st.slider("kNN neighbors", 1, 8, 3)
    capital_alpha = st.slider("Capital-distance weight", 0.0, 1.0, 0.35, 0.05)
    n_states = st.slider("Regime states", 2, 5, 3)
    engine_choice = st.selectbox(
        "Primary regime engine",
        [
            "hmmlearn-viterbi",
            "hmmlearn-posterior",
            "hsmm",
            "pomegranate",
        ],
    )
    compare_all = st.checkbox("Compare all available engines", value=True)
    z_mode = st.selectbox("3D z-axis", ["ricci_stress", "capital_mass"])
    show_node_labels = st.checkbox("Show ticker names", value=True)
    show_edge_labels = st.checkbox("Show edge values", value=True)
    edge_label_metric = st.selectbox(
        "Edge value",
        ["ricci", "correlation", "distance", "capital_flow", "all"],
        index=0,
        format_func=lambda value: {
            "ricci": "Ricci curvature κ",
            "correlation": "Correlation ρ",
            "distance": "Effective distance d",
            "capital_flow": "Capital flow F",
            "all": "ρ + d + κ",
        }[value],
    )
    edge_label_limit = st.select_slider(
        "Number of labeled edges",
        options=[5, 10, 15, 20, 25, 30, 40, 50, "All"],
        value=20,
    )
    edge_label_top_n = (
        None if edge_label_limit == "All" else int(edge_label_limit)
    )

try:
    close, volume, dollar_volume = download_market_data(
        tickers,
        period=period,
        interval=interval,
    )
except Exception as exc:
    st.error(f"yfinance download failed: {exc}")
    st.stop()

for warning in validate_market_data(close, volume, minimum_rows=window):
    st.warning(warning)

returns = prices_to_returns(close)
dollar_volume = dollar_volume.reindex(index=returns.index, columns=returns.columns)

frames, starts = build_rolling_frames(
    returns=returns,
    window_size=window,
    step=step,
    max_frames=max_frames,
    graph_mode=graph_mode,
    k_neighbors=k_neighbors,
    min_corr=0.05,
    max_bridges=3,
    min_pair_obs=4,
    alpha=0.5,
    method="OTD",
    proc=1,
    dollar_volume=dollar_volume,
    use_capital_weighting=True,
    capital_alpha=capital_alpha,
)
feature_df = rolling_feature_table(frames)

if engine_choice.startswith("hmmlearn"):
    primary_name = "hmmlearn"
    decoding = "posterior" if engine_choice.endswith("posterior") else "viterbi"
else:
    primary_name = engine_choice
    decoding = "viterbi"

primary_df, primary_output, primary_labels, primary_benchmark = run_regime_engine(
    feature_df,
    primary_name,
    n_components=n_states,
    decoding=decoding,
)
attach_regime_to_frames(frames, primary_df, primary_labels)
stories = build_frame_stories(frames)
story_df = frame_story_table(stories)

base_graph = build_base_graph_for_layout(frames, all_nodes=returns.columns)
positions_3d = compute_stable_layout_3d(base_graph, seed=42)

comparison_tables = {}
benchmark_df = pd.DataFrame([vars(primary_benchmark)])
consensus_df = None
if compare_all:
    comparison_tables, benchmark_df, consensus_df = compare_regime_engines(
        feature_df,
        engines=("hmmlearn", "hsmm", "pomegranate"),
        n_components=n_states,
    )

main_tabs = st.tabs([
    "Story animation",
    "Selected frame",
    "Regime engines",
    "Decoder comparison",
    "Consensus",
    "Capital and Ricci",
    "Validation export",
])

with main_tabs[0]:
    st.plotly_chart(
        build_3d_ricci_capital_animation(
            frames,
            positions_3d,
            z_mode=z_mode,
            edge_label_metric=edge_label_metric,
            edge_label_top_n=edge_label_top_n,
            show_node_labels=show_node_labels,
            show_edge_labels=show_edge_labels,
            stories=stories,
            title=f"Ricci Finance v14 · {engine_choice}",
        ),
        width="stretch",
    )
    st.dataframe(
        story_df[["frame", "date", "regime_name", "state_probability", "headline"]],
        width="stretch",
        hide_index=True,
    )

selected = st.slider("Selected frame", 0, len(frames) - 1, len(frames) - 1)
selected_frame = frames[selected]
selected_story = stories[selected]

with main_tabs[1]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frame", selected + 1)
    c2.metric("Window end", selected_story.date)
    c3.metric("Regime", selected_story.regime_name)
    confidence = selected_story.state_probability
    c4.metric("Confidence", f"{confidence:.1%}" if np.isfinite(confidence) else "n/a")
    st.subheader(selected_story.headline)
    st.write(selected_story.narrative)
    for change in selected_story.changes:
        st.write(f"• {change}")
    st.plotly_chart(
        visualize_network_3d(
            selected_frame.G,
            positions_3d,
            title=f"Frame {selected + 1}: {selected_story.regime_name}",
            node_cluster=selected_frame.node_cluster,
            z_mode=z_mode,
        ),
        width="stretch",
    )
    st.dataframe(summarize_edges(selected_frame.G), width="stretch", hide_index=True)

with main_tabs[2]:
    st.subheader("Engine benchmark")
    st.dataframe(benchmark_df, width="stretch", hide_index=True)
    st.caption(
        "pomegranate is optional. When unavailable, its row reports the import/API error; "
        "the rest of v13 continues to run."
    )
    if comparison_tables:
        for engine_name, table in comparison_tables.items():
            with st.expander(engine_name, expanded=engine_name == primary_name):
                columns = [
                    column for column in [
                        "date", "regime_state", "regime_name", "state_probability"
                    ] if column in table.columns
                ]
                st.dataframe(table[columns], width="stretch", hide_index=True)

with main_tabs[3]:
    try:
        viterbi_df, _, _, v_benchmark = run_regime_engine(
            feature_df, "hmmlearn", n_components=n_states, decoding="viterbi"
        )
        posterior_df, _, _, p_benchmark = run_regime_engine(
            feature_df, "hmmlearn", n_components=n_states, decoding="posterior"
        )
        decoder_comparison = pd.DataFrame([
            {"decoder": "Viterbi", **vars(v_benchmark)},
            {"decoder": "Posterior argmax", **vars(p_benchmark)},
        ])
        agreement = float(np.mean(
            viterbi_df["regime_state"].to_numpy()
            == posterior_df["regime_state"].to_numpy()
        ))
        st.metric("Decoder agreement", f"{agreement:.1%}")
        st.dataframe(decoder_comparison, width="stretch", hide_index=True)
        decoder_states = pd.DataFrame({
            "date": feature_df["date"],
            "Viterbi": viterbi_df["regime_state"],
            "Posterior argmax": posterior_df["regime_state"],
        }).set_index("date")
        st.line_chart(decoder_states, width="stretch")
        st.info(
            "Use Viterbi for the main coherent story path; use posterior probabilities "
            "for confidence and early-transition warnings."
        )
    except Exception as exc:
        st.error(str(exc))

with main_tabs[4]:
    if consensus_df is None:
        st.info("Consensus requires at least two available engines.")
    else:
        st.metric(
            "Mean engine agreement",
            f"{consensus_df['consensus_agreement'].mean():.1%}",
        )
        st.line_chart(
            consensus_df.set_index(feature_df["date"])[
                ["consensus_state", "consensus_agreement"]
            ],
            width="stretch",
        )
        st.dataframe(consensus_df, width="stretch", hide_index=True)

with main_tabs[5]:
    flow_df = capital_flow_table(selected_frame.G)
    st.plotly_chart(plot_capital_flow_bars(flow_df), width="stretch")
    flowed_graph, flow_history = run_ricci_flow(
        selected_frame.G,
        iterations=8,
        step_size=0.25,
        alpha=0.5,
        method="OTD",
        proc=1,
    )
    st.plotly_chart(plot_ricci_flow_history(flow_history), width="stretch")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            visualize_network_3d(
                selected_frame.G, positions_3d, "Before Ricci flow",
                selected_frame.node_cluster, z_mode
            ),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            visualize_network_3d(
                flowed_graph, positions_3d, "After Ricci flow",
                compute_components(flowed_graph), z_mode
            ),
            width="stretch",
        )

with main_tabs[6]:
    export_df = feature_df.copy()
    export_df["primary_regime_state"] = primary_df["regime_state"]
    export_df["primary_regime_name"] = primary_df["regime_name"]
    export_df["primary_state_probability"] = primary_df["state_probability"]
    st.dataframe(export_df, width="stretch", hide_index=True)
    st.download_button(
        "Download v13 regime features CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="v13_regime_features.csv",
        mime="text/csv",
        width="stretch",
    )
