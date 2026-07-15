from __future__ import annotations

import numpy as np
import streamlit as st

from ricci_finance import *

st.set_page_config(
    page_title="Ricci Finance v12",
    layout="wide",
)

st.title("Ricci Finance v12 — HMM Story Manifold")
st.caption(
    "Each rolling frame combines Ricci geometry, capital transport, "
    "HMM posterior probabilities, and an evidence-based frame narrative."
)

with st.sidebar:
    source = st.radio(
        "Data source",
        ["Synthetic", "yfinance"],
        index=0,
    )

    tickers = parse_tickers(
        st.text_area(
            "Tickers",
            ", ".join(DEFAULT_TICKERS),
            height=120,
        )
    )

    window = st.slider(
        "Rolling window",
        20,
        160,
        60,
        5,
    )

    step = st.slider(
        "Frame step",
        1,
        30,
        5,
    )

    max_frames = st.slider(
        "Maximum frames",
        10,
        80,
        30,
        5,
    )

    graph_mode = st.selectbox(
        "Graph mode",
        ["threshold", "knn", "knn+bridges"],
        index=2,
    )

    k = st.slider(
        "kNN neighbors",
        1,
        8,
        3,
    )

    capital_alpha = st.slider(
        "Capital-distance weight",
        0.0,
        1.0,
        0.35,
        0.05,
    )

    hmm_states = st.slider(
        "HMM states",
        2,
        5,
        3,
    )

    z_mode = st.selectbox(
        "3D z-axis",
        ["ricci_stress", "capital_mass"],
    )

try:
    if source == "Synthetic":
        close, volume, dollar_volume = make_demo_market_data(
            tickers,
            n_days=320,
        )
    else:
        close, volume, dollar_volume = download_market_data(
            tickers
        )
except Exception as exc:
    st.error(str(exc))
    st.stop()

for warning in validate_market_data(
    close,
    volume,
    minimum_rows=window,
):
    st.warning(warning)

returns = prices_to_returns(close)

dollar_volume = dollar_volume.reindex(
    index=returns.index,
    columns=returns.columns,
)

frames, starts = build_rolling_frames(
    returns=returns,
    window_size=window,
    step=step,
    max_frames=max_frames,
    graph_mode=graph_mode,
    k_neighbors=k,
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

hmm_df, regime_names = compute_hmm_regimes(
    frames=frames,
    returns=returns,
    starts=starts,
    window_size=window,
    n_components=hmm_states,
    forward_days=5,
)

stories = build_frame_stories(frames)
story_df = frame_story_table(stories)

feature_df = rolling_feature_table(frames)
base_graph = build_base_graph_for_layout(
    frames,
    all_nodes=returns.columns,
)
positions_3d = compute_stable_layout_3d(
    base_graph,
    seed=42,
)

tabs = st.tabs(
    [
        "HMM story animation",
        "Selected-frame story",
        "HMM probabilities",
        "Rolling diagnostics",
        "Capital flow",
        "Ricci flow",
        "Surgery risk",
    ]
)

with tabs[0]:
    st.plotly_chart(
        build_3d_ricci_capital_animation(
            frames,
            positions_3d,
            z_mode=z_mode,
            stories=stories,
        ),
        width="stretch",
    )

    st.dataframe(
        story_df[
            [
                "frame",
                "date",
                "regime_name",
                "state_probability",
                "headline",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

selected = st.slider(
    "Selected frame",
    0,
    len(frames) - 1,
    len(frames) - 1,
)

selected_frame = frames[selected]
selected_story = stories[selected]

with tabs[1]:
    confidence = selected_story.state_probability

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Frame",
        selected + 1,
    )

    metric_columns[1].metric(
        "Window end",
        selected_story.date,
    )

    metric_columns[2].metric(
        "HMM regime",
        selected_story.regime_name,
    )

    metric_columns[3].metric(
        "Posterior confidence",
        (
            f"{confidence:.1%}"
            if np.isfinite(confidence)
            else "n/a"
        ),
    )

    st.subheader(selected_story.headline)
    st.write(selected_story.narrative)

    st.markdown("#### Changes from the previous frame")

    for change in selected_story.changes:
        st.write(f"• {change}")

    st.markdown(
        f"**Strongest fragile edge:** "
        f"{selected_story.strongest_fragile_edge}"
    )
    st.markdown(
        f"**Largest capital edge:** "
        f"{selected_story.strongest_capital_edge}"
    )

    st.plotly_chart(
        visualize_network_3d(
            selected_frame.G,
            positions_3d,
            title=(
                f"Frame {selected + 1}: "
                f"{selected_story.regime_name}"
            ),
            node_cluster=selected_frame.node_cluster,
            z_mode=z_mode,
        ),
        width="stretch",
    )

    st.dataframe(
        summarize_edges(selected_frame.G),
        width="stretch",
        hide_index=True,
    )

with tabs[2]:
    probability_columns = [
        column
        for column in hmm_df.columns
        if column.startswith("state_probability_")
    ]

    if probability_columns:
        probability_view = hmm_df[
            ["date", "hmm_state", "regime_name", "state_probability"]
            + probability_columns
        ].copy()

        st.line_chart(
            probability_view.set_index("date")[
                probability_columns
            ],
            width="stretch",
        )

        st.dataframe(
            probability_view,
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(
            "Posterior probabilities are unavailable because the "
            "HMM was not fitted."
        )

with tabs[3]:
    st.plotly_chart(
        plot_rolling_features(feature_df),
        width="stretch",
    )

    st.plotly_chart(
        plot_hmm_regimes(hmm_df),
        width="stretch",
    )

    st.dataframe(
        feature_df,
        width="stretch",
        hide_index=True,
    )

with tabs[4]:
    flow_df = capital_flow_table(
        selected_frame.G
    )

    st.plotly_chart(
        plot_capital_flow_bars(flow_df),
        width="stretch",
    )

    first, second, third = st.columns(3)

    with first:
        st.dataframe(
            flow_df,
            width="stretch",
            hide_index=True,
        )

    with second:
        st.dataframe(
            node_capital_table(selected_frame.G),
            width="stretch",
            hide_index=True,
        )

    with third:
        st.dataframe(
            cluster_capital_table(
                selected_frame.G,
                selected_frame.node_cluster,
            ),
            width="stretch",
            hide_index=True,
        )

with tabs[5]:
    flowed_graph, flow_history = run_ricci_flow(
        selected_frame.G,
        iterations=8,
        step_size=0.25,
        alpha=0.5,
        method="OTD",
        proc=1,
    )

    st.plotly_chart(
        plot_ricci_flow_history(flow_history),
        width="stretch",
    )

    before_column, after_column = st.columns(2)

    with before_column:
        st.plotly_chart(
            visualize_network_3d(
                selected_frame.G,
                positions_3d,
                "Before Ricci flow",
                selected_frame.node_cluster,
                z_mode,
            ),
            width="stretch",
        )

    with after_column:
        st.plotly_chart(
            visualize_network_3d(
                flowed_graph,
                positions_3d,
                "After Ricci flow",
                compute_components(flowed_graph),
                z_mode,
            ),
            width="stretch",
        )

    st.dataframe(
        compare_before_after_flow(
            selected_frame.G,
            flowed_graph,
        ),
        width="stretch",
        hide_index=True,
    )

with tabs[6]:
    st.info(
        "The table ranks possible separation directions. "
        "It does not cut the live market graph."
    )

    st.dataframe(
        surgery_risk_direction_table(
            selected_frame.G
        ),
        width="stretch",
        hide_index=True,
    )
