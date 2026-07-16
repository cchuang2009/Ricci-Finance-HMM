import numpy as np
import pandas as pd

from ricci_finance import *

close, volume, dollar_volume = make_demo_market_data(
    DEFAULT_TICKERS[:8],
    n_days=220,
)

returns = prices_to_returns(close)

dollar_volume = dollar_volume.reindex(
    index=returns.index,
    columns=returns.columns,
)

frames, starts = build_rolling_frames(
    returns,
    window_size=40,
    step=8,
    max_frames=16,
    graph_mode="knn+bridges",
    k_neighbors=2,
    max_bridges=2,
    dollar_volume=dollar_volume,
    use_capital_weighting=True,
    capital_alpha=0.35,
)

hmm_df, labels = compute_hmm_regimes(
    frames=frames,
    returns=returns,
    starts=starts,
    window_size=40,
    n_components=3,
)

assert frames
assert "state_probability" in hmm_df.columns
assert "hmm_state" in hmm_df.columns
assert "regime_name" in hmm_df.columns

# When hmmlearn is installed, posterior columns must be present.
# When it is unavailable, the documented fallback state must be returned.
posterior_columns = [
    column
    for column in hmm_df.columns
    if column.startswith("state_probability_")
]

if posterior_columns:
    assert hmm_df[posterior_columns].notna().any().any()
    assert all(
        "hmm_probabilities" in frame.metadata
        for frame in frames
    )
else:
    assert set(hmm_df["hmm_state"]) == {-1}
    assert "unavailable" in hmm_df["regime_name"].iloc[0].lower()

stories = build_frame_stories(frames)

assert len(stories) == len(frames)
assert stories[-1].narrative
assert "hmm_probability" in frames[-1].metadata

base = build_base_graph_for_layout(
    frames,
    all_nodes=returns.columns,
)

positions = compute_stable_layout_3d(base)

figure = build_3d_ricci_capital_animation(
    frames,
    positions,
    stories=stories,
)

assert len(figure.frames) == len(frames)
assert figure.layout.annotations

print("v12 Phase 3 HMM story smoke test passed")
