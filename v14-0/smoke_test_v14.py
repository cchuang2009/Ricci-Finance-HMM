from ricci_finance import *

close, volume, dollar_volume = make_demo_market_data(DEFAULT_TICKERS[:8], n_days=240)
returns = prices_to_returns(close)
dollar_volume = dollar_volume.reindex(index=returns.index, columns=returns.columns)
frames, starts = build_rolling_frames(
    returns, window_size=40, step=8, max_frames=18,
    graph_mode="knn+bridges", k_neighbors=2, max_bridges=2,
    dollar_volume=dollar_volume, use_capital_weighting=True, capital_alpha=0.35,
)
features = rolling_feature_table(frames)
primary, output, labels, benchmark = run_regime_engine(
    features, "hmmlearn", n_components=3, decoding="viterbi"
)
assert len(primary) == len(frames)
if benchmark.available:
    attach_regime_to_frames(frames, primary, labels)
else:
    assert "unavailable" in primary["regime_name"].iloc[0]
stories = build_frame_stories(frames)
assert len(stories) == len(frames)
results, benchmark_table, consensus = compare_regime_engines(
    features, engines=("hmmlearn", "hsmm", "pomegranate"), n_components=3
)
assert set(results) == {"hmmlearn", "hsmm", "pomegranate"}
base = build_base_graph_for_layout(frames, all_nodes=returns.columns)
positions = compute_stable_layout_3d(base)
figure = build_3d_ricci_capital_animation(frames, positions, stories=stories)
assert len(figure.frames) == len(frames)
print("Ricci Finance v14 smoke test passed")
