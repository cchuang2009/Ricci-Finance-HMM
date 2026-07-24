from __future__ import annotations

import pandas as pd

from .features import capital_feature_row, edge_jaccard, graph_feature_row
from .graph import (
    attach_capital_attributes,
    build_graph_from_window,
    compute_ricci_curvature,
)


def build_rolling_frames(
    returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    window: int = 63,
    step: int = 5,
    max_frames: int = 40,
    k: int = 4,
    min_corr: float = 0.2,
    max_bridges: int = 3,
    min_obs: int = 25,
    alpha: float = 0.5,
    curvature_engine: str = "ollivier_lp",
    capital_alpha: float = 0.35,
    use_capital: bool = True,
    progress=None,
):
    """Build rolling graph snapshots and their graph-level feature table."""
    starts = list(range(0, max(0, len(returns) - window + 1), step))
    if max_frames and len(starts) > max_frames:
        starts = starts[-max_frames:]

    frames = []
    rows = []
    previous_graph = None

    for index, start in enumerate(starts):
        window_returns = returns.iloc[start:start + window]
        window_dollar_volume = dollar_volume.reindex(window_returns.index)

        graph, corr, dist = build_graph_from_window(
            window_returns,
            k=k,
            min_corr=min_corr,
            max_bridges=max_bridges,
            min_obs=min_obs,
        )
        graph = attach_capital_attributes(
            graph,
            window_dollar_volume,
            capital_alpha=capital_alpha,
            use_capital=use_capital,
        )
        graph = compute_ricci_curvature(
            graph,
            alpha=alpha,
            engine=curvature_engine,
        )

        date = window_returns.index[-1]
        row = graph_feature_row(graph, date)
        row.update(capital_feature_row(graph))
        row["edge_stability"] = edge_jaccard(previous_graph, graph)
        row["curvature_engine"] = graph.graph.get(
            "curvature_engine", curvature_engine
        )

        frames.append({
            "graph": graph,
            "date": date,
            "correlation": corr,
            "distance": dist,
            "curvature_engine": graph.graph.get(
                "curvature_engine", curvature_engine
            ),
        })
        rows.append(row)
        previous_graph = graph

        if progress:
            progress(index + 1, len(starts), date)

    return frames, pd.DataFrame(rows)
